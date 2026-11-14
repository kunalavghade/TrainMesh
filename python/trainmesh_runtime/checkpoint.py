"""Checkpoint format v2: safetensors for tensors, JSON for everything else, never pickle.

Layout of a checkpoint directory:
    model.safetensors      model state dict
    optimizer.safetensors  optimizer tensors, keyed ``state.<param_id>.<name>``
    rng.safetensors        torch CPU RNG state per rank, keyed ``torch_cpu.rank<r>``
    state.json             training config, optimizer param groups and scalar state
    metadata.json          format version, progress, world size and a SHA-256 per file
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .distributed import DistContext, all_gather_fixed
from .errors import CheckpointError

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2
HASH_CHUNK_BYTES = 8 * 1024 * 1024
MODEL_FILE = "model.safetensors"
OPTIMIZER_FILE = "optimizer.safetensors"
RNG_FILE = "rng.safetensors"
STATE_FILE = "state.json"
METADATA_FILE = "metadata.json"
COMPATIBILITY_KEYS = ("model", "dataset", "batch", "optimizer")


@dataclass(frozen=True, slots=True)
class Progress:
    epoch: int
    global_step: int


def save_checkpoint(
    destination: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: Progress,
    config: dict[str, Any],
    seed: int,
    ctx: DistContext,
) -> None:
    """Collective call: every rank contributes RNG state, only the primary rank writes."""
    rng_states = all_gather_fixed(torch.get_rng_state(), ctx)
    if not ctx.is_primary:
        return

    staging = destination.with_name(destination.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    optimizer_tensors, optimizer_json = _flatten_optimizer_state(optimizer.state_dict())
    save_file(_cpu_contiguous(model.state_dict()), staging / MODEL_FILE)
    save_file(optimizer_tensors, staging / OPTIMIZER_FILE)
    save_file({f"torch_cpu.rank{r}": s for r, s in enumerate(rng_states)}, staging / RNG_FILE)
    state = {"config": config, "optimizer": optimizer_json}
    (staging / STATE_FILE).write_text(json.dumps(state, default=str), encoding="utf-8")

    metadata = {
        "format_version": FORMAT_VERSION,
        "epoch": progress.epoch,
        "global_step": progress.global_step,
        "world_size": ctx.world_size,
        "seed": seed,
        "saved_at_unix": time.time(),
        "dataset": config["training"]["dataset"],
        "files": {name: sha256_file(staging / name) for name in _payload_files()},
    }
    (staging / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _replace_directory(staging, destination)


def load_checkpoint(
    source: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_config: dict[str, Any],
    ctx: DistContext,
) -> Progress:
    """Verify integrity and compatibility, then restore model, optimizer and this rank's RNG."""
    metadata = _read_verified_metadata(source)
    if int(metadata.get("world_size", -1)) != ctx.world_size:
        raise CheckpointError(
            f"checkpoint world size {metadata.get('world_size')} does not match this run "
            f"({ctx.world_size}); change membership through a planned restart"
        )

    state = json.loads((source / STATE_FILE).read_text(encoding="utf-8"))
    saved_training = state.get("config", {}).get("training", {})
    for key in COMPATIBILITY_KEYS:
        if saved_training.get(key) != training_config.get(key):
            raise CheckpointError(f"checkpoint training.{key} differs from current configuration")

    model.load_state_dict(load_file(source / MODEL_FILE))
    optimizer.load_state_dict(
        _unflatten_optimizer_state(load_file(source / OPTIMIZER_FILE), state["optimizer"])
    )
    rng_key = f"torch_cpu.rank{ctx.rank}"
    rng_states = load_file(source / RNG_FILE)
    if rng_key not in rng_states:
        raise CheckpointError(f"checkpoint has no RNG state for rank {ctx.rank}")
    torch.set_rng_state(rng_states[rng_key])
    return Progress(epoch=int(metadata["epoch"]), global_step=int(metadata["global_step"]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _read_verified_metadata(source: Path) -> dict[str, Any]:
    metadata_path = source / METADATA_FILE
    if not metadata_path.is_file():
        raise CheckpointError(f"checkpoint is incomplete: {metadata_path} is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    version = metadata.get("format_version", metadata.get("version"))
    if version != FORMAT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint format version {version!r} (expected {FORMAT_VERSION}); "
            "pickle-based v1 checkpoints are not loaded for safety, retrain to create a v2 one"
        )
    expected_files: dict[str, str] = metadata.get("files", {})
    for name in _payload_files():
        path = source / name
        if not path.is_file():
            raise CheckpointError(f"checkpoint is incomplete: {path} is missing")
        if expected_files.get(name) != sha256_file(path):
            raise CheckpointError(f"checkpoint checksum mismatch for {name}; refusing to resume")
    return metadata


def _payload_files() -> tuple[str, ...]:
    return (MODEL_FILE, OPTIMIZER_FILE, RNG_FILE, STATE_FILE)


def _cpu_contiguous(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: t.detach().cpu().contiguous() for name, t in tensors.items()}


def _flatten_optimizer_state(
    state_dict: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    scalars: dict[str, dict[str, Any]] = {}
    for param_id, param_state in state_dict["state"].items():
        for name, value in param_state.items():
            if isinstance(value, torch.Tensor):
                tensors[f"state.{param_id}.{name}"] = value.detach().cpu().contiguous()
            else:
                scalars.setdefault(str(param_id), {})[name] = value
    return tensors, {"param_groups": state_dict["param_groups"], "scalars": scalars}


def _unflatten_optimizer_state(
    tensors: dict[str, torch.Tensor], optimizer_json: dict[str, Any]
) -> dict[str, Any]:
    state: dict[int, dict[str, Any]] = {}
    for key, tensor in tensors.items():
        _, param_id, name = key.split(".", 2)
        state.setdefault(int(param_id), {})[name] = tensor
    for param_id, values in optimizer_json.get("scalars", {}).items():
        state.setdefault(int(param_id), {}).update(values)
    return {"state": state, "param_groups": optimizer_json["param_groups"]}


def _replace_directory(staging: Path, destination: Path) -> None:
    # Keep the last good checkpoint as `.previous` until the new one is in place.
    previous = destination.with_name(destination.name + ".previous")
    shutil.rmtree(previous, ignore_errors=True)
    if destination.exists():
        destination.rename(previous)
    staging.rename(destination)
    shutil.rmtree(previous, ignore_errors=True)
