"""Checkpoint format v2: safe round trip, integrity and compatibility checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from trainmesh_runtime.checkpoint import (  # noqa: E402
    METADATA_FILE,
    MODEL_FILE,
    Progress,
    load_checkpoint,
    save_checkpoint,
)
from trainmesh_runtime.distributed import DistContext  # noqa: E402
from trainmesh_runtime.errors import CheckpointError  # noqa: E402
from trainmesh_runtime.runtime import MLP  # noqa: E402

SINGLE_RANK = DistContext(rank=0, world_size=1, local_rank=0, local_world_size=1)
CONFIG: dict[str, Any] = {
    "version": 1,
    "training": {
        "model": {"input_size": 4, "hidden_size": 8, "output_size": 2},
        "dataset": {"type": "synthetic", "samples": 16},
        "batch": {"per_rank": 4, "gradient_accumulation_steps": 1},
        "optimizer": {"name": "adam", "learning_rate": 0.01},
    },
}


def _trained(optimizer_name: str = "adam") -> tuple[MLP, torch.optim.Optimizer]:
    torch.manual_seed(1)
    model = MLP(4, 8, 2)
    factory = torch.optim.Adam if optimizer_name == "adam" else torch.optim.SGD
    optimizer = factory(model.parameters(), lr=0.01)
    for _ in range(3):
        loss = model(torch.randn(4, 4)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model, optimizer


def _save(directory: Path, model: MLP, optimizer: torch.optim.Optimizer) -> Path:
    destination = directory / "latest"
    save_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        progress=Progress(epoch=2, global_step=7),
        config=CONFIG,
        seed=42,
        ctx=SINGLE_RANK,
    )
    return destination


def _load_into_fresh(source: Path, training_config: dict[str, Any] | None = None) -> Progress:
    model = MLP(4, 8, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return load_checkpoint(
        source,
        model=model,
        optimizer=optimizer,
        training_config=training_config or CONFIG["training"],
        ctx=SINGLE_RANK,
    )


@pytest.mark.parametrize("optimizer_name", ["adam", "sgd"])
def test_round_trip_restores_model_optimizer_and_rng(tmp_path: Path, optimizer_name: str) -> None:
    model, optimizer = _trained(optimizer_name)
    source = _save(tmp_path, model, optimizer)
    expected_next_random = torch.rand(3)

    torch.manual_seed(999)
    restored = MLP(4, 8, 2)
    factory = torch.optim.Adam if optimizer_name == "adam" else torch.optim.SGD
    restored_optimizer = factory(restored.parameters(), lr=0.01)
    progress = load_checkpoint(
        source,
        model=restored,
        optimizer=restored_optimizer,
        training_config=CONFIG["training"],
        ctx=SINGLE_RANK,
    )

    assert progress == Progress(epoch=2, global_step=7)
    torch.testing.assert_close(torch.rand(3), expected_next_random)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)
    original_state = optimizer.state_dict()["state"]
    for param_id, values in restored_optimizer.state_dict()["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, original_state[param_id][key])
    assert not list(source.glob("*.pt")), "checkpoint must not contain pickle files"


def test_rejects_tampered_tensor_file(tmp_path: Path) -> None:
    source = _save(tmp_path, *_trained())
    payload = bytearray((source / MODEL_FILE).read_bytes())
    payload[-1] ^= 0xFF
    (source / MODEL_FILE).write_bytes(bytes(payload))

    with pytest.raises(CheckpointError, match="checksum mismatch"):
        _load_into_fresh(source)


def test_rejects_legacy_pickle_format(tmp_path: Path) -> None:
    source = _save(tmp_path, *_trained())
    metadata = json.loads((source / METADATA_FILE).read_text(encoding="utf-8"))
    metadata.pop("format_version")
    metadata["version"] = 1
    (source / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(CheckpointError, match="format version 1"):
        _load_into_fresh(source)


def test_rejects_incompatible_training_config(tmp_path: Path) -> None:
    source = _save(tmp_path, *_trained())
    changed = copy.deepcopy(CONFIG["training"])
    changed["batch"]["per_rank"] = 8

    with pytest.raises(CheckpointError, match="training.batch differs"):
        _load_into_fresh(source, changed)


def test_rejects_world_size_mismatch(tmp_path: Path) -> None:
    source = _save(tmp_path, *_trained())
    two_ranks = DistContext(rank=0, world_size=2, local_rank=0, local_world_size=2)
    model = MLP(4, 8, 2)

    with pytest.raises(CheckpointError, match="world size"):
        load_checkpoint(
            source,
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
            training_config=CONFIG["training"],
            ctx=two_ranks,
        )


def test_overwrite_keeps_single_latest_directory(tmp_path: Path) -> None:
    model, optimizer = _trained()
    _save(tmp_path, model, optimizer)
    _save(tmp_path, model, optimizer)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["latest"]
