"""Load and validate the v1 job configuration supported by the CPU/Gloo runtime."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

logger = logging.getLogger(__name__)

SUPPORTED_VERSION = 1
SUPPORTED_OPTIMIZERS = frozenset({"adam", "sgd"})


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Typed view of the `training` and `run` blocks used by the trainer."""

    seed: int
    epochs: int
    per_rank_batch: int
    accumulation_steps: int
    samples: int
    input_size: int
    hidden_size: int
    output_size: int
    optimizer_name: str
    learning_rate: float
    checkpoint_enabled: bool
    checkpoint_directory: Path
    results_path: str


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("version") != SUPPORTED_VERSION:
        raise ConfigError(f"configuration must be a YAML mapping with version: {SUPPORTED_VERSION}")
    _migrate_legacy_keys(config)
    _validate(config)
    return config


def training_settings(config: dict[str, Any]) -> TrainingSettings:
    run = config.get("run", {})
    training = config["training"]
    model = training.get("model", {})
    dataset = training["dataset"]
    optimizer = training.get("optimizer", {})
    batch = training["batch"]
    checkpoint = training.get("checkpoint", {})
    return TrainingSettings(
        seed=int(run.get("seed", 42)),
        epochs=int(training["epochs"]),
        per_rank_batch=int(batch["per_rank"]),
        accumulation_steps=int(batch.get("gradient_accumulation_steps", 1)),
        samples=int(dataset.get("samples", 4096)),
        input_size=int(model.get("input_size", dataset.get("input_size", 32))),
        hidden_size=int(model.get("hidden_size", 64)),
        output_size=int(model.get("output_size", dataset.get("output_size", 4))),
        optimizer_name=str(optimizer.get("name", "adam")).lower(),
        learning_rate=float(optimizer.get("learning_rate", 0.001)),
        checkpoint_enabled=bool(checkpoint.get("enabled", True)),
        checkpoint_directory=Path(checkpoint.get("directory", "./checkpoints")),
        results_path=str(run.get("results_path", "")),
    )


def _migrate_legacy_keys(config: dict[str, Any]) -> None:
    legacy = config.pop("optimizer", None)
    if not isinstance(legacy, dict):
        return
    logger.warning(
        "config key 'optimizer' is deprecated: use 'planner', and 'run.results_path' for results"
    )
    config.setdefault("planner", {k: v for k, v in legacy.items() if k != "results"})
    if legacy.get("results"):
        config.setdefault("run", {}).setdefault("results_path", legacy["results"])


def _validate(config: dict[str, Any]) -> None:
    execution = config.get("execution", {})
    if execution.get("strategy") != "data_parallel":
        raise ConfigError("only execution.strategy=data_parallel is implemented")
    backend = execution.get("backend", {})
    if backend.get("device", "cpu") != "cpu" or backend.get("communication", "gloo") != "gloo":
        raise ConfigError("this release supports CPU with Gloo only; CUDA/NCCL is not available")
    training = config.get("training", {})
    if training.get("precision", "fp32") != "fp32":
        raise ConfigError("only fp32 precision is currently supported")
    if training.get("dataset", {}).get("type") != "synthetic":
        raise ConfigError("only deterministic synthetic datasets are supported")
    batch = training.get("batch", {})
    if int(batch.get("per_rank", 0)) <= 0 or int(batch.get("gradient_accumulation_steps", 1)) <= 0:
        raise ConfigError(
            "training.batch.per_rank and gradient_accumulation_steps must be positive"
        )
    if int(training.get("epochs", 0)) <= 0:
        raise ConfigError("training.epochs must be positive")
    optimizer_name = str(training.get("optimizer", {}).get("name", "adam")).lower()
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        raise ConfigError(
            f"unsupported optimizer {optimizer_name!r}: use one of {sorted(SUPPORTED_OPTIMIZERS)}"
        )
