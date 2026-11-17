"""Load a job config into the generated JobSpec and check the rules in jobspec.proto.

The schema and its validation rules live in proto/trainmesh/v1/jobspec.proto; the Go CLI applies
the same rules, so this module only migrates legacy keys, parses, validates and applies defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import protovalidate
import yaml
from google.protobuf import json_format

from trainmesh.v1.jobspec_pb2 import JobSpec

from .errors import ConfigError

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_INPUT_SIZE = 32
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_OUTPUT_SIZE = 4
DEFAULT_OPTIMIZER = "adam"
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_CHECKPOINT_DIRECTORY = "./checkpoints"
DEFAULT_STRATEGY = "data_parallel"
# DiLoCo defaults: the paper's outer optimizer (Nesterov SGD, lr 0.7, momentum 0.9).
DEFAULT_INNER_STEPS = 100
DEFAULT_OUTER_LEARNING_RATE = 0.7
DEFAULT_OUTER_MOMENTUM = 0.9
DEFAULT_COMPRESSION = "int8"
DEFAULT_ROUND_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class DilocoSettings:
    inner_steps: int
    outer_learning_rate: float
    outer_momentum: float
    compression: str  # none, bf16 or int8
    machines_per_island: int
    min_islands: int  # 0 = majority of the islands
    round_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """JobSpec with every default applied: the trainer reads only this."""

    seed: int
    epochs: int
    per_rank_batch: int
    accumulation_steps: int
    rank_batch_sizes: tuple[int, ...]
    balance: str
    device: str
    communication: str
    precision: str
    gradient_sync: str
    dataset_type: str
    samples: int
    validation_samples: int
    input_size: int
    hidden_size: int
    output_size: int
    optimizer_name: str
    learning_rate: float
    checkpoint_enabled: bool
    checkpoint_directory: Path
    results_path: str
    resume: str
    strategy: str
    diloco: DilocoSettings


def load_config(path: str | Path) -> JobSpec:
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return parse_config(document)


def parse_config(document: Any) -> JobSpec:
    if not isinstance(document, dict):
        raise ConfigError("configuration must be a YAML mapping")
    _migrate_legacy_keys(document)
    spec = JobSpec()
    try:
        json_format.ParseDict(document, spec)
    except json_format.ParseError as exc:
        raise ConfigError(f"invalid job config: {exc}") from exc
    validate(spec)
    return spec


def validate(spec: JobSpec) -> None:
    try:
        protovalidate.validate(spec)
    except protovalidate.ValidationError as exc:
        problems = [
            f"{_field_path(violation.proto.field)}: {violation.proto.message}"
            for violation in exc.violations
        ]
        raise ConfigError("invalid job config:\n  " + "\n  ".join(problems)) from exc


def to_dict(spec: JobSpec) -> dict[str, Any]:
    """Plain-dict form stored in checkpoints and compared on resume."""
    return json_format.MessageToDict(spec, preserving_proto_field_name=True)


def training_settings(spec: JobSpec) -> TrainingSettings:
    training = spec.training
    model, dataset = training.model, training.dataset
    checkpoint = training.checkpoint
    return TrainingSettings(
        seed=spec.run.seed if spec.run.HasField("seed") else DEFAULT_SEED,
        epochs=training.epochs,
        per_rank_batch=training.batch.per_rank,
        accumulation_steps=training.batch.gradient_accumulation_steps or 1,
        rank_batch_sizes=tuple(training.batch.rank_batch_sizes),
        balance=training.batch.balance or "uniform",
        device=spec.execution.backend.device,
        communication=spec.execution.backend.communication,
        precision=training.precision or "fp32",
        gradient_sync=spec.execution.gradient_sync or "auto",
        dataset_type=dataset.type,
        samples=dataset.samples,
        validation_samples=training.validation.samples,
        input_size=model.input_size or dataset.input_size or DEFAULT_INPUT_SIZE,
        hidden_size=model.hidden_size or DEFAULT_HIDDEN_SIZE,
        output_size=model.output_size or dataset.output_size or DEFAULT_OUTPUT_SIZE,
        optimizer_name=training.optimizer.name or DEFAULT_OPTIMIZER,
        learning_rate=training.optimizer.learning_rate or DEFAULT_LEARNING_RATE,
        checkpoint_enabled=checkpoint.enabled if checkpoint.HasField("enabled") else True,
        checkpoint_directory=Path(checkpoint.directory or DEFAULT_CHECKPOINT_DIRECTORY),
        results_path=spec.run.results_path,
        resume=spec.run.resume,
        strategy=spec.execution.strategy or DEFAULT_STRATEGY,
        diloco=_diloco_settings(spec),
    )


def _diloco_settings(spec: JobSpec) -> DilocoSettings:
    diloco = spec.execution.diloco
    return DilocoSettings(
        inner_steps=diloco.inner_steps or DEFAULT_INNER_STEPS,
        outer_learning_rate=(
            diloco.outer_learning_rate
            if diloco.HasField("outer_learning_rate")
            else DEFAULT_OUTER_LEARNING_RATE
        ),
        outer_momentum=(
            diloco.outer_momentum if diloco.HasField("outer_momentum") else DEFAULT_OUTER_MOMENTUM
        ),
        compression=diloco.compression or DEFAULT_COMPRESSION,
        machines_per_island=diloco.machines_per_island or 1,
        min_islands=diloco.min_islands,
        round_timeout_seconds=diloco.round_timeout_seconds or DEFAULT_ROUND_TIMEOUT_SECONDS,
    )


def _migrate_legacy_keys(document: dict[str, Any]) -> None:
    legacy = document.pop("optimizer", None)
    if not isinstance(legacy, dict):
        return
    logger.warning(
        "config key 'optimizer' is deprecated: use 'planner', and 'run.results_path' for results"
    )
    document.setdefault("planner", {k: v for k, v in legacy.items() if k != "results"})
    if legacy.get("results"):
        document.setdefault("run", {}).setdefault("results_path", legacy["results"])


def _field_path(field_path: Any) -> str:
    return ".".join(element.field_name for element in field_path.elements) or "<message>"
