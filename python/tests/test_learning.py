"""Config schema, datasets, validation and proof that the runtime actually learns."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from trainmesh_runtime.config import parse_config, training_settings  # noqa: E402
from trainmesh_runtime.data import (  # noqa: E402
    TEACHER_SEED_OFFSET,
    VALIDATION_SEED_OFFSET,
    build_validation_loader,
    make_dataset,
)
from trainmesh_runtime.distributed import DistContext  # noqa: E402
from trainmesh_runtime.errors import ConfigError  # noqa: E402
from trainmesh_runtime.evaluation import evaluate  # noqa: E402
from trainmesh_runtime.runtime import train  # noqa: E402

MIN_LEARNED_ACCURACY = 0.85


def _document(**training_overrides: object) -> dict:
    training = {
        "model": {"input_size": 32, "hidden_size": 64, "output_size": 4},
        "dataset": {"type": "teacher", "samples": 8192},
        "batch": {"per_rank": 128},
        "epochs": 3,
        "checkpoint": {"enabled": False},
        "validation": {"samples": 2048},
    }
    training.update(training_overrides)
    return {
        "version": 1,
        "run": {"seed": 7},
        "execution": {
            "strategy": "data_parallel",
            "workers": 1,
            "processes_per_worker": 1,
            "backend": {"device": "cpu", "communication": "gloo"},
        },
        "training": training,
    }


def _context(rank: int = 0, world_size: int = 1) -> DistContext:
    return DistContext(rank=rank, world_size=world_size, local_rank=rank, local_world_size=1)


def test_defaults_are_applied_from_the_schema() -> None:
    document = _document()
    document["training"]["model"] = {}
    settings = training_settings(parse_config(document))

    assert (settings.input_size, settings.hidden_size, settings.output_size) == (32, 64, 4)
    assert settings.optimizer_name == "adam"
    assert settings.accumulation_steps == 1
    assert settings.seed == 7


def test_rejects_unknown_fields_and_reports_rule_messages() -> None:
    with pytest.raises(ConfigError, match="no field named"):
        parse_config(_document(epoch=3))
    with pytest.raises(ConfigError, match="training.dataset.type must be synthetic or teacher"):
        parse_config(_document(dataset={"type": "imagenet", "samples": 10}))


def test_teacher_dataset_is_deterministic_and_balanced_enough_to_learn() -> None:
    settings = training_settings(parse_config(_document()))
    first = make_dataset(settings, samples=4096, seed_offset=VALIDATION_SEED_OFFSET)
    second = make_dataset(settings, samples=4096, seed_offset=VALIDATION_SEED_OFFSET)

    assert torch.equal(first.tensors[1], second.tensors[1])
    class_shares = torch.bincount(first.tensors[1], minlength=4).float() / 4096
    assert class_shares.min() > 0.05


def test_validation_shards_are_disjoint_and_cover_every_sample() -> None:
    settings = training_settings(parse_config(_document()))
    seen = []
    for rank in range(3):
        loader = build_validation_loader(settings, _context(rank, world_size=3))
        seen += list(loader.dataset.indices)

    assert sorted(seen) == list(range(settings.validation_samples))


def test_evaluate_reports_perfect_accuracy_for_the_teacher_itself() -> None:
    settings = training_settings(parse_config(_document()))
    loader = build_validation_loader(settings, _context())
    weights = torch.randn(
        settings.input_size,
        settings.output_size,
        generator=torch.Generator().manual_seed(settings.seed + TEACHER_SEED_OFFSET),
    )
    model = torch.nn.Linear(settings.input_size, settings.output_size, bias=False)
    with torch.no_grad():
        model.weight.copy_(weights.T)

    result = evaluate(model, loader, _context())

    assert result.accuracy == 1.0
    assert result.samples == settings.validation_samples


def test_training_on_teacher_data_learns(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_document()), encoding="utf-8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    result = train(str(config))

    history = result["validation"]["accuracy_history"]
    assert result["validation"]["final_accuracy"] >= MIN_LEARNED_ACCURACY
    assert history[-1] > history[0]
    assert result["epoch_losses"][-1] < result["epoch_losses"][0]
