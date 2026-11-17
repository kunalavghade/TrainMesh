"""Unit tests for the e2e helper scripts (config generator and result comparison)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from bench_config import parse_args, per_rank_batch, render
from compare import Scenario, find_problems, load_scenario, main, render_table


def _scenario(
    name: str,
    *,
    machines: int = 1,
    samples_per_second: float = 100.0,
    final_loss: float = 1.0,
    consistent: bool = True,
    accuracy: float | None = 0.9,
) -> Scenario:
    return Scenario(
        name=name,
        machines=machines,
        ranks=2 * machines,
        cpus_total=4.0 * machines,
        environment="test",
        samples_per_second=samples_per_second,
        duration_seconds=1.0,
        final_loss=final_loss,
        parameters_consistent=consistent,
        validation_accuracy=accuracy,
    )


@pytest.mark.parametrize(("global_batch", "world_size", "expected"), [(256, 4, 64), (256, 8, 32)])
def test_per_rank_batch_keeps_global_batch(
    global_batch: int, world_size: int, expected: int
) -> None:
    assert per_rank_batch(global_batch, world_size) == expected


def test_per_rank_batch_rejects_uneven_split() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        per_rank_batch(256, 3)


def test_render_splits_global_batch_and_enables_validation() -> None:
    config = yaml.safe_load(render(parse_args(["--machines", "2", "--ranks-per-machine", "4"])))

    assert config["training"]["batch"]["per_rank"] == 32
    assert config["execution"]["processes_per_worker"] == 4
    assert config["execution"]["workers"] == 2
    assert config["training"]["dataset"]["type"] == "teacher"
    assert config["training"]["validation"]["samples"] == 4096
    assert config["training"]["checkpoint"]["enabled"] is False


def test_find_problems_accepts_matching_runs() -> None:
    scenarios = [_scenario("single"), _scenario("multi", machines=2, final_loss=1.0 + 1e-6)]
    assert find_problems(scenarios, loss_tolerance=1e-3, min_accuracy=0.85) == []


def test_find_problems_reports_divergence_and_inconsistent_replicas() -> None:
    scenarios = [
        _scenario("single"),
        _scenario("multi", machines=2, final_loss=1.5),
        _scenario("broken", consistent=False),
    ]

    problems = find_problems(scenarios, loss_tolerance=1e-3)

    assert any(p.startswith("multi: final loss") for p in problems)
    assert "broken: replicas diverged" in problems


def test_find_problems_requires_learning_and_equal_accuracy() -> None:
    scenarios = [
        _scenario("single", accuracy=0.30),
        _scenario("multi", machines=2, accuracy=0.40),
        _scenario("silent", accuracy=None),
    ]

    problems = find_problems(scenarios, loss_tolerance=1e-3, min_accuracy=0.85)

    assert "single: validation accuracy 0.3000 < 0.85" in problems
    assert any(p.startswith("multi: validation accuracy 0.4000 differs") for p in problems)
    assert "silent: no validation accuracy reported" in problems


def test_render_table_reports_speedup_efficiency_and_accuracy() -> None:
    table = render_table(
        [_scenario("single"), _scenario("multi", machines=2, samples_per_second=150.0)]
    )
    assert "| multi | 2 | 4 | 8 | 150 | 1.00 | 1.50x | 75% | 1.000000 | 90.00% |" in table


def test_main_exits_non_zero_when_losses_diverge(tmp_path: Path) -> None:
    def write(name: str, loss: float) -> Path:
        path = tmp_path / f"{name}.json"
        payload = {
            "world_size": 2,
            "samples_per_second": 10.0,
            "duration_seconds": 1.0,
            "final_loss": loss,
            "parameters_consistent": True,
            "validation": {"final_accuracy": 0.9},
            "topology": {"scenario": name, "machines": 1},
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    exit_code = main([str(write("a", 1.0)), str(write("b", 2.0))])

    assert exit_code == 1
    assert "Problems" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert load_scenario(tmp_path / "a.json").validation_accuracy == 0.9


def _wan_result(
    *,
    accuracy: float,
    seconds: float,
    utilization: float,
    island: int | None = None,
    members: list[int] | None = None,
) -> dict:
    result = {
        "world_size": 3 if island is None else 1,
        "global_step": 192,
        "duration_seconds": seconds,
        "samples_per_second": 98304 * 3 / seconds,
        "compute_utilization": utilization,
        "final_loss": 0.2,
        "model_sha256": "same",
        "validation": {"final_accuracy": accuracy},
    }
    if island is not None:
        result["diloco"] = {
            "island": island,
            "inner_steps": 64,
            "compression": "int8",
            "rounds": 3,
            "parameters": 76810,
            "bytes_sent": 80_000,
            "last_members": members or [0, 1, 2],
        }
    return result


def _wan_inputs(**overrides: object) -> tuple:
    ddp = _wan_result(accuracy=0.93, seconds=60.0, utilization=0.02)
    islands = [
        _wan_result(accuracy=0.925, seconds=4.0, utilization=0.8, island=i) for i in range(3)
    ]
    killed = {
        "state": "SUCCEEDED",
        "islands": [{"state": "FAILED"}, {"state": "SUCCEEDED"}, {"state": "SUCCEEDED"}],
        "result": _wan_result(
            accuracy=0.92, seconds=5.0, utilization=0.8, island=1, members=[1, 2]
        ),
    }
    values = {"ddp": ddp, "islands": islands, "killed": killed}
    values.update(overrides)
    return values["ddp"], values["islands"], values["killed"]


def test_wan_report_accepts_a_good_run() -> None:
    from wan_report import find_problems as wan_problems
    from wan_report import render as wan_render

    ddp, islands, killed = _wan_inputs()
    assert wan_problems(ddp, islands, killed, killed_island=0) == []
    summary = wan_render(
        ddp, islands, killed, {"description": "netem", "rtt_ms": 50.0, "mbit_per_s": 99.0}
    )
    assert "15.0x faster" in summary and "| diloco (H=64, int8)" in summary


def test_wan_report_flags_each_failure() -> None:
    from wan_report import ddp_bytes
    from wan_report import find_problems as wan_problems

    ddp, islands, killed = _wan_inputs()
    islands[2] = {**islands[2], "model_sha256": "other"}
    islands[0] = {**islands[0], "validation": {"final_accuracy": 0.80}, "compute_utilization": 0.3}
    killed = {**killed, "islands": [{"state": "FAILED"}] * 3, "state": "FAILED"}
    problems = " | ".join(wan_problems(ddp, islands, killed, killed_island=0))
    for expected in (
        "different models",
        "trails data_parallel",
        "utilization",
        "killed-island job",
    ):
        assert expected in problems
    # Ring all-reduce: 2 (n-1)/n of the fp32 gradient per rank per step.
    assert ddp_bytes({"world_size": 3, "global_step": 1}, 300) == int(2 * 2 / 3 * 1200 * 3)
