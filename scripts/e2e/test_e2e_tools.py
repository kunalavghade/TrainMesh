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
    )


@pytest.mark.parametrize(("global_batch", "world_size", "expected"), [(256, 4, 64), (256, 8, 32)])
def test_per_rank_batch_keeps_global_batch(
    global_batch: int, world_size: int, expected: int
) -> None:
    assert per_rank_batch(global_batch, world_size) == expected


def test_per_rank_batch_rejects_uneven_split() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        per_rank_batch(256, 3)


def test_render_splits_global_batch_across_all_machines() -> None:
    config = yaml.safe_load(render(parse_args(["--machines", "2", "--ranks-per-machine", "4"])))

    assert config["training"]["batch"]["per_rank"] == 32
    assert config["execution"]["processes_per_worker"] == 4
    assert config["training"]["checkpoint"]["enabled"] is False


def test_find_problems_accepts_matching_runs() -> None:
    scenarios = [
        _scenario("single"),
        _scenario("multi", machines=2, final_loss=1.0 + 1e-6),
    ]
    assert find_problems(scenarios, loss_tolerance=1e-3) == []


def test_find_problems_reports_divergence_and_inconsistent_replicas() -> None:
    scenarios = [
        _scenario("single"),
        _scenario("multi", machines=2, final_loss=1.5),
        _scenario("broken", consistent=False),
    ]

    problems = find_problems(scenarios, loss_tolerance=1e-3)

    assert any(p.startswith("multi: final loss") for p in problems)
    assert "broken: replicas diverged" in problems


def test_render_table_reports_speedup_and_per_machine_efficiency() -> None:
    table = render_table(
        [_scenario("single"), _scenario("multi", machines=2, samples_per_second=150.0)]
    )
    assert "| multi | 2 | 4 | 8 | 150 | 1.00 | 1.50x | 75% | 1.000000 |" in table


def test_main_exits_non_zero_when_losses_diverge(tmp_path: Path) -> None:
    def write(name: str, loss: float) -> Path:
        path = tmp_path / f"{name}.json"
        payload = {
            "world_size": 2,
            "samples_per_second": 10.0,
            "duration_seconds": 1.0,
            "final_loss": loss,
            "parameters_consistent": True,
            "topology": {"scenario": name, "machines": 1},
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    exit_code = main([str(write("a", 1.0)), str(write("b", 2.0))])

    assert exit_code == 1
    assert "Problems" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert load_scenario(tmp_path / "a.json").name == "a"
