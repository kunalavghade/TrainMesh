"""Compare TrainMesh benchmark results across topologies and check they trained the same model.

Each input is a scenario JSON written by an e2e script: the runtime's result plus a `topology`
block. Prints a markdown table, writes summary.md next to the inputs, and exits non-zero when
a run is inconsistent, final losses or validation accuracies diverge between topologies, or
validation accuracy stays below the learning threshold (proof that the model actually learned).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOSS_TOLERANCE = 1e-3
DEFAULT_ACCURACY_TOLERANCE = 0.005
DEFAULT_MIN_ACCURACY = 0.85


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    machines: int
    ranks: int
    cpus_total: float
    environment: str
    samples_per_second: float
    duration_seconds: float
    final_loss: float
    parameters_consistent: bool
    validation_accuracy: float | None


def load_scenario(path: Path) -> Scenario:
    data = json.loads(path.read_text(encoding="utf-8"))
    topology = data.get("topology", {})
    return Scenario(
        name=topology.get("scenario", path.stem),
        machines=int(topology.get("machines", 1)),
        ranks=int(data["world_size"]),
        cpus_total=float(topology.get("cpus_total", 0)),
        environment=topology.get("environment", "unknown"),
        samples_per_second=float(data["samples_per_second"]),
        duration_seconds=float(data["duration_seconds"]),
        final_loss=float(data["final_loss"]),
        parameters_consistent=bool(data.get("parameters_consistent", False)),
        validation_accuracy=(data.get("validation") or {}).get("final_accuracy"),
    )


def find_problems(
    scenarios: list[Scenario],
    loss_tolerance: float,
    accuracy_tolerance: float = DEFAULT_ACCURACY_TOLERANCE,
    min_accuracy: float | None = None,
) -> list[str]:
    problems = [f"{s.name}: replicas diverged" for s in scenarios if not s.parameters_consistent]
    problems += [
        f"{s.name}: final loss is not finite" for s in scenarios if not math.isfinite(s.final_loss)
    ]
    problems += _accuracy_problems(scenarios, accuracy_tolerance, min_accuracy)
    reference = scenarios[0]
    for scenario in scenarios[1:]:
        delta = abs(scenario.final_loss - reference.final_loss)
        if delta > loss_tolerance:
            problems.append(
                f"{scenario.name}: final loss {scenario.final_loss:.6f} differs from "
                f"{reference.name} {reference.final_loss:.6f} by {delta:.2e} (> {loss_tolerance:g})"
            )
    return problems


def _accuracy_problems(
    scenarios: list[Scenario], tolerance: float, min_accuracy: float | None
) -> list[str]:
    measured = [s for s in scenarios if s.validation_accuracy is not None]
    problems = []
    if min_accuracy is not None:
        problems += [
            f"{s.name}: no validation accuracy reported"
            for s in scenarios
            if s.validation_accuracy is None
        ]
        problems += [
            f"{s.name}: validation accuracy {s.validation_accuracy:.4f} < {min_accuracy:.2f}"
            for s in measured
            if s.validation_accuracy < min_accuracy
        ]
    if len(measured) > 1:
        reference = measured[0]
        problems += [
            f"{s.name}: validation accuracy {s.validation_accuracy:.4f} differs from "
            f"{reference.name} {reference.validation_accuracy:.4f} (> {tolerance:g})"
            for s in measured[1:]
            if abs(s.validation_accuracy - reference.validation_accuracy) > tolerance
        ]
    return problems


def render_table(scenarios: list[Scenario]) -> str:
    baseline = scenarios[0]
    lines = [
        (
            "| Scenario | Machines | Ranks | CPUs | Samples/s | Duration (s) | Speedup | "
            "Per-machine efficiency | Final loss | Val accuracy |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in scenarios:
        speedup = s.samples_per_second / baseline.samples_per_second
        efficiency = speedup / (s.machines / baseline.machines)
        lines.append(
            f"| {s.name} | {s.machines} | {s.ranks} | {s.cpus_total:g} | "
            f"{s.samples_per_second:,.0f} | {s.duration_seconds:.2f} | {speedup:.2f}x | "
            f"{efficiency:.0%} | {s.final_loss:.6f} | {_percent(s.validation_accuracy)} |"
        )
    environments = sorted({s.environment for s in scenarios})
    lines += [
        "",
        f"Baseline: `{baseline.name}`. Environment: {', '.join(environments)}.",
    ]
    return "\n".join(lines)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", nargs="+", type=Path, help="scenario JSON files; first is baseline"
    )
    parser.add_argument("--loss-tolerance", type=float, default=DEFAULT_LOSS_TOLERANCE)
    parser.add_argument("--accuracy-tolerance", type=float, default=DEFAULT_ACCURACY_TOLERANCE)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=DEFAULT_MIN_ACCURACY,
        help="required final validation accuracy; pass a negative value to skip",
    )
    parser.add_argument("--summary", type=Path, help="markdown output (default: next to inputs)")
    args = parser.parse_args(argv)

    scenarios = [load_scenario(path) for path in args.results]
    table = render_table(scenarios)
    min_accuracy = args.min_accuracy if args.min_accuracy >= 0 else None
    problems = find_problems(scenarios, args.loss_tolerance, args.accuracy_tolerance, min_accuracy)
    report = table + ("\n\nProblems:\n" + "\n".join(f"- {p}" for p in problems) if problems else "")

    print(report)
    summary = args.summary or args.results[0].parent / "summary.md"
    summary.write_text(report + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
