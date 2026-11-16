"""Compare TrainMesh benchmark results across topologies and check they trained the same model.

Each input is a scenario JSON written by an e2e script: the runtime's result plus a `topology`
block. Prints a markdown table, writes summary.md next to the inputs, and exits non-zero when a
run is inconsistent or final losses diverge.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOSS_TOLERANCE = 1e-3


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
    )


def find_problems(scenarios: list[Scenario], loss_tolerance: float) -> list[str]:
    problems = [f"{s.name}: replicas diverged" for s in scenarios if not s.parameters_consistent]
    problems += [
        f"{s.name}: final loss is not finite" for s in scenarios if not math.isfinite(s.final_loss)
    ]
    reference = scenarios[0]
    for scenario in scenarios[1:]:
        delta = abs(scenario.final_loss - reference.final_loss)
        if delta > loss_tolerance:
            problems.append(
                f"{scenario.name}: final loss {scenario.final_loss:.6f} differs from "
                f"{reference.name} {reference.final_loss:.6f} by {delta:.2e} (> {loss_tolerance:g})"
            )
    return problems


def render_table(scenarios: list[Scenario]) -> str:
    baseline = scenarios[0]
    lines = [
        (
            "| Scenario | Machines | Ranks | CPUs | Samples/s | Duration (s) | Speedup | "
            "Per-machine efficiency | Final loss |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in scenarios:
        speedup = s.samples_per_second / baseline.samples_per_second
        efficiency = speedup / (s.machines / baseline.machines)
        lines.append(
            f"| {s.name} | {s.machines} | {s.ranks} | {s.cpus_total:g} | "
            f"{s.samples_per_second:,.0f} | {s.duration_seconds:.2f} | {speedup:.2f}x | "
            f"{efficiency:.0%} | {s.final_loss:.6f} |"
        )
    environments = sorted({s.environment for s in scenarios})
    lines += [
        "",
        f"Baseline: `{baseline.name}`. Environment: {', '.join(environments)}.",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", nargs="+", type=Path, help="scenario JSON files; first is baseline"
    )
    parser.add_argument("--loss-tolerance", type=float, default=DEFAULT_LOSS_TOLERANCE)
    parser.add_argument("--summary", type=Path, help="markdown output (default: next to inputs)")
    args = parser.parse_args(argv)

    scenarios = [load_scenario(path) for path in args.results]
    table = render_table(scenarios)
    problems = find_problems(scenarios, args.loss_tolerance)
    report = table + ("\n\nProblems:\n" + "\n".join(f"- {p}" for p in problems) if problems else "")

    print(report)
    summary = args.summary or args.results[0].parent / "summary.md"
    summary.write_text(report + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
