"""Judge the WAN e2e: data-parallel vs DiLoCo over the same emulated link, plus a lost island.

Inputs are the result JSON of the data-parallel job, every island leader's result of the DiLoCo
job, and `trainmesh job get` of the DiLoCo job whose root island was killed. Writes a markdown
summary and exits 1 with the list of problems if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MIN_ACCURACY = 0.85
# DiLoCo may trail per-step synchronisation slightly at equal samples.
ACCURACY_TOLERANCE = 0.02
MIN_UTILIZATION = 0.60
BYTES_PER_FP32 = 4


def ddp_bytes(result: dict[str, Any], parameters: int) -> int:
    """Bytes all ranks send in a ring all-reduce of fp32 gradients, once per optimizer step."""
    world = result["world_size"]
    per_rank_per_step = 2 * (world - 1) / world * parameters * BYTES_PER_FP32
    return int(per_rank_per_step * world * result["global_step"])


def diloco_bytes(islands: list[dict[str, Any]]) -> int:
    return sum(result["diloco"]["bytes_sent"] for result in islands)


def find_problems(
    ddp: dict[str, Any],
    islands: list[dict[str, Any]],
    killed: dict[str, Any],
    killed_island: int,
    *,
    min_utilization: float = MIN_UTILIZATION,
) -> list[str]:
    problems: list[str] = []
    ddp_accuracy = ddp["validation"]["final_accuracy"]
    diloco_accuracy = islands[0]["validation"]["final_accuracy"]
    for name, accuracy in (("data_parallel", ddp_accuracy), ("diloco", diloco_accuracy)):
        if accuracy < MIN_ACCURACY:
            problems.append(f"{name} accuracy {accuracy:.4f} < {MIN_ACCURACY}")
    if diloco_accuracy < ddp_accuracy - ACCURACY_TOLERANCE:
        problems.append(
            f"diloco accuracy {diloco_accuracy:.4f} trails data_parallel {ddp_accuracy:.4f} "
            f"by more than {ACCURACY_TOLERANCE}"
        )
    if len({result["model_sha256"] for result in islands}) != 1:
        problems.append("diloco islands finished with different models")
    everyone = list(range(len(islands)))
    for result in islands:
        if result["diloco"]["last_members"] != everyone:
            problems.append(
                f"island {result['diloco']['island']} last round had "
                f"{result['diloco']['last_members']}, want {everyone}"
            )
    if islands[0]["duration_seconds"] >= ddp["duration_seconds"]:
        problems.append("diloco was not faster than data_parallel over the WAN")
    utilization = islands[0]["compute_utilization"] or 0.0
    if utilization < min_utilization:
        problems.append(f"diloco compute utilization {utilization:.2f} < {min_utilization}")

    states = [island["state"] for island in killed.get("islands", [])]
    expected = ["FAILED" if i == killed_island else "SUCCEEDED" for i in range(len(states))]
    if killed.get("state") != "SUCCEEDED" or states != expected:
        problems.append(
            f"killed-island job {killed.get('state')} with islands {states}, "
            f"want SUCCEEDED with {expected}"
        )
    else:
        result = killed["result"]
        survivors = [i for i in range(len(states)) if i != killed_island]
        if result["diloco"]["last_members"] != survivors:
            problems.append(
                f"after the kill the last round had {result['diloco']['last_members']}"
                f", want {survivors}"
            )
        if result["validation"]["final_accuracy"] < MIN_ACCURACY:
            problems.append("the surviving islands did not reach the accuracy floor")
    return problems


def render(
    ddp: dict[str, Any],
    islands: list[dict[str, Any]],
    killed: dict[str, Any],
    network: dict[str, Any],
) -> str:
    parameters = islands[0]["diloco"]["parameters"]
    outer = islands[0]["diloco"]
    rows = [
        ("data_parallel (sync every step)", ddp, ddp_bytes(ddp, parameters)),
        (
            f"diloco (H={outer['inner_steps']}, {outer['compression']})",
            islands[0],
            diloco_bytes(islands),
        ),
    ]
    lines = [
        f"Emulated link: {network.get('description', '?')}; measured RTT "
        f"{network.get('rtt_ms', float('nan')):.1f} ms, throughput "
        f"{network.get('mbit_per_s', float('nan')):.0f} Mbit/s.",
        "",
        "| Strategy | Wall time (s) | Samples/s | Compute utilization | Bytes between machines "
        "| Val accuracy | Final loss |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, result, sent in rows:
        lines.append(
            f"| {name} | {result['duration_seconds']:.1f} | {result['samples_per_second']:,.0f} "
            f"| {_percent(result['compute_utilization'])} | {sent / 1e6:,.1f} MB "
            f"| {result['validation']['final_accuracy']:.2%} | {result['final_loss']:.4f} |"
        )
    speedup = ddp["duration_seconds"] / max(islands[0]["duration_seconds"], 1e-9)
    traffic = ddp_bytes(ddp, parameters) / max(diloco_bytes(islands), 1)
    lines += [
        "",
        f"DiLoCo was {speedup:.1f}x faster and moved {traffic:.0f}x fewer bytes; "
        f"{parameters:,} parameters, {islands[0]['diloco']['rounds']} outer rounds.",
    ]
    if killed.get("result"):
        survivor = killed["result"]
        lines.append(
            f"Root island killed mid-run: job {killed['state']}, islands "
            f"{[i['state'] for i in killed['islands']]}, last round members "
            f"{survivor['diloco']['last_members']}, accuracy "
            f"{survivor['validation']['final_accuracy']:.2%}."
        )
    return "\n".join(lines) + "\n"


def _percent(value: float | None) -> str:
    """Utilization is None when DDP hid communication inside backward()."""
    return "n/a" if value is None else f"{value:.0%}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddp", type=Path, required=True)
    parser.add_argument("--diloco", type=Path, nargs="+", required=True, help="island leaders")
    parser.add_argument("--killed-job", type=Path, required=True)
    parser.add_argument("--killed-island", type=int, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--min-utilization", type=float, default=MIN_UTILIZATION)
    args = parser.parse_args(argv)

    def load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    ddp, killed, network = load(args.ddp), load(args.killed_job), load(args.network)
    islands = sorted((load(path) for path in args.diloco), key=lambda r: r["diloco"]["island"])
    summary = render(ddp, islands, killed, network)
    args.summary.write_text(summary, encoding="utf-8")
    print(summary)
    problems = find_problems(
        ddp, islands, killed, args.killed_island, min_utilization=args.min_utilization
    )
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
