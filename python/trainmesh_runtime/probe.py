"""Report this machine's training capabilities as JSON; used by `trainmesh doctor` and tm-agent.

`--benchmark` also times real training steps of a fixed reference model on each offered device,
so the coordinator can make a first split of batches by measured speed instead of guessing from
specs. Each step runs like a step in a mixed-device mesh group: the batch is copied from host
memory and, on MPS, the flattened gradients make a round trip through host memory (Gloo cannot
read MPS tensors). The runtime refines the split on the real job (balance.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import time
from typing import Any

from . import __version__

# Reference workload. A toy model would mislead: at 256-1024-10 the dev Mac's MPS was slower
# than one CPU thread at batch 256 and faster at 1024; at hidden 4096 it was ~4x faster.
REFERENCE_INPUT, REFERENCE_HIDDEN, REFERENCE_OUTPUT = 256, 4096, 10
REFERENCE_BATCH = 512
WARMUP_STEPS = 5
# Median of short rounds: MPS varied 415k-562k samples/s between single 1 s runs on the dev Mac.
MEASURE_ROUNDS = 3
ROUND_SECONDS = 0.5
MAX_ROUND_STEPS = 500


def capability_report(*, benchmark: bool = False, device: str = "auto") -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch": "missing",
        "distributed": False,
        "gloo": False,
        "torchrun": False,
        "safetensors": importlib.util.find_spec("safetensors") is not None,
        "cpu_count": os.cpu_count(),
        "runtime_version": __version__,
    }
    if importlib.util.find_spec("torch") is None:
        return report

    import torch
    import torch.distributed as dist

    from .devices import available_kinds

    report["torch"] = torch.__version__
    report["distributed"] = dist.is_available()
    report["gloo"] = dist.is_available() and dist.is_gloo_available()
    report["torchrun"] = importlib.util.find_spec("torch.distributed.run") is not None
    available = available_kinds()
    if device != "auto" and device not in available:
        raise SystemExit(f"device {device!r} is not available here (available: {available})")
    offered = available if device == "auto" else [device]
    report["available_devices"] = available
    report["offered_devices"] = offered
    report["device"] = offered[0]
    if benchmark:
        report["devices"] = [
            {"kind": kind, "samples_per_second": reference_throughput(kind)} for kind in offered
        ]
        report["device"] = max(report["devices"], key=lambda d: d["samples_per_second"])["kind"]
    return report


def reference_throughput(kind: str) -> float:
    """Training samples per second of the reference model on one device, one process."""
    import torch
    from torch import nn

    from .devices import synchronize

    staged = kind == "mps"  # mirrors DeviceChoice.stages_to_host for a Gloo group

    device = torch.device(kind)
    generator = torch.Generator().manual_seed(0)
    model = nn.Sequential(
        nn.Linear(REFERENCE_INPUT, REFERENCE_HIDDEN),
        nn.ReLU(),
        nn.Linear(REFERENCE_HIDDEN, REFERENCE_OUTPUT),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = torch.randn(REFERENCE_BATCH, REFERENCE_INPUT, generator=generator)
    target = torch.randint(REFERENCE_OUTPUT, (REFERENCE_BATCH,), generator=generator)

    def step() -> None:
        loss = nn.functional.cross_entropy(model(features.to(device)), target.to(device))
        loss.backward()
        flat = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
        if staged:
            flat.copy_(flat.cpu())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(WARMUP_STEPS):
        step()
    synchronize(device)
    rounds = []
    for _ in range(MEASURE_ROUNDS):
        steps, started = 0, time.perf_counter()
        while steps < MAX_ROUND_STEPS and time.perf_counter() - started < ROUND_SECONDS:
            step()
            steps += 1
        synchronize(device)
        rounds.append(steps * REFERENCE_BATCH / (time.perf_counter() - started))
    return statistics.median(rounds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="time the reference model")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    print(json.dumps(capability_report(benchmark=args.benchmark, device=args.device)))


if __name__ == "__main__":
    main()
