"""Throughput balancing: per-rank micro-batches proportional to each rank's measured speed.

The coordinator makes the first split from each node's reference benchmark. That benchmark uses
a fixed model, and on the development Mac the MPS/CPU speed ratio changed with model and batch
size (MLP 256-1024-10 training step: MPS 0.6x one CPU thread at batch 256, 1.1x at 1024), and
the probe's 1.7x at hidden 4096 became about 1x in a real two-agent job on that Mac. So with
`balance: throughput` every rank re-measures its own speed on the real job each epoch and the
group re-splits the same global batch for the next epoch. The global batches stay identical, so
the trained model does not depend on the split.

Speed is the median per-step compute time (per-epoch totals swung by 40% between epochs on the
shared dev Mac, and the first epoch carries device warm-up), and a split is only changed when
the measured imbalance exceeds REBALANCE_THRESHOLD, so noise does not make it oscillate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .distributed import DistContext, all_gather_fixed

MIN_RANK_BATCH = 1
# Re-split only when the slowest rank's step takes this much longer than the fastest's.
REBALANCE_THRESHOLD = 1.10


def apportion(total: int, weights: Sequence[float]) -> list[int]:
    """Split total into parts proportional to weights (largest-remainder method).

    Parts always sum to total; ties go to the lower rank; a part rounded below MIN_RANK_BATCH is
    raised to it, taken from the largest part. Same algorithm as the coordinator's planner.
    """
    weight_sum = sum(weights)
    shares = [total * w / weight_sum for w in weights]
    sizes = [math.floor(share) for share in shares]
    by_remainder = sorted(range(len(weights)), key=lambda r: (-(shares[r] - sizes[r]), r))
    for rank in by_remainder[: total - sum(sizes)]:
        sizes[rank] += 1
    for rank in range(len(sizes)):
        while sizes[rank] < MIN_RANK_BATCH:
            sizes[sizes.index(max(sizes))] -= 1
            sizes[rank] += 1
    return sizes


@dataclass(frozen=True, slots=True)
class EpochBalance:
    epoch: int
    rank_batch_sizes: list[int]
    step_compute_seconds: list[float]  # median per-step compute time of each rank

    @property
    def compute_imbalance(self) -> float:
        """Slowest rank's step compute over the fastest's for this epoch; 1.0 is perfect."""
        return max(self.step_compute_seconds) / max(min(self.step_compute_seconds), 1e-9)

    def next_split(self) -> list[int]:
        """The split for the next epoch: unchanged unless this one was clearly imbalanced."""
        if self.compute_imbalance <= REBALANCE_THRESHOLD:
            return list(self.rank_batch_sizes)
        rates = [
            size / max(seconds, 1e-9)
            for size, seconds in zip(self.rank_batch_sizes, self.step_compute_seconds, strict=True)
        ]
        return apportion(sum(self.rank_batch_sizes), rates)

    def as_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "rank_batch_sizes": self.rank_batch_sizes,
            "step_compute_ms": [round(seconds * 1e3, 3) for seconds in self.step_compute_seconds],
            "compute_imbalance": self.compute_imbalance,
        }


def measure_epoch(
    epoch: int, sizes: Sequence[int], step_compute_seconds: float, ctx: DistContext
) -> EpochBalance:
    """Collective call: every rank receives every rank's median step compute time."""
    gathered = all_gather_fixed(torch.tensor([step_compute_seconds], dtype=torch.float64), ctx)
    return EpochBalance(
        epoch=epoch,
        rank_batch_sizes=list(sizes),
        step_compute_seconds=[float(t.item()) for t in gathered],
    )
