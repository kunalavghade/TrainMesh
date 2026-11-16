"""Validation loss and accuracy over a dataset sharded across ranks, reduced to global numbers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from .distributed import DistContext, all_reduce_sum_


@dataclass(frozen=True, slots=True)
class ValidationResult:
    loss: float
    accuracy: float
    samples: int


def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, ...]],
    ctx: DistContext,
    device: torch.device | None = None,
) -> ValidationResult:
    """Collective call: every rank evaluates its shard, then sums are all-reduced."""
    was_training = model.training
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64)  # loss sum, correct, samples
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for features, target in loader:
            if device is not None:
                features, target = features.to(device), target.to(device)
            logits = model(features)
            # Accumulate on CPU in float64: MPS has no float64.
            totals[0] += criterion(logits.float(), target).detach().cpu().double()
            totals[1] += (logits.argmax(dim=1) == target).sum().cpu()
            totals[2] += len(target)
    model.train(was_training)

    loss_sum, correct, samples = all_reduce_sum_(totals, ctx).tolist()
    count = max(1.0, samples)
    return ValidationResult(loss=loss_sum / count, accuracy=correct / count, samples=int(samples))
