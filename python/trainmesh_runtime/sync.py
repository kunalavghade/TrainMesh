"""Gradient synchronization strategies.

- DDPSync: PyTorch DDP; overlaps the all-reduce with backward. Needs equal per-rank batches and a
  device the collective backend can read (CPU or CUDA).
- AllReduceSync: one flattened all-reduce (SUM) per optimizer step, copied through CPU memory
  when the device cannot be read by the backend (MPS). Losses are divided by the *global* batch,
  so unequal per-rank batches still produce the exact global-mean gradient.
"""

from __future__ import annotations

import contextlib
from typing import Protocol

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .devices import DeviceChoice
from .distributed import DistContext
from .errors import ConfigError


class GradientSync(Protocol):
    name: str

    def wrap(self, model: nn.Module) -> nn.Module: ...

    def micro_batch(
        self, model: nn.Module, closes_window: bool
    ) -> contextlib.AbstractContextManager: ...

    def synchronize(self, model: nn.Module) -> None: ...

    def loss_divisor(self, local_batch: int, global_batch: int, accumulation: int) -> int: ...


class DDPSync:
    name = "ddp"

    def __init__(self, ctx: DistContext, choice: DeviceChoice) -> None:
        self._ctx = ctx
        self._choice = choice

    def wrap(self, model: nn.Module) -> nn.Module:
        if not self._ctx.is_distributed:
            return model
        device_ids = [self._choice.device.index] if self._choice.kind == "cuda" else None
        return DistributedDataParallel(model, device_ids=device_ids)

    def micro_batch(
        self, model: nn.Module, closes_window: bool
    ) -> contextlib.AbstractContextManager:
        # Accumulate locally; DDP all-reduces only on the micro-batch that closes the window.
        if isinstance(model, DistributedDataParallel) and not closes_window:
            return model.no_sync()
        return contextlib.nullcontext()

    def synchronize(self, model: nn.Module) -> None:
        """DDP already reduced during backward."""

    def loss_divisor(self, local_batch: int, global_batch: int, accumulation: int) -> int:
        return local_batch * accumulation  # DDP averages over ranks


class AllReduceSync:
    name = "allreduce"

    def __init__(self, ctx: DistContext, choice: DeviceChoice) -> None:
        self._ctx = ctx
        self._choice = choice

    def wrap(self, model: nn.Module) -> nn.Module:
        broadcast_parameters(model, self._ctx, self._choice)
        return model

    def micro_batch(
        self, model: nn.Module, closes_window: bool
    ) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()

    def synchronize(self, model: nn.Module) -> None:
        if not self._ctx.is_distributed:
            return
        parameters = [p for p in model.parameters() if p.requires_grad]
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in parameters]
        flat = torch.cat([g.reshape(-1) for g in grads])
        buffer = flat.cpu() if self._choice.stages_to_host else flat
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        if buffer is not flat:
            flat.copy_(buffer)
        offset = 0
        for parameter, grad in zip(parameters, grads, strict=True):
            count = grad.numel()
            reduced = flat[offset : offset + count].view_as(grad)
            if parameter.grad is None:
                parameter.grad = reduced.clone()
            else:
                parameter.grad.copy_(reduced)
            offset += count

    def loss_divisor(self, local_batch: int, global_batch: int, accumulation: int) -> int:
        return global_batch * accumulation  # SUM across ranks then gives the global mean


def broadcast_parameters(model: nn.Module, ctx: DistContext, choice: DeviceChoice) -> None:
    """Copy rank 0's parameters to every rank (through CPU memory when needed)."""
    if not ctx.is_distributed:
        return
    with torch.no_grad():
        for tensor in model.state_dict().values():
            buffer = tensor.cpu() if choice.stages_to_host else tensor
            dist.broadcast(buffer, src=0)
            if buffer is not tensor:
                tensor.copy_(buffer)


def max_parameter_drift(model: nn.Module, ctx: DistContext, choice: DeviceChoice) -> float:
    """Largest absolute difference between this rank's parameters and rank 0's, over all ranks."""
    if not ctx.is_distributed:
        return 0.0
    drift = torch.zeros(1, dtype=torch.float64)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            local = tensor.detach().cpu().double()
            reference = local.clone() if choice.stages_to_host else local.to(tensor.device)
            dist.broadcast(reference, src=0)
            drift = torch.maximum(drift, (local - reference.cpu()).abs().max().reshape(1))
    dist.all_reduce(drift, op=dist.ReduceOp.MAX)
    return float(drift.item())


def choose_sync(
    requested: str,
    ctx: DistContext,
    choice: DeviceChoice,
    *,
    uneven_batches: bool,
    group_device_kinds: list[str],
) -> GradientSync:
    """Pick one strategy for the whole group: every rank must run the same collectives.

    The all-reduce strategy is required when per-rank batches differ, when any rank is on MPS
    (its gradients go through host memory) or when ranks use different device kinds.
    """
    mixed_kinds = len(set(group_device_kinds)) > 1
    needs_allreduce = uneven_batches or "mps" in group_device_kinds or mixed_kinds
    if requested == "ddp" and needs_allreduce:
        reason = (
            "per-rank batches are unequal" if uneven_batches else "ranks use mps or mixed devices"
        )
        raise ConfigError(
            f"execution.gradient_sync=ddp is not possible because {reason}; use allreduce"
        )
    if requested == "allreduce" or needs_allreduce:
        return AllReduceSync(ctx, choice)
    return DDPSync(ctx, choice)
