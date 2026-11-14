"""Process-group helpers that keep collectives tensor-based (no pickle on the wire)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .errors import ConfigError, DivergenceError

SHA256_DIGEST_BYTES = 32


@dataclass(frozen=True, slots=True)
class DistContext:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def context_from_env() -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return DistContext(
        rank=int(os.environ.get("RANK", "0")),
        world_size=world_size,
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size))),
    )


def init_process_group(ctx: DistContext) -> None:
    if not ctx.is_distributed:
        return
    if not dist.is_available() or not dist.is_gloo_available():
        raise ConfigError("installed PyTorch does not provide the Gloo process-group backend")
    dist.init_process_group(backend="gloo", init_method="env://")


def destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier(ctx: DistContext) -> None:
    if ctx.is_distributed:
        dist.barrier()


def all_reduce_sum_(tensor: torch.Tensor, ctx: DistContext) -> torch.Tensor:
    if ctx.is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def max_across_ranks(value: float, ctx: DistContext) -> float:
    tensor = torch.tensor([value], dtype=torch.float64)
    if ctx.is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def all_gather_fixed(tensor: torch.Tensor, ctx: DistContext) -> list[torch.Tensor]:
    """Gather an equally shaped tensor from every rank, in rank order."""
    if not ctx.is_distributed:
        return [tensor]
    gathered = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return gathered


def assert_replicas_identical(digest_hex: str, ctx: DistContext) -> None:
    digest = torch.frombuffer(bytearray(bytes.fromhex(digest_hex)), dtype=torch.uint8)
    if digest.numel() != SHA256_DIGEST_BYTES:
        raise ValueError(f"expected a {SHA256_DIGEST_BYTES}-byte digest, got {digest.numel()}")
    digests = [bytes(t.tolist()).hex() for t in all_gather_fixed(digest, ctx)]
    if len(set(digests)) != 1:
        raise DivergenceError(f"DDP model parameters diverged across ranks: {digests}")
