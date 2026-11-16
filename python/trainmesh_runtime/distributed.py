"""Process-group helpers that keep collectives tensor-based (no pickle on the wire)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

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
    # Where tensors must live for this process group's collectives (CUDA for NCCL).
    collective_device: torch.device = field(default_factory=lambda: torch.device("cpu"))

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


def init_process_group(ctx: DistContext, backend: str = "gloo") -> None:
    if not ctx.is_distributed:
        return
    available = {
        "gloo": dist.is_available() and dist.is_gloo_available(),
        "nccl": dist.is_available() and dist.is_nccl_available(),
    }
    if not available.get(backend, False):
        raise ConfigError(f"installed PyTorch does not provide the {backend} process-group backend")
    dist.init_process_group(backend=backend, init_method="env://")


def destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier(ctx: DistContext) -> None:
    if ctx.is_distributed:
        dist.barrier()


def all_reduce_sum_(tensor: torch.Tensor, ctx: DistContext) -> torch.Tensor:
    if ctx.is_distributed:
        buffer = tensor.to(ctx.collective_device)
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        tensor.copy_(buffer)
    return tensor


def max_across_ranks(value: float, ctx: DistContext) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=ctx.collective_device)
    if ctx.is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def all_gather_fixed(tensor: torch.Tensor, ctx: DistContext) -> list[torch.Tensor]:
    """Gather an equally shaped tensor from every rank, in rank order (results on CPU)."""
    if not ctx.is_distributed:
        return [tensor]
    local = tensor.contiguous().to(ctx.collective_device)
    gathered = [torch.empty_like(local) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, local)
    return [t.cpu() for t in gathered]


def assert_replicas_identical(digest_hex: str, ctx: DistContext) -> None:
    digest = torch.frombuffer(bytearray(bytes.fromhex(digest_hex)), dtype=torch.uint8)
    if digest.numel() != SHA256_DIGEST_BYTES:
        raise ValueError(f"expected a {SHA256_DIGEST_BYTES}-byte digest, got {digest.numel()}")
    digests = [bytes(t.tolist()).hex() for t in all_gather_fixed(digest, ctx)]
    if len(set(digests)) != 1:
        raise DivergenceError(f"DDP model parameters diverged across ranks: {digests}")
