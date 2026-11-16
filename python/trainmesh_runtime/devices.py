"""Device, precision and collective-backend selection for one rank.

Measured on the development Mac (torch 2.15, MLP 256-1024-10): MPS was 1.5-4.3x faster than one CPU
thread, while bf16 autocast on that CPU was 16-20x slower than fp32 and fp16 on MPS was slower
than fp32. Defaults therefore stay fp32 and "auto" precision only picks bf16 on CUDA.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .errors import ConfigError

# Set by tm-agent so `device: auto` uses the device this machine offers to the mesh.
DEVICE_OVERRIDE_ENV = "TRAINMESH_DEVICE"
DEVICE_PRIORITY = ("cuda", "mps", "cpu")
DEVICE_KIND_CODES = {"cpu": 0, "mps": 1, "cuda": 2}


@dataclass(frozen=True, slots=True)
class DeviceChoice:
    device: torch.device
    collective_backend: str  # "gloo" or "nccl"
    autocast_dtype: torch.dtype | None
    precision: str  # resolved name: fp32, bf16 or fp16

    @property
    def kind(self) -> str:
        return self.device.type

    @property
    def stages_to_host(self) -> bool:
        """Whether gradients must be copied to CPU memory before a collective."""
        return self.collective_backend == "gloo" and self.kind == "mps"


def available_kinds() -> list[str]:
    kinds = []
    if torch.cuda.is_available():
        kinds.append("cuda")
    if torch.backends.mps.is_available():
        kinds.append("mps")
    kinds.append("cpu")
    return kinds


def resolve_device(requested: str, local_rank: int) -> torch.device:
    kind = requested
    if requested == "auto":
        kind = os.environ.get(DEVICE_OVERRIDE_ENV) or available_kinds()[0]
    if kind not in available_kinds():
        raise ConfigError(f"device {kind!r} was requested but is not available on this machine")
    if kind == "cuda":
        return torch.device("cuda", local_rank % torch.cuda.device_count())
    return torch.device(kind)


def resolve_backend(requested: str, device: torch.device) -> str:
    if requested == "nccl" or (requested == "auto" and device.type == "cuda"):
        if device.type != "cuda" or not (dist.is_available() and dist.is_nccl_available()):
            if requested == "nccl":
                raise ConfigError(
                    "communication=nccl needs a CUDA device and a PyTorch build with NCCL"
                )
            return "gloo"
        return "nccl"
    return "gloo"


def resolve_precision(requested: str, device: torch.device) -> tuple[str, torch.dtype | None]:
    precision = requested or "fp32"
    if precision == "auto":
        cuda_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        precision = "bf16" if cuda_bf16 else "fp32"
    if precision == "fp32":
        return precision, None
    if precision == "fp16" and device.type == "cpu":
        raise ConfigError(
            "training.precision=fp16 needs a cuda or mps device; use bf16 or fp32 on cpu"
        )
    return precision, torch.bfloat16 if precision == "bf16" else torch.float16


def choose(device: str, communication: str, precision: str, local_rank: int) -> DeviceChoice:
    resolved = resolve_device(device, local_rank)
    name, dtype = resolve_precision(precision, resolved)
    return DeviceChoice(
        device=resolved,
        collective_backend=resolve_backend(communication, resolved),
        autocast_dtype=dtype,
        precision=name,
    )


def synchronize(device: torch.device) -> None:
    """Wait for queued device work, so wall-clock timings mean something."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
