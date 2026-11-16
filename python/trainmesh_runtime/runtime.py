"""Data-parallel trainer: builds the job from config and runs the epoch loop on CPU, MPS or CUDA."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from . import devices
from .balance import EpochBalance, measure_epoch
from .checkpoint import Progress, load_checkpoint, save_checkpoint
from .config import TrainingSettings, load_config, to_dict, training_settings
from .data import Batches, build_train_loader, build_validation_loader, rank_batch_sizes
from .devices import DeviceChoice
from .distributed import (
    DistContext,
    all_gather_fixed,
    all_reduce_sum_,
    assert_replicas_identical,
    barrier,
    context_from_env,
    destroy_process_group,
    init_process_group,
    max_across_ranks,
)
from .evaluation import ValidationResult, evaluate
from .sync import GradientSync, broadcast_parameters, choose_sync, max_parameter_drift

__all__ = ["MLP", "EpochStats", "StepPlan", "load_config", "train", "train_epoch"]

logger = logging.getLogger(__name__)

OPTIMIZER_FACTORIES: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}
LATEST_CHECKPOINT = "latest"
DEFAULT_THREADS_PER_RANK = 1
BALANCE_UNIFORM = "uniform"
BALANCE_THROUGHPUT = "throughput"
KIND_NAMES = {code: kind for kind, code in devices.DEVICE_KIND_CODES.items()}


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True, slots=True)
class StepPlan:
    """How one rank turns micro-batches into optimizer steps."""

    device: torch.device
    sync: GradientSync
    accumulation_steps: int
    local_batch: int  # this rank's nominal micro-batch size
    global_batch: int  # sum of every rank's nominal micro-batch size
    autocast_dtype: torch.dtype | None = None
    scaler: torch.amp.GradScaler | None = None


@dataclass(frozen=True, slots=True)
class EpochStats:
    loss_sum: torch.Tensor  # float64 on CPU, read once per epoch
    samples: int
    optimizer_steps: int
    compute_seconds: float
    sync_seconds: float
    # Median compute time of one optimizer step: robust to warm-up and stray slow steps.
    median_step_compute_seconds: float


def train(
    config_path: str, output_path: str | None = None, resume_path: str | None = None
) -> dict[str, Any]:
    spec = load_config(config_path)
    settings = training_settings(spec)
    ctx = context_from_env()
    choice = devices.choose(
        settings.device, settings.communication, settings.precision, ctx.local_rank
    )
    if choice.collective_backend == "nccl":
        ctx = replace(ctx, collective_device=choice.device)
    init_process_group(ctx, choice.collective_backend)
    try:
        resume = resume_path or settings.resume
        result = _run_job(to_dict(spec), settings, ctx, choice, resume_path=resume)
        _write_results(result, output_path or settings.results_path, ctx)
        barrier(ctx)
        return result
    finally:
        destroy_process_group()


def train_epoch(
    model: nn.Module,
    loader: Batches,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    plan: StepPlan,
) -> EpochStats:
    """Run one epoch; gradients are synchronized once per optimizer step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # MPS has no float64; the per-epoch sum fits float32 comfortably.
    accumulator_dtype = torch.float32 if plan.device.type == "mps" else torch.float64
    loss_sum = torch.zeros((), dtype=accumulator_dtype, device=plan.device)
    samples = pending = optimizer_steps = 0
    sync_seconds = 0.0
    step_compute: list[float] = []
    last_index = len(loader) - 1
    started = step_started = time.perf_counter()

    for batch_index, (features, target) in enumerate(loader):
        features, target = features.to(plan.device), target.to(plan.device)
        pending += 1
        closes_window = pending == plan.accumulation_steps or batch_index == last_index
        with plan.sync.micro_batch(model, closes_window):
            with torch.autocast(
                plan.device.type, dtype=plan.autocast_dtype, enabled=plan.autocast_dtype is not None
            ):
                loss = criterion(model(features), target)
            scaled = loss / _loss_divisor(plan, len(target))
            (plan.scaler.scale(scaled) if plan.scaler else scaled).backward()
        if closes_window:
            step_sync = _timed_sync(model, plan)
            sync_seconds += step_sync
            _rescale_partial_window(model, pending, plan.accumulation_steps)
            _optimizer_step(optimizer, plan)
            optimizer_steps += 1
            pending = 0
            now = time.perf_counter()
            step_compute.append(now - step_started - step_sync)
            step_started = now
        loss_sum += loss.detach().to(accumulator_dtype)
        samples += len(target)

    devices.synchronize(plan.device)
    elapsed = time.perf_counter() - started
    return EpochStats(
        loss_sum=loss_sum.cpu().double(),
        samples=samples,
        optimizer_steps=optimizer_steps,
        compute_seconds=elapsed - sync_seconds,
        sync_seconds=sync_seconds,
        median_step_compute_seconds=statistics.median(step_compute) if step_compute else 0.0,
    )


def configure_threads() -> int:
    # The Go launcher always sets OMP_NUM_THREADS; 1 matches torchrun's own default.
    configured = os.environ.get("OMP_NUM_THREADS", "")
    torch.set_num_threads(max(1, int(configured) if configured else DEFAULT_THREADS_PER_RANK))
    return torch.get_num_threads()


def build_optimizer(model: nn.Module, settings: TrainingSettings) -> torch.optim.Optimizer:
    return OPTIMIZER_FACTORIES[settings.optimizer_name](
        model.parameters(), lr=settings.learning_rate
    )


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(unwrap(model).state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _run_job(
    config: dict[str, Any],
    settings: TrainingSettings,
    ctx: DistContext,
    choice: DeviceChoice,
    *,
    resume_path: str | None,
) -> dict[str, Any]:
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    threads = configure_threads()
    sizes = rank_batch_sizes(settings, ctx)
    rebalancing = settings.balance == BALANCE_THROUGHPUT and ctx.is_distributed
    device_kinds = _gather_device_kinds(choice, ctx)
    mixed_devices = len(set(device_kinds)) > 1
    sync = choose_sync(
        settings.gradient_sync,
        ctx,
        choice,
        # A throughput-balanced split can change every epoch, so it always needs all-reduce.
        uneven_batches=len(set(sizes)) > 1 or rebalancing,
        group_device_kinds=device_kinds,
    )
    loader = build_train_loader(settings, ctx)
    validation_loader = build_validation_loader(settings, ctx)
    model = sync.wrap(
        MLP(settings.input_size, settings.hidden_size, settings.output_size).to(choice.device)
    )
    optimizer = build_optimizer(model, settings)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    plan = StepPlan(
        device=choice.device,
        sync=sync,
        accumulation_steps=settings.accumulation_steps,
        local_batch=sizes[ctx.rank],
        global_batch=sum(sizes),
        autocast_dtype=choice.autocast_dtype,
        scaler=torch.amp.GradScaler(choice.kind) if choice.precision == "fp16" else None,
    )

    progress = Progress(epoch=0, global_step=0)
    if resume_path:
        progress = load_checkpoint(
            Path(resume_path),
            model=unwrap(model),
            optimizer=optimizer,
            training_config=config["training"],
            ctx=ctx,
        )
    start_epoch = progress.epoch

    started = time.perf_counter()
    epoch_losses: list[float] = []
    validation_history: list[ValidationResult] = []
    processed_samples = 0
    compute_seconds = sync_seconds = max_drift = 0.0
    balance_history: list[EpochBalance] = []
    for epoch in range(start_epoch, settings.epochs):
        loader.sampler.set_epoch(epoch)
        stats = train_epoch(model, loader, optimizer, criterion, plan)
        compute_seconds += stats.compute_seconds
        sync_seconds += stats.sync_seconds
        if rebalancing:
            balance = measure_epoch(epoch + 1, sizes, stats.median_step_compute_seconds, ctx)
            balance_history.append(balance)
            sizes = balance.next_split()
            loader.sampler.resize(sizes)
            plan = replace(plan, local_batch=sizes[ctx.rank])
        totals = torch.stack([stats.loss_sum, torch.tensor(stats.samples, dtype=torch.float64)])
        all_reduce_sum_(totals, ctx)
        loss_total, sample_total = totals.tolist()
        epoch_loss = loss_total / max(1.0, sample_total)
        epoch_losses.append(epoch_loss)
        processed_samples += int(sample_total)
        if mixed_devices:
            # Different device kinds round Adam's update differently; measure, then realign.
            max_drift = max(max_drift, max_parameter_drift(unwrap(model), ctx, choice))
            broadcast_parameters(unwrap(model), ctx, choice)
        global_step = progress.global_step + stats.optimizer_steps
        progress = Progress(epoch=epoch + 1, global_step=global_step)
        event: dict[str, Any] = {
            "event": "epoch",
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "samples": int(sample_total),
            "world_size": ctx.world_size,
        }
        if validation_loader is not None:
            validation = evaluate(unwrap(model), validation_loader, ctx, choice.device)
            validation_history.append(validation)
            event.update(val_loss=validation.loss, val_accuracy=validation.accuracy)
        if ctx.is_primary:
            _emit_event(event)
        if settings.checkpoint_enabled:
            save_checkpoint(
                settings.checkpoint_directory / LATEST_CHECKPOINT,
                model=unwrap(model),
                optimizer=optimizer,
                progress=progress,
                config=config,
                seed=settings.seed,
                ctx=ctx,
            )

    elapsed = max_across_ranks(time.perf_counter() - started, ctx)
    digest = model_digest(model)
    assert_replicas_identical(digest, ctx)
    # The split the last epoch trained with (the loop already computed the next one).
    final_sizes = balance_history[-1].rank_batch_sizes if balance_history else sizes
    rank_stats = _gather_rank_stats(
        final_sizes[ctx.rank], compute_seconds, sync_seconds, choice, ctx
    )
    return {
        "world_size": ctx.world_size,
        "rank": ctx.rank,
        "local_rank": ctx.local_rank,
        "threads_per_rank": threads,
        "device": choice.kind,
        "precision": choice.precision,
        "collective_backend": choice.collective_backend,
        "gradient_sync": sync.name,
        "balance": settings.balance or BALANCE_UNIFORM,
        "rank_batch_sizes": final_sizes,
        "balance_history": [b.as_dict() for b in balance_history],
        "epochs": settings.epochs - start_epoch,
        "global_step": progress.global_step,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "epoch_losses": epoch_losses,
        "samples_processed": processed_samples,
        "duration_seconds": elapsed,
        "samples_per_second": processed_samples / max(elapsed, 1e-9),
        "model_sha256": digest,
        "parameters_consistent": True,
        "mixed_devices": mixed_devices,
        "max_parameter_drift": max_drift,
        "ranks": rank_stats,
        "compute_imbalance": _imbalance(rank_stats),
        "dataset_type": settings.dataset_type,
        "validation": _validation_summary(validation_history),
    }


def _loss_divisor(plan: StepPlan, batch_length: int) -> float:
    """Turn a summed loss into this rank's share of the global-mean loss.

    DDP averages gradients over ranks, so each rank divides by its own samples. The all-reduce
    strategy sums gradients, so each rank divides by the global batch it is part of (scaled for a
    short final batch the same way on every rank).
    """
    if plan.sync.name == "ddp":
        return batch_length * plan.accumulation_steps
    return batch_length * plan.accumulation_steps * plan.global_batch / plan.local_batch


def _timed_sync(model: nn.Module, plan: StepPlan) -> float:
    devices.synchronize(plan.device)  # finish queued compute so it is not counted as sync time
    start = time.perf_counter()
    plan.sync.synchronize(model)
    devices.synchronize(plan.device)
    return time.perf_counter() - start


def _optimizer_step(optimizer: torch.optim.Optimizer, plan: StepPlan) -> None:
    if plan.scaler is not None:
        plan.scaler.step(optimizer)
        plan.scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _gather_device_kinds(choice: DeviceChoice, ctx: DistContext) -> list[str]:
    code = torch.tensor([devices.DEVICE_KIND_CODES[choice.kind]], dtype=torch.int64)
    return [KIND_NAMES[int(t.item())] for t in all_gather_fixed(code, ctx)]


def _gather_rank_stats(
    batch: int, compute: float, sync: float, choice: DeviceChoice, ctx: DistContext
) -> list[dict[str, Any]]:
    local = torch.tensor(
        [batch, compute, sync, devices.DEVICE_KIND_CODES[choice.kind]], dtype=torch.float64
    )
    return [
        {
            "rank": rank,
            "device": KIND_NAMES[int(row[3].item())],
            "batch_size": int(row[0].item()),
            "compute_seconds": float(row[1].item()),
            "sync_seconds": float(row[2].item()),
        }
        for rank, row in enumerate(all_gather_fixed(local, ctx))
    ]


def _imbalance(rank_stats: list[dict[str, Any]]) -> float:
    """Slowest rank's compute time over the fastest's; 1.0 means perfectly balanced."""
    compute = [r["compute_seconds"] for r in rank_stats]
    return max(compute) / max(min(compute), 1e-9)


def _validation_summary(history: list[ValidationResult]) -> dict[str, Any] | None:
    if not history:
        return None
    final = history[-1]
    return {
        "samples": final.samples,
        "final_loss": final.loss,
        "final_accuracy": final.accuracy,
        "accuracy_history": [entry.accuracy for entry in history],
        "loss_history": [entry.loss for entry in history],
    }


def _rescale_partial_window(model: nn.Module, pending: int, accumulation_steps: int) -> None:
    # Losses were divided by the full window; a short final window needs scaling back up.
    if pending == accumulation_steps:
        return
    correction = accumulation_steps / pending
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(correction)


def _write_results(result: dict[str, Any], destination: str, ctx: DistContext) -> None:
    if not ctx.is_primary or not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _emit_event(payload: dict[str, Any]) -> None:
    # stdout JSON lines are the runtime's event protocol for the Go launcher.
    print(json.dumps(payload), flush=True)
