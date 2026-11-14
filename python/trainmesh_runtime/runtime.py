"""CPU/Gloo data-parallel trainer: builds the job from config and runs the epoch loop."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from .checkpoint import Progress, load_checkpoint, save_checkpoint
from .config import TrainingSettings, load_config, training_settings
from .distributed import (
    DistContext,
    all_reduce_sum_,
    assert_replicas_identical,
    barrier,
    context_from_env,
    destroy_process_group,
    init_process_group,
    max_across_ranks,
)

__all__ = ["MLP", "EpochStats", "load_config", "train", "train_epoch"]

logger = logging.getLogger(__name__)

OPTIMIZER_FACTORIES: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}
LATEST_CHECKPOINT = "latest"
DEFAULT_THREADS_PER_RANK = 1


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True, slots=True)
class EpochStats:
    loss_sum: torch.Tensor  # stays a tensor so the loop never forces a host sync
    samples: int
    optimizer_steps: int


def train(
    config_path: str, output_path: str | None = None, resume_path: str | None = None
) -> dict[str, Any]:
    config = load_config(config_path)
    settings = training_settings(config)
    ctx = context_from_env()
    init_process_group(ctx)
    try:
        resume = resume_path or config.get("run", {}).get("resume")
        result = _run_job(config, settings, ctx, resume_path=resume)
        _write_results(result, output_path or settings.results_path, ctx)
        barrier(ctx)
        return result
    finally:
        destroy_process_group()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    accumulation_steps: int,
) -> EpochStats:
    """Run one epoch; DDP all-reduces only on the micro-batch that closes a window."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), dtype=torch.float64)
    samples = pending = optimizer_steps = 0
    last_index = len(loader) - 1

    for batch_index, (features, target) in enumerate(loader):
        pending += 1
        closes_window = pending == accumulation_steps or batch_index == last_index
        defers_sync = isinstance(model, DistributedDataParallel) and not closes_window
        with model.no_sync() if defers_sync else contextlib.nullcontext():
            loss = criterion(model(features), target)
            (loss / (len(target) * accumulation_steps)).backward()
        if closes_window:
            _rescale_partial_window(model, pending, accumulation_steps)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            pending = 0
        loss_sum += loss.detach()
        samples += len(target)

    return EpochStats(loss_sum=loss_sum, samples=samples, optimizer_steps=optimizer_steps)


def configure_threads() -> int:
    # The Go launcher always sets OMP_NUM_THREADS; 1 matches torchrun's own default.
    configured = os.environ.get("OMP_NUM_THREADS", "")
    torch.set_num_threads(max(1, int(configured) if configured else DEFAULT_THREADS_PER_RANK))
    return torch.get_num_threads()


def build_loader(
    settings: TrainingSettings, ctx: DistContext
) -> tuple[DataLoader, DistributedSampler | None]:
    # Identical deterministic dataset on every rank; DistributedSampler assigns disjoint indices.
    generator = torch.Generator().manual_seed(settings.seed + 1)
    features = torch.randn(settings.samples, settings.input_size, generator=generator)
    target = torch.randint(settings.output_size, (settings.samples,), generator=generator)
    dataset = TensorDataset(features, target)
    sampler = (
        DistributedSampler(
            dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True, seed=settings.seed
        )
        if ctx.is_distributed
        else None
    )
    loader = DataLoader(
        dataset, batch_size=settings.per_rank_batch, sampler=sampler, shuffle=sampler is None
    )
    return loader, sampler


def build_model(settings: TrainingSettings, ctx: DistContext) -> nn.Module:
    model = MLP(settings.input_size, settings.hidden_size, settings.output_size)
    return DistributedDataParallel(model) if ctx.is_distributed else model


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
    *,
    resume_path: str | None,
) -> dict[str, Any]:
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    threads = configure_threads()
    loader, sampler = build_loader(settings, ctx)
    model = build_model(settings, ctx)
    optimizer = build_optimizer(model, settings)
    criterion = nn.CrossEntropyLoss(reduction="sum")

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
    processed_samples = 0
    for epoch in range(start_epoch, settings.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        stats = train_epoch(
            model, loader, optimizer, criterion, accumulation_steps=settings.accumulation_steps
        )
        totals = torch.stack([stats.loss_sum, torch.tensor(stats.samples, dtype=torch.float64)])
        all_reduce_sum_(totals, ctx)
        loss_total, sample_total = totals.tolist()
        epoch_loss = loss_total / max(1.0, sample_total)
        epoch_losses.append(epoch_loss)
        processed_samples += int(sample_total)
        global_step = progress.global_step + stats.optimizer_steps
        progress = Progress(epoch=epoch + 1, global_step=global_step)
        if ctx.is_primary:
            _emit_event(
                {
                    "event": "epoch",
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "samples": int(sample_total),
                    "world_size": ctx.world_size,
                }
            )
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
    return {
        "world_size": ctx.world_size,
        "rank": ctx.rank,
        "local_rank": ctx.local_rank,
        "threads_per_rank": threads,
        "epochs": settings.epochs - start_epoch,
        "global_step": progress.global_step,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "epoch_losses": epoch_losses,
        "samples_processed": processed_samples,
        "duration_seconds": elapsed,
        "samples_per_second": processed_samples / max(elapsed, 1e-9),
        "model_sha256": digest,
        "parameters_consistent": True,
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
