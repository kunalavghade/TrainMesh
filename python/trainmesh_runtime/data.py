"""Deterministic in-memory datasets and the loaders that shard them across ranks.

Every rank builds the same tensors from the job seed, so no data moves between machines.
- `synthetic`: random labels, nothing to learn; used for throughput tests.
- `teacher`: labels are the argmax of a fixed random linear "teacher", so a model can learn
  them; validation accuracy then shows whether training worked.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset, TensorDataset

from .config import TrainingSettings
from .distributed import DistContext

TRAIN_SEED_OFFSET = 1
VALIDATION_SEED_OFFSET = 2
TEACHER_SEED_OFFSET = 3
VALIDATION_BATCH_SIZE = 1024


def make_dataset(settings: TrainingSettings, *, samples: int, seed_offset: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(settings.seed + seed_offset)
    features = torch.randn(samples, settings.input_size, generator=generator)
    if settings.dataset_type == "teacher":
        target = (features @ _teacher_weights(settings)).argmax(dim=1)
    else:
        target = torch.randint(settings.output_size, (samples,), generator=generator)
    return TensorDataset(features, target)


def build_train_loader(
    settings: TrainingSettings, ctx: DistContext
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = make_dataset(settings, samples=settings.samples, seed_offset=TRAIN_SEED_OFFSET)
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


def build_validation_loader(settings: TrainingSettings, ctx: DistContext) -> DataLoader | None:
    """Each rank gets every world_size-th sample: disjoint, no padding, exact global counts."""
    if settings.validation_samples <= 0:
        return None
    dataset = make_dataset(
        settings, samples=settings.validation_samples, seed_offset=VALIDATION_SEED_OFFSET
    )
    shard = Subset(dataset, range(ctx.rank, settings.validation_samples, ctx.world_size))
    return DataLoader(shard, batch_size=VALIDATION_BATCH_SIZE, shuffle=False)


def _teacher_weights(settings: TrainingSettings) -> torch.Tensor:
    generator = torch.Generator().manual_seed(settings.seed + TEACHER_SEED_OFFSET)
    return torch.randn(settings.input_size, settings.output_size, generator=generator)
