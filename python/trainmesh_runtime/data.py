"""Deterministic in-memory datasets and the loaders that shard them across ranks.

Every rank builds the same tensors from the job seed, so no data moves between machines.
- `synthetic`: random labels, nothing to learn; used for throughput tests.
- `teacher`: labels are the argmax of a fixed random linear "teacher", so a model can learn
  them; validation accuracy then shows whether training worked.

Every global step covers the same block of the same shuffled order whatever the world size or
the per-rank split (`training.batch.rank_batch_sizes`, set for throughput balancing), so every
topology and every split trains on identical global batches.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

import torch
from torch.utils.data import TensorDataset

from .config import TrainingSettings
from .distributed import DistContext
from .errors import ConfigError

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


class Batches(Protocol):
    """What the trainer iterates: a sized sequence of (features, target) batches."""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]: ...


class GlobalBatchSampler:
    """Yields this rank's slice (an index tensor) of each global batch.

    The shuffled order is DistributedSampler's (randperm seeded with seed + epoch) and step k
    uses positions [k*G, (k+1)*G), G = sum(rank_batch_sizes); rank r takes its contiguous
    slice of that block. The union over ranks is therefore the same global batch for any split
    and any world size. An incomplete final global batch is dropped.
    """

    def __init__(self, dataset_size: int, rank_batch_sizes: Sequence[int], rank: int, seed: int):
        self._size = dataset_size
        self._rank = rank
        self._seed = seed
        self._epoch = 0
        self.resize(rank_batch_sizes)

    def resize(self, rank_batch_sizes: Sequence[int]) -> None:
        """Change the split; the global batch (their sum) must stay the same."""
        sizes = list(rank_batch_sizes)
        if hasattr(self, "_global") and sum(sizes) != self._global:
            raise ValueError(f"resize must keep the global batch {self._global}, got {sizes}")
        self._global = sum(sizes)
        self._offset = sum(sizes[: self._rank])
        self._count = sizes[self._rank]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self._size // self._global

    def __iter__(self) -> Iterator[torch.Tensor]:
        generator = torch.Generator().manual_seed(self._seed + self._epoch)
        order = torch.randperm(self._size, generator=generator)
        for start in range(self._offset, len(self) * self._global, self._global):
            yield order[start : start + self._count]


class BatchLoader:
    """Indexes in-memory tensors with a whole batch of indices at once.

    Fetching samples one by one through a DataLoader and collating them cost ~3-4 ms per
    1024-sample batch on the development Mac: more than the model step itself, which hid the
    difference between devices.
    """

    def __init__(self, dataset: TensorDataset, sampler: GlobalBatchSampler):
        self._tensors = dataset.tensors
        self.sampler = sampler

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        for indices in self.sampler:
            yield tuple(tensor[indices] for tensor in self._tensors)


class TensorBatches:
    """Fixed, unshuffled batches over in-memory tensors (validation)."""

    def __init__(self, tensors: Sequence[torch.Tensor], batch_size: int):
        self._tensors = tensors
        self._batch_size = batch_size

    def __len__(self) -> int:
        return -(-len(self._tensors[0]) // self._batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        for start in range(0, len(self._tensors[0]), self._batch_size):
            yield tuple(tensor[start : start + self._batch_size] for tensor in self._tensors)


def rank_batch_sizes(settings: TrainingSettings, ctx: DistContext) -> list[int]:
    """Per-rank micro-batch sizes; validated against the world size."""
    if not settings.rank_batch_sizes:
        return [settings.per_rank_batch] * ctx.world_size
    sizes = list(settings.rank_batch_sizes)
    expected_total = settings.per_rank_batch * ctx.world_size
    if len(sizes) != ctx.world_size or sum(sizes) != expected_total:
        raise ConfigError(
            f"training.batch.rank_batch_sizes {sizes} must have {ctx.world_size} entries "
            f"summing to per_rank x world_size = {expected_total}"
        )
    return sizes


def build_train_loader(settings: TrainingSettings, ctx: DistContext) -> BatchLoader:
    dataset = make_dataset(settings, samples=settings.samples, seed_offset=TRAIN_SEED_OFFSET)
    sampler = GlobalBatchSampler(
        len(dataset), rank_batch_sizes(settings, ctx), ctx.rank, settings.seed
    )
    return BatchLoader(dataset, sampler)


def build_validation_loader(settings: TrainingSettings, ctx: DistContext) -> TensorBatches | None:
    """Each rank gets every world_size-th sample: disjoint, no padding, exact global counts."""
    if settings.validation_samples <= 0:
        return None
    dataset = make_dataset(
        settings, samples=settings.validation_samples, seed_offset=VALIDATION_SEED_OFFSET
    )
    shard = [tensor[ctx.rank :: ctx.world_size] for tensor in dataset.tensors]
    return TensorBatches(shard, VALIDATION_BATCH_SIZE)


def _teacher_weights(settings: TrainingSettings) -> torch.Tensor:
    generator = torch.Generator().manual_seed(settings.seed + TEACHER_SEED_OFFSET)
    return torch.randn(settings.input_size, settings.output_size, generator=generator)
