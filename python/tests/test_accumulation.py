"""Gradient accumulation must communicate once per optimizer step, not once per micro-batch.

No `from __future__ import annotations` here: DDP inspects the comm hook's real annotations.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from trainmesh_runtime.devices import DeviceChoice  # noqa: E402
from trainmesh_runtime.distributed import DistContext  # noqa: E402
from trainmesh_runtime.runtime import MLP, StepPlan, train_epoch  # noqa: E402
from trainmesh_runtime.sync import DDPSync  # noqa: E402

INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE = 8, 16, 3


@pytest.fixture(scope="module")
def single_rank_group(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("this PyTorch build does not provide Gloo")
    monkeypatch = pytest.MonkeyPatch()
    if sys.platform == "darwin":
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    store = Path(tmp_path_factory.mktemp("pg")) / "store"
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()
    monkeypatch.undo()


def _loader(samples: int, batch_size: int) -> DataLoader:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(samples, INPUT_SIZE, generator=generator)
    target = torch.randint(OUTPUT_SIZE, (samples,), generator=generator)
    return DataLoader(TensorDataset(features, target), batch_size=batch_size, shuffle=False)


def _ddp_plan(accumulation_steps: int, batch: int) -> StepPlan:
    cpu = DeviceChoice(
        device=torch.device("cpu"), collective_backend="gloo", autocast_dtype=None, precision="fp32"
    )
    ctx = DistContext(rank=0, world_size=1, local_rank=0, local_world_size=1)
    return StepPlan(
        device=cpu.device,
        sync=DDPSync(ctx, cpu),
        accumulation_steps=accumulation_steps,
        local_batch=batch,
        global_batch=batch,
    )


def _ddp_model_with_counter() -> tuple[DistributedDataParallel, dict[str, int]]:
    torch.manual_seed(0)
    model = DistributedDataParallel(MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE))
    calls = {"all_reduce": 0}

    def counting_hook(
        state: dict[str, int], bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        state["all_reduce"] += 1
        future: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        future.set_result(bucket.buffer())
        return future

    model.register_comm_hook(calls, counting_hook)
    return model, calls


@pytest.mark.parametrize(
    ("batches", "accumulation_steps", "expected_steps"),
    [(8, 1, 8), (8, 4, 2), (6, 4, 2)],
    ids=["no-accumulation", "full-windows", "partial-final-window"],
)
@pytest.mark.usefixtures("single_rank_group")
def test_all_reduce_runs_once_per_optimizer_step(
    batches: int, accumulation_steps: int, expected_steps: int
) -> None:
    model, calls = _ddp_model_with_counter()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    stats = train_epoch(
        model, _loader(batches * 4, 4), optimizer, criterion, _ddp_plan(accumulation_steps, 4)
    )

    assert stats.optimizer_steps == expected_steps
    assert calls["all_reduce"] == expected_steps
    assert stats.samples == batches * 4


@pytest.mark.usefixtures("single_rank_group")
def test_accumulated_step_matches_one_large_batch() -> None:
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    accumulated, _ = _ddp_model_with_counter()
    large_batch, _ = _ddp_model_with_counter()

    train_epoch(
        accumulated,
        _loader(16, 4),
        torch.optim.SGD(accumulated.parameters(), lr=0.1),
        criterion,
        _ddp_plan(4, 4),
    )
    train_epoch(
        large_batch,
        _loader(16, 16),
        torch.optim.SGD(large_batch.parameters(), lr=0.1),
        criterion,
        _ddp_plan(1, 16),
    )

    for left, right in zip(accumulated.parameters(), large_batch.parameters(), strict=True):
        torch.testing.assert_close(left, right)
