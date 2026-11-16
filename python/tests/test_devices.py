"""Device and precision selection, and the proportional batch sampler."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DistributedSampler  # noqa: E402

from trainmesh_runtime import devices  # noqa: E402
from trainmesh_runtime.data import GlobalBatchSampler  # noqa: E402
from trainmesh_runtime.errors import ConfigError  # noqa: E402

CPU = torch.device("cpu")


def test_precision_defaults_to_fp32_and_auto_never_picks_bf16_on_cpu() -> None:
    assert devices.resolve_precision("", CPU) == ("fp32", None)
    assert devices.resolve_precision("auto", CPU) == ("fp32", None)
    assert devices.resolve_precision("bf16", CPU) == ("bf16", torch.bfloat16)


def test_fp16_is_refused_on_cpu() -> None:
    with pytest.raises(ConfigError, match="fp16 needs a cuda or mps device"):
        devices.resolve_precision("fp16", CPU)


def test_auto_device_honours_the_agent_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(devices.DEVICE_OVERRIDE_ENV, "cpu")
    assert devices.resolve_device("auto", local_rank=0) == CPU


def test_unavailable_device_is_a_config_error() -> None:
    missing = next(
        (kind for kind in ("cuda", "mps") if kind not in devices.available_kinds()), None
    )
    if missing is None:
        pytest.skip("every device kind is available here")
    with pytest.raises(ConfigError, match="not available on this machine"):
        devices.resolve_device(missing, local_rank=0)


def test_gloo_is_used_unless_nccl_is_possible() -> None:
    assert devices.resolve_backend("auto", CPU) == "gloo"
    with pytest.raises(ConfigError, match="nccl needs a CUDA device"):
        devices.resolve_backend("nccl", CPU)


def test_mps_gradients_are_staged_through_host_memory() -> None:
    mps = devices.DeviceChoice(torch.device("mps"), "gloo", None, "fp32")
    assert mps.stages_to_host
    assert not devices.DeviceChoice(CPU, "gloo", None, "fp32").stages_to_host


@pytest.mark.parametrize("sizes", [[8, 8], [12, 4], [1, 15]])
def test_uneven_slices_cover_the_same_global_batches_as_distributed_sampler(
    sizes: list[int],
) -> None:
    dataset_size, seed, epoch = 64, 7, 3
    reference = []
    for rank in range(2):
        sampler = DistributedSampler(range(dataset_size), num_replicas=2, rank=rank, seed=seed)
        sampler.set_epoch(epoch)
        indices = list(sampler)
        reference.append([indices[i : i + 8] for i in range(0, len(indices), 8)])

    proportional = []
    for rank in range(2):
        batch_sampler = GlobalBatchSampler(dataset_size, sizes, rank, seed)
        batch_sampler.set_epoch(epoch)
        proportional.append([indices.tolist() for indices in batch_sampler])

    assert len(proportional[0]) == len(proportional[1]) == len(reference[0])
    for step in range(len(reference[0])):
        expected = set(reference[0][step]) | set(reference[1][step])
        got = proportional[0][step] + proportional[1][step]
        assert [len(proportional[r][step]) for r in range(2)] == sizes
        assert set(got) == expected and len(got) == len(expected)


def test_one_rank_sees_the_same_global_batches_as_any_split() -> None:
    single = GlobalBatchSampler(64, [16], rank=0, seed=7)
    halves = [GlobalBatchSampler(64, [4, 12], rank=r, seed=7) for r in range(2)]
    for step, whole in enumerate(single):
        parts = [list(sampler)[step].tolist() for sampler in halves]
        assert sorted(parts[0] + parts[1]) == sorted(whole.tolist())


def test_resize_keeps_the_global_batch() -> None:
    sampler = GlobalBatchSampler(64, [8, 8], rank=1, seed=0)
    sampler.resize([12, 4])
    assert [len(indices) for indices in sampler] == [4] * 4
    with pytest.raises(ValueError, match="global batch 16"):
        sampler.resize([8, 4])
