"""Largest-remainder apportionment and per-epoch re-splitting (no process group needed)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from trainmesh_runtime.balance import EpochBalance, apportion  # noqa: E402


@pytest.mark.parametrize(
    ("total", "weights", "expected"),
    [
        # Same cases as internal/coordinator/planner_test.go: both sides must split identically.
        (128, [1, 1], [64, 64]),
        (128, [3, 1], [96, 32]),
        (128, [512000, 270000], [84, 44]),
        (10, [1, 1, 1], [4, 3, 3]),
        (3, [1000, 1, 1], [1, 1, 1]),
        (100, [1e6, 1, 1], [98, 1, 1]),
        (7, [2, 1], [5, 2]),
    ],
)
def test_apportion_matches_the_coordinator(
    total: int, weights: list[float], expected: list[int]
) -> None:
    assert apportion(total, weights) == expected


def test_next_split_follows_measured_speed() -> None:
    # Rank 1 took three times as long for the same 8 samples: it is a third as fast.
    epoch = EpochBalance(epoch=1, rank_batch_sizes=[8, 8], step_compute_seconds=[1.0, 3.0])
    assert epoch.compute_imbalance == pytest.approx(3.0)
    assert epoch.next_split() == [12, 4]


def test_a_balanced_split_stays_put() -> None:
    # Both ranks took equally long: the split already matches their speeds.
    epoch = EpochBalance(epoch=2, rank_batch_sizes=[12, 4], step_compute_seconds=[1.0, 1.0])
    assert epoch.next_split() == [12, 4]


def test_small_imbalance_is_left_alone() -> None:
    # 5% apart is within REBALANCE_THRESHOLD: measurement noise must not move the split.
    epoch = EpochBalance(epoch=3, rank_batch_sizes=[12, 4], step_compute_seconds=[1.05, 1.0])
    assert epoch.next_split() == [12, 4]
