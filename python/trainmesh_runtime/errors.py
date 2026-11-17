"""Typed errors raised by the TrainMesh runtime; callers branch on type, not message text."""

from __future__ import annotations


class TrainMeshError(Exception):
    """Base class for every expected runtime failure."""

    code = "TRAINMESH_ERROR"


class ConfigError(TrainMeshError, ValueError):
    """The job configuration is invalid or unsupported. Not retryable."""

    code = "CONFIG_INVALID"


class CheckpointError(TrainMeshError, ValueError):
    """A checkpoint is missing, corrupted, unsafe or incompatible. Not retryable."""

    code = "CHECKPOINT_INVALID"


class DivergenceError(TrainMeshError, RuntimeError):
    """Replicas that must hold identical parameters do not."""

    code = "REPLICA_DIVERGENCE"


class OuterSyncError(TrainMeshError, RuntimeError):
    """The DiLoCo exchange between islands failed (quorum lost, divergence, excluded)."""

    code = "OUTER_SYNC_FAILED"
