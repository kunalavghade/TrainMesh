"""DiLoCo: islands train on their own, then average pseudo-gradients every H inner steps.

Inside an island (one torchrun world), ranks synchronise every step as usual. Every
`inner_steps` optimizer steps the island leader (island rank 0) sends the island's
pseudo-gradient `anchor - current` (anchor = weights at the start of the round) to the other
island leaders through the Rust `tmcomm` extension, which compresses it (int8/bf16 with error
feedback), averages over the islands that answered, and returns the mean. Every rank of every
island then applies the same outer step (Nesterov SGD, computed on the CPU in float32), so all
islands hold bit-identical weights after each round; tmcomm checks that with a digest of the
anchor in every contribution.

The coordinator tells each island who it is and where the other leaders are, through
environment variables set by tm-agent. Without them the job is a single island (useful locally).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .config import DilocoSettings
from .distributed import DistContext, broadcast_
from .errors import ConfigError, OuterSyncError

ISLAND_ENV = "TRAINMESH_ISLAND"
PEERS_ENV = "TRAINMESH_ISLAND_PEERS"
LISTEN_ENV = "TRAINMESH_TMCOMM_LISTEN"
# The agent's node identity (certificate with the node ID as DNS name, key, cluster CA).
TLS_ENVS = ("TRAINMESH_TLS_CERT", "TRAINMESH_TLS_KEY", "TRAINMESH_TLS_CA")
CONNECT_TIMEOUT_SECONDS = 10.0
# The first round also waits for the slowest island to start (load PyTorch, build data).
STARTUP_TIMEOUT_FACTOR = 3.0
STATUS_OK, STATUS_FAILED = 0.0, 1.0


@dataclass(frozen=True, slots=True)
class IslandSetup:
    island: int
    peers: list[tuple[int, str, str]]  # (island, host:port, node name)
    listen: str
    tls: tuple[str, str, str] | None
    quorum: int

    @property
    def islands(self) -> int:
        return len(self.peers)


def island_setup(settings: DilocoSettings, environ: dict[str, str] | None = None) -> IslandSetup:
    """Reads the island layout tm-agent passes; no layout means one local island."""
    env = os.environ if environ is None else environ
    raw_peers = env.get(PEERS_ENV)
    if not raw_peers:
        address = f"127.0.0.1:{_free_port()}"
        return IslandSetup(
            island=0, peers=[(0, address, "local")], listen=address, tls=None, quorum=1
        )
    try:
        peers = sorted(
            (int(p["island"]), str(p["address"]), str(p["name"])) for p in json.loads(raw_peers)
        )
        island = int(env[ISLAND_ENV])
        listen = env[LISTEN_ENV]
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"malformed island layout from the agent: {exc}") from exc
    tls_files = tuple(env.get(name, "") for name in TLS_ENVS)
    tls = tls_files if all(tls_files) else None
    quorum = settings.min_islands or len(peers) // 2 + 1
    return IslandSetup(island=island, peers=peers, listen=listen, tls=tls, quorum=quorum)


@dataclass(slots=True)
class OuterStats:
    rounds: int = 0
    seconds: float = 0.0
    bytes_sent: int = 0
    bytes_received: int = 0
    members: list[int] = field(default_factory=list)
    alive: list[int] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)


class OuterLoop:
    """Owns the anchor weights, the outer momentum and (on the island leader) the tmcomm group."""

    def __init__(
        self,
        model: nn.Module,
        settings: DilocoSettings,
        setup: IslandSetup,
        ctx: DistContext,
    ) -> None:
        self._settings = settings
        self._setup = setup
        self._ctx = ctx
        self._params = [p for p in model.parameters() if p.requires_grad]
        self._anchor = _flatten(self._params)
        self._momentum = torch.zeros_like(self._anchor)
        self._steps_since_sync = 0
        self.stats = OuterStats(alive=[peer[0] for peer in setup.peers])
        self._group = self._open_group() if ctx.is_primary else None

    def after_step(self) -> float:
        """Call after every inner optimizer step; returns seconds spent in an outer round."""
        self._steps_since_sync += 1
        if self._steps_since_sync < self._settings.inner_steps:
            return 0.0
        return self._sync()

    def finish(self) -> float:
        """Final round for steps since the last one, so every island ends with the same model."""
        seconds = self._sync() if self._steps_since_sync else 0.0
        if self._group is not None:
            self._group.close()
        return seconds

    def summary(self) -> dict[str, Any]:
        return {
            "island": self._setup.island,
            "islands": self._setup.islands,
            "quorum": self._setup.quorum,
            "inner_steps": self._settings.inner_steps,
            "compression": self._settings.compression,
            "outer_learning_rate": self._settings.outer_learning_rate,
            "outer_momentum": self._settings.outer_momentum,
            "rounds": self.stats.rounds,
            "outer_seconds": self.stats.seconds,
            "bytes_sent": self.stats.bytes_sent,
            "bytes_received": self.stats.bytes_received,
            "parameters": self._anchor.numel(),
            "last_members": self.stats.members,
            "alive": self.stats.alive,
            "rounds_history": self.stats.history,
        }

    def _open_group(self) -> Any:
        try:
            import tmcomm
        except ImportError as exc:
            raise ConfigError(
                "strategy=diloco needs the tmcomm extension: run `make native` "
                "(maturin builds native/tmcomm into the venv)"
            ) from exc
        round_timeout = self._settings.round_timeout_seconds
        return tmcomm.OuterGroup(
            island=self._setup.island,
            peers=self._setup.peers,
            listen=self._setup.listen,
            quorum=self._setup.quorum,
            max_values=self._anchor.numel(),
            codec=self._settings.compression,
            tls=self._setup.tls,
            round_timeout=round_timeout,
            startup_timeout=round_timeout * STARTUP_TIMEOUT_FACTOR,
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
        )

    def _sync(self) -> float:
        started = time.perf_counter()
        pseudo_gradient = self._anchor - _flatten(self._params)
        status = torch.tensor([STATUS_OK], dtype=torch.float64)
        failure = ""
        if self._group is not None:
            digest = hashlib.sha256(self._anchor.numpy().tobytes()).digest()
            try:
                report = self._group.all_reduce_mean(
                    self.stats.rounds, pseudo_gradient.numpy(), digest
                )
                self._record(report)
            except RuntimeError as exc:  # tmcomm errors subclass RuntimeError
                status[0] = STATUS_FAILED
                failure = f"{type(exc).__name__}: {exc}"
        # Island ranks wait on the leader; tell them whether the round worked before they read.
        broadcast_(status, self._ctx)
        if status.item() != STATUS_OK:
            raise OuterSyncError(failure or "the island leader's outer round failed")
        broadcast_(pseudo_gradient, self._ctx)
        self._outer_step(pseudo_gradient)
        self._steps_since_sync = 0
        if self._group is None:
            self.stats.rounds += 1
        seconds = time.perf_counter() - started
        self.stats.seconds += seconds
        return seconds

    def _outer_step(self, pseudo_gradient: torch.Tensor) -> None:
        """Nesterov SGD on the anchor, then load it into the model."""
        momentum = self._settings.outer_momentum
        self._momentum.mul_(momentum).add_(pseudo_gradient)
        update = pseudo_gradient.add(self._momentum, alpha=momentum)
        self._anchor.sub_(update, alpha=self._settings.outer_learning_rate)
        with torch.no_grad():
            offset = 0
            for param in self._params:
                count = param.numel()
                param.copy_(self._anchor[offset : offset + count].view_as(param))
                offset += count

    def _record(self, report: dict[str, Any]) -> None:
        self.stats.rounds += 1
        self.stats.bytes_sent += int(report["bytes_sent"])
        self.stats.bytes_received += int(report["bytes_received"])
        self.stats.members = list(report["members"])
        self.stats.alive = list(report["alive"])
        self.stats.history.append(
            {
                "round": int(report["round"]),
                "root": int(report["root"]),
                "members": list(report["members"]),
                "seconds": float(report["seconds"]),
            }
        )
        print(
            json.dumps(
                {
                    "event": "outer_round",
                    "round": int(report["round"]),
                    "island": self._setup.island,
                    "members": list(report["members"]),
                    "root": int(report["root"]),
                    "seconds": round(float(report["seconds"]), 4),
                    "bytes_sent": int(report["bytes_sent"]),
                }
            ),
            flush=True,
        )


def _flatten(params: list[nn.Parameter]) -> torch.Tensor:
    """A CPU float32 copy of the parameters, contiguous (tmcomm writes into its buffer)."""
    return torch.cat([p.detach().reshape(-1).to("cpu", torch.float32) for p in params]).contiguous()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
