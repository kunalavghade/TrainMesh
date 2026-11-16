from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from trainmesh_runtime.runtime import load_config  # noqa: E402

CONFIG = """version: 1
execution:
  strategy: data_parallel
  workers: 1
  processes_per_worker: 2
  backend: {device: cpu, communication: gloo}
training:
  dataset: {type: synthetic, samples: 64}
  batch: {per_rank: 8, gradient_accumulation_steps: 1}
  epochs: 1
  precision: fp32
"""
RANK_TIMEOUT_SECONDS = 120
WORLD_SIZE = 2


@dataclass(frozen=True, slots=True)
class TwoRankRun:
    returncodes: list[int]
    rank0_stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return all(code == 0 for code in self.returncodes)


def _write_config(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _read_metadata(checkpoint_dir: Path) -> dict:
    return json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _rank_env(rank: int, master_port: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        OMP_NUM_THREADS="1",
        USE_LIBUV="0",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
        LOCAL_WORLD_SIZE=str(WORLD_SIZE),
    )
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    return env


def _run_two_ranks(config: Path, logs: Path, *runtime_args: str) -> TwoRankRun:
    """Start both ranks directly with the env torchrun would set.

    torchrun's agent reverse-resolves the host name at startup, which takes 35 s per lookup on
    GitHub's macOS runners. The launcher path is covered by scripts/e2e/local.sh instead.
    """
    logs.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    command = [sys.executable, "-m", "trainmesh_runtime", "--config", str(config), *runtime_args]
    processes: list[subprocess.Popen[bytes]] = []
    # Files, not pipes: a full pipe would block one rank while the other waits in a collective.
    for rank in range(WORLD_SIZE):
        with (
            (logs / f"rank{rank}.out").open("wb") as out,
            (logs / f"rank{rank}.err").open("wb") as err,
        ):
            processes.append(
                subprocess.Popen(command, env=_rank_env(rank, port), stdout=out, stderr=err)
            )
    try:
        returncodes = [p.wait(timeout=RANK_TIMEOUT_SECONDS) for p in processes]
    except subprocess.TimeoutExpired:
        for process in processes:
            process.kill()
            process.wait()
        pytest.fail(f"ranks timed out after {RANK_TIMEOUT_SECONDS}s\n{_read_logs(logs)}")
    return TwoRankRun(
        returncodes=returncodes,
        rank0_stdout=(logs / "rank0.out").read_text(encoding="utf-8", errors="replace"),
        stderr=_read_logs(logs, suffix="err"),
    )


def _read_logs(logs: Path, suffix: str = "*") -> str:
    parts = []
    for path in sorted(logs.glob(f"rank*.{suffix}")):
        parts.append(f"--- {path.name} ---\n{path.read_text(encoding='utf-8', errors='replace')}")
    return "\n".join(parts)


def test_loads_supported_cpu_config(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml", CONFIG)
    assert load_config(config).execution.backend.communication == "gloo"


def test_rejects_cuda_configuration(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml", CONFIG.replace("device: cpu", "device: cuda"))
    with pytest.raises(
        ValueError, match="execution.backend.device: this release supports device=cpu"
    ):
        load_config(config)


def test_legacy_optimizer_key_is_migrated(tmp_path: Path) -> None:
    legacy = CONFIG + "optimizer: {enabled: false, results: out.json}\n"
    loaded = load_config(_write_config(tmp_path / "config.yaml", legacy))
    assert dict(loaded.planner) == {"enabled": False}
    assert loaded.run.results_path == "out.json"


def test_two_rank_gloo_training_and_checkpoint(tmp_path: Path) -> None:
    if not torch.distributed.is_gloo_available():
        pytest.skip("this PyTorch build does not provide Gloo")
    checkpoint_dir = tmp_path / "checkpoints" / "latest"
    config = _write_config(
        tmp_path / "config.yaml",
        f"""version: 1
run: {{seed: 42}}
execution:
  strategy: data_parallel
  workers: 1
  processes_per_worker: 2
  backend: {{device: cpu, communication: gloo}}
training:
  model: {{input_size: 8, hidden_size: 16, output_size: 3}}
  dataset: {{type: synthetic, samples: 64, input_size: 8, output_size: 3}}
  batch: {{per_rank: 8, gradient_accumulation_steps: 1}}
  epochs: 1
  precision: fp32
  checkpoint: {{enabled: true, directory: {json.dumps(str(tmp_path / 'checkpoints'))}}}
""",
    )

    run = _run_two_ranks(config, tmp_path / "logs-first")

    assert run.succeeded, f"{run.returncodes}\n{run.rank0_stdout}\n{run.stderr}"
    assert '"world_size": 2' in run.rank0_stdout
    assert '"parameters_consistent": true' in run.rank0_stdout
    metadata = _read_metadata(checkpoint_dir)
    assert metadata["world_size"] == 2
    assert metadata["format_version"] == 2
    assert set(metadata["files"]) == {
        "model.safetensors",
        "optimizer.safetensors",
        "rng.safetensors",
        "state.json",
    }

    config.write_text(
        config.read_text(encoding="utf-8").replace("epochs: 1", "epochs: 2"), encoding="utf-8"
    )
    resumed = _run_two_ranks(config, tmp_path / "logs-resume", "--resume", str(checkpoint_dir))

    assert resumed.succeeded, f"{resumed.returncodes}\n{resumed.rank0_stdout}\n{resumed.stderr}"
    resumed_metadata = _read_metadata(checkpoint_dir)
    assert resumed_metadata["epoch"] == 2
    assert resumed_metadata["global_step"] == 8
