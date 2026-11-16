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


def _run_two_ranks(
    config: Path, logs: Path, *runtime_args: str, rank_env: dict[int, dict[str, str]] | None = None
) -> TwoRankRun:
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
            env = _rank_env(rank, port) | (rank_env or {}).get(rank, {})
            processes.append(subprocess.Popen(command, env=env, stdout=out, stderr=err))
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


def test_rejects_nccl_without_cuda(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "config.yaml", CONFIG.replace("communication: gloo", "communication: nccl")
    )
    with pytest.raises(ValueError, match="communication=nccl needs device=cuda"):
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


def _teacher_config(tmp_path: Path, name: str, *, device: str, extra_batch: str = "") -> Path:
    return _write_config(
        tmp_path / f"{name}.yaml",
        f"""version: 1
run: {{seed: 42}}
execution:
  strategy: data_parallel
  workers: 1
  processes_per_worker: 2
  backend: {{device: {device}, communication: gloo}}
training:
  model: {{input_size: 16, hidden_size: 32, output_size: 4}}
  dataset: {{type: teacher, samples: 512, input_size: 16, output_size: 4}}
  batch: {{per_rank: 16{extra_batch}}}
  epochs: 2
  checkpoint: {{enabled: false}}
  validation: {{samples: 256}}
""",
    )


def _result(run: TwoRankRun) -> dict:
    assert run.succeeded, f"{run.returncodes}\n{run.rank0_stdout}\n{run.stderr}"
    start = run.rank0_stdout.index("{\n")
    return json.loads(run.rank0_stdout[start:])


def test_uneven_rank_batches_train_the_same_model_as_uniform(tmp_path: Path) -> None:
    uniform = _result(
        _run_two_ranks(_teacher_config(tmp_path, "uniform", device="cpu"), tmp_path / "u")
    )
    uneven = _result(
        _run_two_ranks(
            _teacher_config(
                tmp_path, "uneven", device="cpu", extra_batch=", rank_batch_sizes: [24, 8]"
            ),
            tmp_path / "v",
        )
    )

    assert uniform["gradient_sync"] == "ddp" and uneven["gradient_sync"] == "allreduce"
    assert [r["batch_size"] for r in uneven["ranks"]] == [24, 8]
    assert abs(uniform["final_loss"] - uneven["final_loss"]) < 1e-5
    assert uniform["validation"]["final_accuracy"] == uneven["validation"]["final_accuracy"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_cpu_and_mps_ranks_train_one_model(tmp_path: Path) -> None:
    cpu_only = _result(
        _run_two_ranks(_teacher_config(tmp_path, "cpu", device="cpu"), tmp_path / "c")
    )
    mixed = _result(
        _run_two_ranks(
            _teacher_config(tmp_path, "mixed", device="auto"),
            tmp_path / "m",
            rank_env={0: {"TRAINMESH_DEVICE": "cpu"}, 1: {"TRAINMESH_DEVICE": "mps"}},
        )
    )

    assert [r["device"] for r in mixed["ranks"]] == ["cpu", "mps"]
    assert mixed["mixed_devices"] and mixed["gradient_sync"] == "allreduce"
    assert mixed["parameters_consistent"]
    assert mixed["max_parameter_drift"] < 1e-4
    assert abs(cpu_only["final_loss"] - mixed["final_loss"]) < 1e-3
    assert (
        abs(cpu_only["validation"]["final_accuracy"] - mixed["validation"]["final_accuracy"])
        <= 0.01
    )


def _balance_config(tmp_path: Path, name: str, batch: str) -> Path:
    # Large enough that step time grows with the batch: on a tiny model fixed per-step cost
    # dominates and a measured split has nothing to correct.
    return _write_config(
        tmp_path / f"{name}.yaml",
        f"""version: 1
run: {{seed: 42}}
execution:
  strategy: data_parallel
  workers: 1
  processes_per_worker: 2
  backend: {{device: cpu, communication: gloo}}
training:
  model: {{input_size: 64, hidden_size: 1024, output_size: 4}}
  dataset: {{type: teacher, samples: 4096, input_size: 64, output_size: 4}}
  batch: {{per_rank: 256{batch}}}
  epochs: 2
  checkpoint: {{enabled: false}}
  validation: {{samples: 512}}
""",
    )


def test_throughput_balance_resplits_without_changing_the_model(tmp_path: Path) -> None:
    uniform = _result(_run_two_ranks(_balance_config(tmp_path, "uniform", ""), tmp_path / "u"))
    balanced = _result(
        _run_two_ranks(
            _balance_config(
                tmp_path, "balanced", ", balance: throughput, rank_batch_sizes: [448, 64]"
            ),
            tmp_path / "b",
        )
    )

    history = balanced["balance_history"]
    assert balanced["balance"] == "throughput" and balanced["gradient_sync"] == "allreduce"
    assert len(history) == balanced["epochs"]
    assert history[0]["rank_batch_sizes"] == [448, 64]
    # Two equal CPU ranks: after one measured epoch the 448/64 guess moves toward an even split.
    assert history[1]["rank_batch_sizes"][0] < 448
    assert all(sum(epoch["rank_batch_sizes"]) == 512 for epoch in history)
    assert abs(uniform["final_loss"] - balanced["final_loss"]) < 1e-5
    assert uniform["validation"]["final_accuracy"] == balanced["validation"]["final_accuracy"]
