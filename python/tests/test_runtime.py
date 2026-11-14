from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from trainmesh_runtime.runtime import load_config  # noqa: E402

CONFIG = """version: 1
execution:
  strategy: data_parallel
  backend: {device: cpu, communication: gloo}
training:
  dataset: {type: synthetic, samples: 64}
  batch: {per_rank: 8, gradient_accumulation_steps: 1}
  epochs: 1
  precision: fp32
"""
TORCHRUN_TIMEOUT_SECONDS = 120


def _write_config(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _read_metadata(checkpoint_dir: Path) -> dict:
    return json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _torchrun_two_ranks(config: Path, *runtime_args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["USE_LIBUV"] = "0"
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--node_rank=0",
        "--nproc_per_node=2",
        "--master_addr=127.0.0.1",
        f"--master_port={_free_port()}",
        "-m",
        "trainmesh_runtime",
        "--config",
        str(config),
        *runtime_args,
    ]
    try:
        return subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=TORCHRUN_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"torchrun timed out after {TORCHRUN_TIMEOUT_SECONDS}s\n"
            f"--- stdout ---\n{_decode(exc.stdout)}\n--- stderr ---\n{_decode(exc.stderr)}"
        )


def _decode(output: bytes | str | None) -> str:
    # TimeoutExpired always carries bytes, even with text=True.
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output or ""


def test_loads_supported_cpu_config(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml", CONFIG)
    assert load_config(config)["execution"]["backend"]["communication"] == "gloo"


def test_rejects_cuda_configuration(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml", CONFIG.replace("device: cpu", "device: cuda"))
    with pytest.raises(ValueError, match="CPU with Gloo"):
        load_config(config)


def test_legacy_optimizer_key_is_migrated(tmp_path: Path) -> None:
    legacy = CONFIG + "optimizer: {enabled: false, results: out.json}\n"
    loaded = load_config(_write_config(tmp_path / "config.yaml", legacy))
    assert "optimizer" not in loaded
    assert loaded["planner"] == {"enabled": False}
    assert loaded["run"]["results_path"] == "out.json"


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

    proc = _torchrun_two_ranks(config)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '"world_size": 2' in proc.stdout
    assert '"parameters_consistent": true' in proc.stdout
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
    resumed = _torchrun_two_ranks(config, "--resume", str(checkpoint_dir))

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    resumed_metadata = _read_metadata(checkpoint_dir)
    assert resumed_metadata["epoch"] == 2
    assert resumed_metadata["global_step"] == 8
