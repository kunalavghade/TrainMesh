from __future__ import annotations

import os
import json
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from trainmesh_runtime.runtime import load_config


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


def test_loads_supported_cpu_config() -> None:
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "config.yaml"
        config.write_text(CONFIG, encoding="utf-8")
        assert load_config(config)["execution"]["backend"]["communication"] == "gloo"


def test_rejects_cuda_configuration() -> None:
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "config.yaml"
        config.write_text(CONFIG.replace("device: cpu", "device: cuda"), encoding="utf-8")
        with pytest.raises(ValueError, match="CPU with Gloo"):
            load_config(config)


def test_two_rank_gloo_training_and_checkpoint() -> None:
    if not torch.distributed.is_gloo_available():
        pytest.skip("this PyTorch build does not provide Gloo")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / "config.yaml"
        config.write_text(
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
  checkpoint: {{enabled: true, directory: {json.dumps(str(root / 'checkpoints'))}}}
""",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        env["OMP_NUM_THREADS"] = "1"
        env["USE_LIBUV"] = "0"
        if sys.platform == "darwin":
            env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        proc = subprocess.run(
            [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--node_rank=0", "--nproc_per_node=2", "--master_addr=127.0.0.1", f"--master_port={port}", "-m", "trainmesh_runtime", "--config", str(config)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert '"world_size": 2' in proc.stdout
        assert '"parameters_consistent": true' in proc.stdout
        metadata = json.loads((root / "checkpoints/latest/metadata.json").read_text(encoding="utf-8"))
        assert metadata["world_size"] == 2
        assert metadata["format_version"] == 2
        assert set(metadata["files"]) == {
            "model.safetensors",
            "optimizer.safetensors",
            "rng.safetensors",
            "state.json",
        }
        config.write_text(config.read_text(encoding="utf-8").replace("epochs: 1", "epochs: 2"), encoding="utf-8")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        resumed = subprocess.run(
            [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--node_rank=0", "--nproc_per_node=2", "--master_addr=127.0.0.1", f"--master_port={port}", "-m", "trainmesh_runtime", "--config", str(config), "--resume", str(root / "checkpoints/latest")],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        resumed_metadata = json.loads((root / "checkpoints/latest/metadata.json").read_text(encoding="utf-8"))
        assert resumed_metadata["epoch"] == 2
        assert resumed_metadata["global_step"] == 8
