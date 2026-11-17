"""DiLoCo: island layout, island data sharding, and real islands exchanging through tmcomm."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
tmcomm = pytest.importorskip("tmcomm", reason="build the extension with `make native`")

from trainmesh_runtime.config import parse_config, training_settings  # noqa: E402
from trainmesh_runtime.data import GlobalBatchSampler  # noqa: E402
from trainmesh_runtime.diloco import (  # noqa: E402
    ISLAND_ENV,
    LISTEN_ENV,
    PEERS_ENV,
    island_setup,
)
from trainmesh_runtime.errors import ConfigError  # noqa: E402

ISLAND_TIMEOUT_SECONDS = 180


def _config(tmp_path: Path, *, epochs: int = 3, inner_steps: int = 8, extra: str = "") -> Path:
    path = tmp_path / "diloco.yaml"
    path.write_text(
        f"""version: 1
run: {{seed: 42}}
execution:
  strategy: diloco
  workers: 3
  processes_per_worker: 1
  backend: {{device: cpu, communication: gloo}}
  diloco: {{inner_steps: {inner_steps}, compression: int8{extra}}}
training:
  model: {{input_size: 32, hidden_size: 128, output_size: 4}}
  dataset: {{type: teacher, samples: 6144, input_size: 32, output_size: 4}}
  optimizer: {{name: adam, learning_rate: 0.003}}
  batch: {{per_rank: 64}}
  epochs: {epochs}
  checkpoint: {{enabled: false}}
  validation: {{samples: 1024}}
""",
        encoding="utf-8",
    )
    return path


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _layout(islands: int) -> list[dict]:
    return [
        {"island": i, "address": f"127.0.0.1:{_free_port()}", "name": f"node-{i}"}
        for i in range(islands)
    ]


def _start_island(config: Path, layout: list[dict], island: int, out: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        OMP_NUM_THREADS="1",
        **{
            ISLAND_ENV: str(island),
            PEERS_ENV: json.dumps(layout),
            LISTEN_ENV: layout[island]["address"],
        },
    )
    return subprocess.Popen(
        [sys.executable, "-m", "trainmesh_runtime", "--config", str(config), "--output", str(out)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_layout_comes_from_the_agent_environment() -> None:
    settings = training_settings(parse_config(_document())).diloco
    layout = [
        {"island": 1, "address": "10.0.0.2:29701", "name": "node-b"},
        {"island": 0, "address": "10.0.0.1:29700", "name": "node-a"},
        {"island": 2, "address": "10.0.0.3:29702", "name": "node-c"},
    ]
    setup = island_setup(
        settings,
        {
            PEERS_ENV: json.dumps(layout),
            ISLAND_ENV: "1",
            LISTEN_ENV: "0.0.0.0:29701",
            "TRAINMESH_TLS_CERT": "/state/node.pem",
            "TRAINMESH_TLS_KEY": "/state/node-key.pem",
            "TRAINMESH_TLS_CA": "/state/ca.pem",
        },
    )
    assert setup.island == 1 and setup.islands == 3 and setup.quorum == 2
    assert setup.peers[0] == (0, "10.0.0.1:29700", "node-a")
    assert setup.tls == ("/state/node.pem", "/state/node-key.pem", "/state/ca.pem")

    local = island_setup(settings, {})
    assert local.islands == 1 and local.quorum == 1 and local.tls is None

    with pytest.raises(ConfigError, match="malformed island layout"):
        island_setup(settings, {PEERS_ENV: "[{}]", ISLAND_ENV: "0", LISTEN_ENV: "x"})


def test_islands_see_disjoint_data_that_covers_one_epoch() -> None:
    size, batch, islands = 96, 8, 3
    per_island = [
        [
            indices.tolist()
            for indices in GlobalBatchSampler(size, [batch], 0, 5, island=k, islands=3)
        ]
        for k in range(islands)
    ]
    single = [indices.tolist() for indices in GlobalBatchSampler(size, [batch * 3], 0, 5)]
    seen = [i for island in per_island for step in island for i in step]
    assert len(seen) == len(set(seen)) == size
    # Step s of island k is the k-th third of what one big group would see at step s.
    for step in range(len(single)):
        assert sorted(sum((per_island[k][step] for k in range(islands)), [])) == sorted(
            single[step]
        )


def test_three_islands_train_one_model(tmp_path: Path) -> None:
    config = _config(tmp_path)
    layout = _layout(3)
    outputs = [tmp_path / f"island{k}.json" for k in range(3)]
    processes = [_start_island(config, layout, k, outputs[k]) for k in range(3)]
    logs = [p.communicate(timeout=ISLAND_TIMEOUT_SECONDS)[0] for p in processes]
    for process, log in zip(processes, logs, strict=True):
        assert process.returncode == 0, log[-3000:]
    results = [json.loads(path.read_text()) for path in outputs]

    # 6144 samples / 3 islands / 64 per step = 32 inner steps per epoch, 96 in all: 12 rounds.
    for result in results:
        diloco = result["diloco"]
        assert result["strategy"] == "diloco"
        assert diloco["rounds"] == 12 and diloco["last_members"] == [0, 1, 2]
        assert result["validation"]["final_accuracy"] >= 0.85
        assert diloco["bytes_sent"] > 0
    assert len({r["model_sha256"] for r in results}) == 1, "every island must end identical"
    # int8: about one byte per parameter per round, not four.
    member = results[1]["diloco"]
    assert member["bytes_sent"] < member["parameters"] * 1.1 * member["rounds"]


def test_losing_an_island_leaves_a_quorum_training(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=4)
    layout = _layout(3)
    outputs = [tmp_path / f"island{k}.json" for k in range(3)]
    processes = [_start_island(config, layout, k, outputs[k]) for k in range(3)]

    # Kill island 2 as soon as it has finished its first outer round.
    victim = processes[2]
    deadline = time.monotonic() + ISLAND_TIMEOUT_SECONDS
    assert victim.stdout is not None
    for line in victim.stdout:
        if '"event": "outer_round"' in line or time.monotonic() > deadline:
            break
    victim.kill()
    victim.wait()

    logs = [p.communicate(timeout=ISLAND_TIMEOUT_SECONDS)[0] for p in processes[:2]]
    for process, log in zip(processes[:2], logs, strict=True):
        assert process.returncode == 0, log[-3000:]
    results = [json.loads(outputs[k].read_text()) for k in range(2)]
    for result in results:
        assert result["diloco"]["last_members"] == [0, 1]
        assert result["diloco"]["alive"] == [0, 1]
        assert result["validation"]["final_accuracy"] >= 0.85
    assert results[0]["model_sha256"] == results[1]["model_sha256"]


def test_a_single_local_island_needs_no_layout(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=1)
    out = tmp_path / "single.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    for name in (ISLAND_ENV, PEERS_ENV, LISTEN_ENV):
        env.pop(name, None)
    run = subprocess.run(
        [sys.executable, "-m", "trainmesh_runtime", "--config", str(config), "--output", str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=ISLAND_TIMEOUT_SECONDS,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    diloco = json.loads(out.read_text())["diloco"]
    assert diloco["islands"] == 1 and diloco["rounds"] == 12


def _document() -> dict:
    return {
        "version": 1,
        "execution": {
            "strategy": "diloco",
            "workers": 3,
            "processes_per_worker": 1,
            "backend": {"device": "cpu", "communication": "gloo"},
        },
        "training": {
            "dataset": {"type": "teacher", "samples": 64},
            "batch": {"per_rank": 8},
            "epochs": 1,
        },
    }
