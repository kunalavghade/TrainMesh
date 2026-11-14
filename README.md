# TrainMesh

Adaptive distributed AI training engine written in Go and Python. Distribute model-training computation across machines and optimize compute, communication and training time.

TrainMesh currently runs **real single-host CPU DistributedDataParallel (DDP)** with PyTorch Gloo. It does not require a GPU. Cluster identity and local planning are implemented; remote enrollment, multi-machine process orchestration, automatic pilot selection, and fault recovery are not yet implemented. See [implementation status](docs/IMPLEMENTATION_STATUS.md) before treating this as a multi-machine release.

```mermaid
flowchart LR
  CLI[Go CLI / planner] -->|validated config, process lifecycle| Launcher[torchrun]
  Launcher --> R0[Python rank 0]
  Launcher --> R1[Python rank 1]
  R0 <-->|Gloo all-reduce| R1
  R0 --> Checkpoint[Local checkpoint]
  R0 --> Metrics[JSON metrics]
```

## Quick start

Requirements: Go 1.24+, Python 3.10+, CPU PyTorch with Gloo, and `uv` (or pip).

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e 'python[test]'
go build -o bin/trainmesh ./cmd/trainmesh
./bin/trainmesh doctor
./bin/trainmesh plan --config examples/cpu-ddp.yaml
./bin/trainmesh run --config examples/cpu-ddp.yaml
```

The example uses deterministic synthetic data, a small MLP, two CPU ranks, Gloo gradient synchronization, and saves `checkpoints/latest/checkpoint.pt`. The global batch is `per_rank × world_size × accumulation_steps`; this example has `64 × 2 × 1 = 128`.

## Commands

- `trainmesh doctor` inspects Python, PyTorch, Gloo, and torchrun.
- `trainmesh cluster init` creates a local cluster identity; `cluster status` reports the local host.
- `trainmesh plan --config ...` validates supported settings and prints the resolved batch/world size.
- `trainmesh run --config ...` launches local CPU training through torchrun.
- `trainmesh benchmark --config ... --output benchmark-results/run.json` records actual runtime metrics.

Remote workers are deliberately reported as unsupported until the authenticated agent and multi-node launcher are complete. A local multi-process test is not a substitute for multi-machine validation. Use [the manual Multipass and two-host guide](docs/MULTI_MACHINE_MANUAL_TEST.md) for reproducible validation steps and current environment limits.

## Docs

- [Architecture review and target design](docs/TRAINMESH_ARCHITECTURE.md) (start here)
- [CPU backend setup](docs/CPU_BACKEND.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Manual integration test](docs/MULTI_MACHINE_MANUAL_TEST.md)
- [Implementation status](docs/IMPLEMENTATION_STATUS.md)
- [Roadmap](docs/ROADMAP.md)

License: Apache-2.0.
