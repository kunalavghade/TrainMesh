# CPU backend setup

TrainMesh uses PyTorch DistributedDataParallel with the Gloo backend. CUDA and NCCL are intentionally not required or advertised.

## macOS setup (Apple Silicon or Intel)

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e 'python[test]'
go build -o bin/trainmesh ./cmd/trainmesh
./bin/trainmesh doctor
```

The doctor should show a PyTorch version, `gloo: true`, and `torchrun` via `python -m torch.distributed.run`. The Go launcher invokes the selected Python interpreter, so an independent `torchrun` executable is not required. On macOS, local jobs bind Gloo to loopback (`lo0`) to avoid routing local collective traffic through the Wi-Fi interface; a future multi-machine runner must instead set the correct reachable interface on each host.

## Run and inspect results

```bash
./bin/trainmesh plan --config examples/cpu-ddp.yaml
./bin/trainmesh run --config examples/cpu-ddp.yaml
./bin/trainmesh run --config examples/cpu-ddp.yaml --resume ./checkpoints/latest
cat checkpoints/latest/metadata.json
```

Tune `execution.processes_per_worker` and `training.batch.per_rank`. The effective global batch is their product with worker count and gradient accumulation. The CLI rejects process counts above the host's logical CPU count to avoid accidental oversubscription.

## Supported config subset

Version 1, `data_parallel`, `cpu`, `gloo`, synthetic integer-label data, FP32, and Adam or SGD are supported. Non-synthetic data, CUDA/NCCL, other precision modes, and multi-host execution are rejected or explicitly reported as unavailable. See `examples/cpu-ddp.yaml`.
