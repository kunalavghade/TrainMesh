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

Tune `execution.processes_per_worker` and `training.batch.per_rank`. The effective global batch is their product with worker count and gradient accumulation. Gradient accumulation communicates once per optimizer step (`DDP.no_sync`), so raising `gradient_accumulation_steps` cuts gradient traffic proportionally.

Intra-op threads per rank come from `execution.cpu_threads_per_rank`, then `OMP_NUM_THREADS`, then 1. On the reference 12-core Mac, extra threads were slower for small models and at most ~7% faster at hidden=4096. Raise the value only after measuring with `trainmesh benchmark`. The CLI rejects configurations where ranks, or ranks × threads, exceed the logical CPU count.

## Checkpoints

Format v2 writes these files, and nothing is pickled:
- `model.safetensors`
- `optimizer.safetensors`
- `rng.safetensors` (one torch RNG state per rank)
- `state.json`
- `metadata.json`, with a SHA-256 for every other file

Resume checks every hash, the world size and the training shape before loading anything. Old v1 `checkpoint.pt` files are refused, because loading them would unpickle arbitrary objects.

## Config changes in 0.2

- The top-level `optimizer:` block is now `planner:`, and the results file is `run.results_path`. The old key still loads with a deprecation warning.
- New `execution.cpu_threads_per_rank` setting (0 = default).

## Supported config subset

Version 1, `data_parallel`, `cpu`, `gloo`, synthetic integer-label data, FP32, and Adam or SGD are supported. Non-synthetic data, CUDA/NCCL, other precision modes, and multi-host execution are rejected or explicitly reported as unavailable. See `examples/cpu-ddp.yaml`.
