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

## Config changes in 0.3

- One schema for Go and Python with the same validation messages; unknown keys are now errors.
- `execution.workers` and `processes_per_worker` are now checked by the runtime too (≥ 1).
- New `training.dataset.type: teacher` (learnable) and `training.validation.samples`.
- Epoch events and results include `val_loss` / `val_accuracy` when validation is enabled.
- New runtime dependencies: `protobuf` 7.x and `protovalidate` 1.x.

## Config changes in 0.2

- The top-level `optimizer:` block is now `planner:`, and the results file is `run.results_path`. The old key still loads with a deprecation warning.
- New `execution.cpu_threads_per_rank` setting (0 = default).

## Supported config subset

The full schema and its rules are in `proto/trainmesh/v1/jobspec.proto` (explained in
[modules/config.md](modules/config.md)); the CLI and the runtime both reject anything outside it,
with the same messages. Supported today: version 1, `data_parallel`, `cpu` + `gloo`, MLP models,
the `synthetic` (random labels, throughput) and `teacher` (learnable, for accuracy) datasets,
optional held-out validation (`training.validation.samples`), FP32, Adam or SGD. CUDA/NCCL, other
precisions and CLI-launched multi-machine jobs are rejected. See `examples/cpu-ddp.yaml`.

## Developer toolchain (only to change the schema)

Generated code is committed, so building and testing need only Go, Python and uv. To edit
`jobspec.proto` and run `make generate` you also need:

```bash
brew install buf protobuf
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
```

Rust (for the Phase 3 `tmcomm` layer) comes from Homebrew's `rustup`, which is not on `PATH` by
default: add `export PATH="/opt/homebrew/opt/rustup/bin:$PATH"` to your shell profile.
