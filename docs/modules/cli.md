# CLI

**Paths:** `cmd/trainmesh/main.go` (entry point), `internal/cli/` (commands).

## What it does

`main` creates a context that is cancelled on Ctrl-C or SIGTERM and runs the cobra command tree
from `cli.NewRoot()`. All logic lives in `internal/cli`, so commands are testable without a
process (`internal/cli/train_test.go` runs `plan` and `run` in-process).

The training ranks run in their own process group (see [launcher.md](launcher.md)), so a terminal
Ctrl-C would not reach them directly; cancelling the context is how the CLI stops them.

## Commands

| Command | File | What happens |
|---------|------|--------------|
| `trainmesh version` | `root.go` | Prints the CLI version |
| `trainmesh doctor` | `doctor.go` | Finds the runtime, runs `python -m trainmesh_runtime.probe`, prints a JSON report (PyTorch version, Gloo, torchrun, safetensors) and fails if PyTorch is missing |
| `trainmesh plan --config F` | `train.go` | Loads and validates the config, prints world size, threads per rank, effective global batch |
| `trainmesh run --config F [--resume DIR]` | `train.go` | Validates, checks CPU oversubscription, launches training on this machine |
| `trainmesh benchmark --config F [--output F]` | `train.go` | Same as `run` with a results file |
| `trainmesh cluster init [--name N]` / `status` | `cluster.go` | Writes / reads `.trainmesh/cluster.json` (local identity only; no registry yet) |

## `run` and `benchmark` in detail (`launchTraining`)

1. `config.Load` — parse and validate (see [config.md](config.md)); warn about legacy keys.
2. Refuse `execution.workers != 1` with `ErrMultiMachineUnsupported` (multi-machine launch is
   Phase 1).
3. Threads per rank: `execution.cpu_threads_per_rank`, else `OMP_NUM_THREADS`, else 1.
4. Refuse when ranks, or ranks × threads, exceed the logical CPU count.
5. `launcher.DiscoverRuntime` → interpreter and `python/` directory.
6. `launcher.Run` with a `launcher.Spec`; its exit status becomes the command's.

## Host abstraction

Commands read the environment through a small `host` struct (env lookup, cwd, executable dir,
GOOS, CPU count) created by `currentHost()`, which keeps OS access in one place.

## Errors

Errors are wrapped with context (`fmt.Errorf("…: %w", err)`) and printed once by `main` as
`trainmesh: <error>` with exit status 1. Callers can test for `config.ErrInvalid`,
`launcher.ErrRuntimeNotFound`, `launcher.ErrRendezvousPortUnavailable` and
`cli.ErrMultiMachineUnsupported` with `errors.Is`.
