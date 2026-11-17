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
| `trainmesh run --config F --coordinator H:P [--output F] [--timeout D]` | `mesh.go` | Submits the job to `meshd`, prints progress per node rank, prints rank 0's result JSON; Ctrl-C or timeout cancels it everywhere |
| `trainmesh token create [--ttl 15m] [--uses 1]` | `mesh.go` | Prints a join string for new machines (`tm-agent --join …`) |
| `trainmesh nodes` | `mesh.go` | Machines with state, address, CPUs, memory, PyTorch/runtime version, device, measured samples/s per device, island, current job |
| `trainmesh job get ID` / `job cancel ID` | `mesh.go` | Job state, strategy, islands (state, master, tmcomm address), collective backend, per-rank batch split, per-node assignments, result; or cancel it |
| `trainmesh cluster init [--name N]` / `status` | `cluster.go` | Local identity file only (predates the mesh; kept for compatibility) |

Mesh commands talk to `meshd` with the admin certificate from `--admin-dir` (default
`~/.trainmesh/meshd`, i.e. `meshd`'s state dir) at `--coordinator` (default `127.0.0.1:7443`).
`TRAINMESH_ADMIN_DIR` and `TRAINMESH_COORDINATOR` change the defaults; `run` uses the mesh only
when `--coordinator` is given explicitly. See [mesh.md](mesh.md).

## `run` and `benchmark` in detail (`launchTraining`)

1. `config.Load` — parse and validate (see [config.md](config.md)); warn about legacy keys.
2. Without `--coordinator`, refuse `execution.workers != 1` with `ErrMultiMachineUnsupported`,
   whose message says how to run across machines.
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
