# Launcher

**Path:** `internal/launcher/` — `launcher.go` (args, env, supervision, port retry),
`discover.go` (finding Python and the runtime), `procgroup_unix.go` / `procgroup_windows.go`.

## Responsibilities

1. Find the Python interpreter and the `python/` directory that holds `trainmesh_runtime`.
2. Turn a `Spec` into a command line and environment.
3. Run it as a supervised **process group** so that cancelling stops every rank.
4. Survive the rendezvous-port race with a bounded retry.

## Finding the runtime (`DiscoverRuntime`)

| Order | Python interpreter | Runtime directory |
|-------|--------------------|-------------------|
| 1 | `TRAINMESH_PYTHON` | `TRAINMESH_RUNTIME_DIR` |
| 2 | `<project>/.venv/bin/python` (or `Scripts\python.exe`) | `<project>/python` |
| 3 | `python3` on `PATH` | — (error `ErrRuntimeNotFound`) |

`<project>` is the first ancestor of the working directory, then of the executable's directory,
that contains `python/trainmesh_runtime/__init__.py`.

## Command and environment (`Spec.Args`, `Spec.Env`)

- 1 rank: `python -m trainmesh_runtime --config <abs path> [extra args]`.
- N ranks: `python -m torch.distributed.run --nnodes=1 --node_rank=0 --nproc_per_node=N
  --master_addr=127.0.0.1 --master_port=<port> -m trainmesh_runtime …`.
  `--standalone` is deliberately **not** used: its c10d store ignores `USE_LIBUV=0` and hangs on
  macOS.
- Environment added on top of the caller's: `PYTHONPATH=<runtime dir>[:existing]`,
  `OMP_NUM_THREADS=<threads>`, `USE_LIBUV=0`, and on macOS `GLOO_SOCKET_IFNAME=lo0` unless the
  user already set an interface.

## Supervision (`supervise`)

```text
start child with Setpgid ──► wait
        │                      ├─ exits ──► return exit status (non-zero = error)
        │                      └─ ctx cancelled ──► SIGTERM to the whole group
        │                                            └─ grace period (10 s) ──► SIGKILL group
        └─ always on return: SIGKILL the group (reaps ranks orphaned by a crashed torchrun)
```

On Windows there are no POSIX process groups; the fallback kills the parent process.

## Port retry (`Run`)

The launcher must choose `--master_port` before torchrun binds it, so another program can grab it
in between. `Run` picks a free loopback port, starts torchrun, and watches stderr through a
`conflictDetector` for "address already in use" (matched across write boundaries). If it appears,
it relaunches on a new port, at most `maxPortAttempts` (3) times, then returns
`ErrRendezvousPortUnavailable`.

## Threads per rank

`ResolveThreadsPerRank(configured, OMP_NUM_THREADS)`: explicit config wins, then the environment,
then `DefaultThreadsPerRank = 1`. On the reference 12-core Mac, extra threads were 5–10% slower
for small models and at most ~7% faster at hidden size 4096, so 1 is the default.

## Tests

`launcher_test.go` (args, env per OS, threads, port-conflict detection, discovery) and
`procgroup_unix_test.go` (cancellation kills grandchildren; non-zero exit is reported).
