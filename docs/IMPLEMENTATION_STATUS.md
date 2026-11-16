# Implementation status

Updated 2026-11-14.

| Capability | Status | Evidence / limitation |
|---|---|---|
| Go CLI, version, doctor | Implemented | `cmd/trainmesh`; doctor checks local Python/PyTorch/Gloo |
| YAML config validation and effective batch calculation | Implemented | `internal/config`; Go unit tests |
| Real local CPU DDP/Gloo training | Implemented and executed | `python/trainmesh_runtime/runtime.py`; two local ranks completed three epochs on macOS CPU/Gloo |
| DistributedSampler, model gradient sync, loss aggregation | Implemented and tested | Two-rank Gloo test asserts a shared world size, equal post-training model hashes, sample-weighted loss, and successful checkpoint resume |
| Local checkpoint/resume | Implemented for single-host jobs | Format v2: safetensors + JSON (no pickle), streamed SHA-256 per file, per-rank RNG state, world-size and training-shape checks; v1 pickle checkpoints refused. Writes are synchronous; recovery across worker failure remains unimplemented |
| Gradient accumulation traffic | Implemented and tested | `DDP.no_sync` on non-final micro-batches; `python/tests/test_accumulation.py` counts all-reduces with a comm hook and checks equivalence with one large batch |
| Launcher process lifecycle | Implemented and tested | `internal/launcher`: ranks run in their own process group; Ctrl-C/SIGTERM → SIGTERM to the group → SIGKILL after a grace period; bounded retry when the rendezvous port is taken |
| Lint and CI | Configured | `.github/workflows/ci.yml` (Go + Python on Linux and macOS), `.golangci.yml`, ruff + black, `.pre-commit-config.yaml`; `make lint` runs them locally |
| Single config schema (protobuf) | Not implemented | Needs `protoc`/`buf`; Go and Python validators are still separate |
| Local benchmark JSON | Implemented and measured | `benchmark-results/local-two-rank.json`; one local smoke run, not a comparative benchmark |
| Cluster identity and local status | Implemented | `.trainmesh/cluster.json`; no worker registry |
| Enrollment, authenticated agent, heartbeats, revocation | Not implemented | No remote execution surface is exposed |
| Multi-machine DDP | Runtime works multi-node; CLI launch not implemented | Runtime trains across nodes when started by `torchrun` directly: verified with two node processes on one Mac (same final loss as one node). The minikube `multi` scenario has not completed yet (`single-2x` was OOM-killed first; see `docs/E2E_TESTING.md`). No agent coordination or CLI multi-node launch; physical two-machine run still pending |
| End-to-end tests | Implemented | `scripts/e2e/local.sh` (CLI, in CI on Linux), `scripts/e2e/minikube.sh` (2-node cluster, 1-vs-multi machine comparison with loss-parity check), `scripts/e2e/manual-node.sh` (physical machines) |
| Adaptive pilot optimizer/network profiler | Not implemented | Planner reports configuration, not a measured optimum |
| GPU/NCCL | Unsupported | No GPU required; CPU/Gloo only |
| Multipass VM test | Blocked by host installer privilege (minikube used instead) | Homebrew downloaded Multipass 1.17.1, but its package installer requires interactive `sudo`; this session cannot supply a system password. No VM or Multipass test was run. See `docs/MULTI_MACHINE_MANUAL_TEST.md` |

Host observed during implementation: macOS arm64, Go 1.26.3, Python 3.12, Docker 28.5.1, PyTorch 2.15.0 CPU/Gloo. Gloo required explicit `lo0` selection for local execution. The single-host smoke measurement is recorded in JSON; it is not compared with another configuration and is not evidence that more workers improve performance.
