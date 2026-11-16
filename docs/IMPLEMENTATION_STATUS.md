# Implementation status

Updated 2026-11-16. Phase 0 is complete; Phases 1–6 have not started (see `docs/TRAINMESH_ARCHITECTURE.md` §10).

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
| Single config schema (protobuf) | Implemented and tested | `proto/trainmesh/v1/jobspec.proto` defines shape and rules (protovalidate); Go (`internal/config`) and Python (`trainmesh_runtime/config.py`) use generated types and the same rules; `make generate` regenerates |
| Learnable task and validation | Implemented and tested | `teacher` dataset, held-out validation set, per-epoch `val_loss`/`val_accuracy` reduced across ranks; `python/tests/test_learning.py` requires ≥ 85% accuracy (measured ≈ 92–94%) |
| Local benchmark JSON | Implemented and measured | `benchmark-results/local-two-rank.json`; one local smoke run, not a comparative benchmark |
| Cluster identity and local status | Implemented | `.trainmesh/cluster.json`; no worker registry |
| Enrollment, authenticated agent, heartbeats, revocation | Not implemented | No remote execution surface is exposed |
| Multi-machine DDP | Runtime proven multi-node; CLI launch not implemented | Runtime trains across nodes when started by `torchrun` directly. Proven on a 2-node minikube cluster and with two node processes on one Mac: identical validation accuracy (91.70%) and loss as one node. Not yet run on two physical machines. No agent or CLI multi-node launch (Phase 1) |
| End-to-end tests | Implemented and run | `scripts/e2e/local.sh` (CLI incl. learning check; CI on Linux), `scripts/e2e/minikube.sh` (2-node cluster; 1-vs-multi machine comparison with loss, accuracy and learning checks; memory preflight), `scripts/e2e/manual-node.sh` (physical machines) |
| Adaptive pilot optimizer/network profiler | Not implemented | Planner reports configuration, not a measured optimum |
| GPU/NCCL | Unsupported | No GPU required; CPU/Gloo only. Apple MPS is available on the dev Mac for Phase 2 |
| Multipass VM test | Blocked by host installer privilege (minikube used instead) | Homebrew downloaded Multipass 1.17.1, but its package installer requires interactive `sudo`; this session cannot supply a system password. No Multipass VM test was run; `scripts/e2e/manual-node.sh` works inside VMs as on physical machines (`docs/E2E_TESTING.md`) |

Host observed during implementation: macOS arm64, Go 1.26.3, Python 3.12, Docker 28.5.1, PyTorch 2.15.0 CPU/Gloo, minikube 1.39.0, buf 1.75.0, protoc 36.2, Rust 1.100. Gloo required explicit `lo0` selection for local execution. The single-host smoke measurement is recorded in JSON; it is not compared with another configuration and is not evidence that more workers improve performance.
