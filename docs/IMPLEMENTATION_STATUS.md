# Implementation status

Updated 2026-11-14.

| Capability | Status | Evidence / limitation |
|---|---|---|
| Go CLI, version, doctor | Implemented | `cmd/trainmesh`; doctor checks local Python/PyTorch/Gloo |
| YAML config validation and effective batch calculation | Implemented | `internal/config`; Go unit tests |
| Real local CPU DDP/Gloo training | Implemented and executed | `python/trainmesh_runtime/runtime.py`; two local ranks completed three epochs on macOS CPU/Gloo |
| DistributedSampler, model gradient sync, loss aggregation | Implemented and tested | Two-rank Gloo test asserts a shared world size, equal post-training model hashes, sample-weighted loss, and successful checkpoint resume |
| Local checkpoint/resume | Implemented for single-host jobs | Rank-zero model/optimizer and RNG state, SHA-256 integrity check, world-size and training-shape compatibility checks; recovery across worker failure remains unimplemented |
| Local benchmark JSON | Implemented and measured | `benchmark-results/local-two-rank.json`; one local smoke run, not a comparative benchmark |
| Cluster identity and local status | Implemented | `.trainmesh/cluster.json`; no worker registry |
| Enrollment, authenticated agent, heartbeats, revocation | Not implemented | No remote execution surface is exposed |
| Multi-machine DDP | Not implemented | No agent coordination or multi-node launch; physical acceptance pending |
| Adaptive pilot optimizer/network profiler | Not implemented | Planner reports configuration, not a measured optimum |
| GPU/NCCL | Unsupported | No GPU required; CPU/Gloo only |
| Multipass VM test | Blocked by host installer privilege | Homebrew downloaded Multipass 1.17.1, but its package installer requires interactive `sudo`; this session cannot supply a system password. No VM or Multipass test was run. See `docs/MULTI_MACHINE_MANUAL_TEST.md` |

Host observed during implementation: macOS arm64, Go 1.26.3, Python 3.12, Docker 28.5.1, PyTorch 2.15.0 CPU/Gloo. Gloo required explicit `lo0` selection for local execution. The single-host smoke measurement is recorded in JSON; it is not compared with another configuration and is not evidence that more workers improve performance.
