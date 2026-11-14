# Roadmap

1. **Local CPU correctness:** validate actual single and two-rank Gloo training, checkpoint integrity, deterministic resume, and numerical parity under fixed effective batch semantics.
2. **Secure cluster control plane:** TLS/mTLS, expiring single-use tokens, persistent worker identities, heartbeat/lifecycle state, revocation, and capability/environment matching.
3. **Multi-node CPU:** safe per-agent process launch, artifact transfer/checksum, Gloo network validation, peer connectivity, two-host integration tests, and group termination.
4. **Measurement and optimization:** per-rank step/communication metrics, real baseline comparisons, candidate feasibility, pilot runs, and evidence-based recommendations.
5. **Recovery and reliability:** checkpoint hashes, corruption tests, disk checks, failure detection, whole-group restart with a new rendezvous, and dataset/config compatibility checks.
6. **GPU support:** CUDA/NCCL discovery and tested GPU execution only after hardware is available.

FSDP, tensor parallelism, pipeline parallelism, and heterogeneous GPU scheduling remain future work; there are no mock implementations.
