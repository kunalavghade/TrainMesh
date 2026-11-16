# End-to-end and multi-machine testing

End-to-end tests, from fastest to most realistic. All of them run real PyTorch training; none
use mocks.

| Level | Command | What it proves | Runs in CI |
|-------|---------|----------------|------------|
| 1. Local CLI | `make e2e-local` | The `trainmesh` binary works end to end on one machine | Yes (`e2e` job, Linux) |
| 1b. Local mesh | `make e2e-mesh-local` | `meshd` + 2 `tm-agent`s on one machine: join, train through `trainmesh run --coordinator`, same model as 1 machine, a killed agent fails the job and goes LOST | Yes (`e2e` job, Linux) |
| 2. minikube multi-node | `make e2e-minikube` | Training across 2 kubernetes nodes reaches the same model as 1 node (torchrun started by Kubernetes); measures overhead | No (needs Docker + minikube) |
| 2b. minikube mesh | `make e2e-minikube-mesh` | `meshd` + one `tm-agent` pod per kubernetes node; the Phase 1 path over a real pod network | No (needs Docker + minikube) |
| 3. Physical machines | `scripts/e2e/manual-node.sh`, or `meshd` + `tm-agent` as in [modules/mesh.md](modules/mesh.md) | Real 1-machine vs multi-machine speed over your network | No (manual) |

Levels 2 and 3 with `manual-node.sh` start `torchrun` directly, testing the runtime and the
network path. Levels 1b and 2b go through the Phase 1 control plane (`meshd`, agents, CLI).

## How the 1-machine vs multi-machine comparison stays fair

All scenarios use [`scripts/e2e/bench_config.py`](../scripts/e2e/bench_config.py), which keeps
the **global batch fixed** (default 256) and splits it across however many ranks exist:
`per_rank = global_batch / (machines × ranks_per_machine)`.

Because `DistributedSampler` hands rank *r* the indices `r, r+W, r+2W, …` of one shuffled list,
the union of all ranks' *k*-th batches is always the same block of the list. Every optimizer step
therefore sees the **same samples in every topology**, so every topology must reach the **same
final loss** (up to floating-point summation order). [`compare.py`](../scripts/e2e/compare.py)
fails if final losses differ by more than `1e-3` or if any run reports diverged replicas.
That turns the benchmark into a correctness test as well.

Default workload: the learnable `teacher` task, MLP 256 → 1024 → 10, 65,536 training samples,
4,096 held-out validation samples, 3 epochs, Adam, fp32, 1 rank per CPU, `OMP_NUM_THREADS=1`.

### Proof that the mesh trains a real model

Throughput alone does not show that distributed training *works*. Every e2e level therefore
trains the `teacher` task (labels from a hidden linear "teacher" network, see
[modules/runtime.md](modules/runtime.md)) and checks validation accuracy on held-out data:

- **It learned:** final validation accuracy must be ≥ 85% (`compare.py --min-accuracy`, chance is
  10% for 10 classes; measured ≈ 92%).
- **Every topology learned the same model:** validation accuracy may differ by at most 0.5
  points (`--accuracy-tolerance`) and final loss by `1e-3` from the 1-machine baseline, and every
  run must report identical parameters on all ranks.

`compare.py` reports, relative to the first (baseline) scenario:

- **Speedup** = samples/s ÷ baseline samples/s.
- **Per-machine efficiency** = speedup ÷ (machines ÷ baseline machines). 100% means adding a
  machine gave a proportional gain; ≤ 50% for 2 machines means no gain at all.

## Level 1 — local CLI test

```bash
make e2e-local          # KEEP_WORK=1 scripts/e2e/local.sh keeps the work directory
```

Checks, in order: CLI builds; `doctor` reports Gloo; `plan` resolves world size and global batch;
`run` trains 2 epochs on the `teacher` task with consistent replicas and writes a v2 safetensors
checkpoint (no `.pt`); `run --resume` continues to epoch 3 with the right global step and
**validation accuracy ≥ 85%** (measured 93.2%; chance 25%); `benchmark` writes throughput;
a v1 pickle checkpoint is rejected with `CHECKPOINT_INVALID`; `workers: 2` is rejected.

## Level 2 — minikube multi-node test

Requirements: Docker running with at least 10 CPUs and 7 GB RAM available to it, `minikube`,
`kubectl`. The first run downloads the CPU PyTorch wheel into the image (a few hundred MB).

```bash
make e2e-minikube       # = scripts/e2e/minikube.sh all  (up, build, bench)
make e2e-minikube-down  # deletes the 'trainmesh' minikube profile
```

Steps: `up` starts a 2-node cluster (profile `trainmesh`, docker driver); `build` builds
[`deploy/docker/runtime.Dockerfile`](../deploy/docker/runtime.Dockerfile) and loads it into both
nodes; `bench` runs three scenarios as Kubernetes Indexed Jobs with a headless Service for
rendezvous and pod anti-affinity so each "machine" is a different node:

| Scenario | Machines | CPUs each | Question it answers |
|----------|----------|-----------|---------------------|
| `single` | 1 | `POD_CPUS` (2) | baseline |
| `single-2x` | 1 | 2 × `POD_CPUS` | what if the one machine had twice the cores? |
| `multi` | 2 | `POD_CPUS` | what if a second machine with the same cores joins? |

`multi` vs `single-2x` isolates the cost of going over the network with the same total cores.
The script fails if any scenario's pods did not land on the expected number of distinct nodes.

Results land in `benchmark-results/e2e/minikube-<timestamp>/` (git-ignored): per-scenario config,
pod logs, result JSON (with a `topology` block) and `summary.md`.

Tunables (environment variables): `PROFILE`, `NODE_CPUS` (5), `NODE_MEMORY` (3g), `POD_CPUS` (2),
`MEMORY_BASE_MI` (340), `MEMORY_PER_RANK_MI` (280), `MEMORY_HEADROOM_MI` (256),
`TORCH_VERSION` (2.15.0), `JOB_TIMEOUT_SECONDS` (900). Before each scenario the script checks the
Docker VM's free memory (all nodes share it) and stops with a clear message if it is too low.

**Limit:** both nodes are containers on one Docker host. The pod network is virtual and fast, and
the nodes share one CPU package and memory bus. This level proves multi-node *correctness* and
shows communication overhead; it does **not** predict the speed of real separate machines.

### Measured on the development Mac (2026-11-16)

Run `benchmark-results/e2e/minikube-20261116-222818/` (git-ignored). Workload: `teacher` task,
MLP 256 → 1024 → 10, 65,536 samples, global batch 256, 3 epochs, 4,096 validation samples.

| Scenario | Nodes used | Ranks | CPUs | Samples/s | Final loss | Val accuracy per epoch |
|----------|-----------|-------|------|-----------|------------|------------------------|
| `single` | m02 | 2 | 2 | 51,535 | 0.101872 | 89.87% → 90.92% → **91.67%** |
| `single-2x` | m02 | 4 | 4 | 51,258 | 0.101847 | 89.89% → 90.97% → **91.70%** |
| `multi` | m02 + control plane | 4 | 4 | 45,620 | 0.101847 | 89.89% → 90.97% → **91.70%** |

What this proves and what it does not:

- **The mesh trains correctly across machines.** The 2-node run learned the task (chance is 10%)
  and matched the 1-node run: identical accuracy history and identical final loss to the
  4-rank single-node run. The 2-rank run differs only in the 5th significant digit (float
  summation order over 2 vs 4 ranks), far inside the tolerances.
- **More hardware did not make this small model faster.** Doubling ranks on one node gave no
  speedup, and the same 4 ranks split over two nodes were **11% slower** than on one node
  (per-machine efficiency 44%). Per step, each rank computes a small batch but still exchanges
  every gradient (~1 MB), so communication dominates. This is the motivation for P3's
  low-communication training (DiLoCo) and for larger per-step work. It is also why the comparison
  must be repeated on real machines and real models before drawing speed conclusions.

History: the first run (2026-11-16, 21:43) had `single-2x` OOM-killed. The kernel log showed a
`global_oom` of the whole Docker VM (7.8 GB shared with unrelated containers), not a pod or node
limit. The script now sizes pods from measured memory (≈ 340 MiB + 280 MiB per rank) and refuses
to start a scenario without enough free VM memory.

### Also proven without Kubernetes

Two node processes on one Mac with `manual-node.sh` (`--machines 2`, 2 ranks each) against one
node with 4 ranks: identical loss and validation accuracy every epoch
(89.89% → 90.97% → 91.70%, final loss 0.101882).

## Level 3 — physical machines (manual)

Use this for the real 1-machine vs multi-machine answer. Every machine needs the same commit,
`.venv` with the same torch version (`uv pip install --python .venv/bin/python -e 'python[dev]'`),
and TCP reachability between machines on the chosen port (default 29500) plus Gloo's ephemeral
ports. Use a trusted network: rendezvous traffic is not authenticated yet.

1. Find each machine's LAN interface and address (`ipconfig getifaddr en0` on macOS,
   `ip -brief address` on Linux). Pick machine A as node 0.
2. **Baseline on A alone:**
   ```bash
   scripts/e2e/manual-node.sh --scenario single --machines 1 --node-rank 0 \
     --master-addr 127.0.0.1 --iface lo0 --ranks-per-machine 4
   ```
   (`--iface lo` on Linux.)
3. **Two machines**, same `--ranks-per-machine`; start both within the rendezvous timeout:
   ```bash
   # on A (192.168.1.10)
   scripts/e2e/manual-node.sh --scenario multi --machines 2 --node-rank 0 \
     --master-addr 192.168.1.10 --iface en0 --ranks-per-machine 4
   # on B
   scripts/e2e/manual-node.sh --scenario multi --machines 2 --node-rank 1 \
     --master-addr 192.168.1.10 --iface en0 --ranks-per-machine 4
   ```
4. **Compare on A:**
   ```bash
   python3 scripts/e2e/compare.py benchmark-results/e2e/manual/single.json \
     benchmark-results/e2e/manual/multi.json
   ```

Use the same `--ranks-per-machine` on both machines, close other heavy apps, and repeat each
scenario 3 times; report the median. On Wi-Fi expect the network to dominate; wired gigabit or
faster gives a fairer picture.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Hangs at `Rendezvous'ing worker group` | Slow reverse DNS of the host name (seen on GitHub macOS runners: 35 s per lookup) | Time it: `python3 -c "import socket,time;t=time.time();socket.getfqdn();print(time.time()-t)"`. Fix DNS or add the host to `/etc/hosts`. |
| `Connection refused` / timeout to master | Firewall or wrong `--master-addr` | Allow the port on node 0; use node 0's LAN IP on every machine |
| Gloo picks the wrong network | Several interfaces (VPN, Docker) | Pass the LAN interface with `--iface` |
| `compare.py` reports different final losses | Different `--ranks-per-machine`/commit/torch version, or a real bug | Make inputs identical; if it persists, report it with both JSON files |

## Level 1b — local mesh (Phase 1)

```bash
make e2e-mesh-local     # = scripts/e2e/mesh-local.sh; KEEP_WORK=1 keeps logs and state
```

Builds `trainmesh`, `meshd` and `tm-agent`; starts `meshd` on a free loopback port; joins two
agents (separate state dirs = separate identities) with single-use tokens; checks that a used
join string is refused; trains the `teacher` task across both agents with
`trainmesh run --coordinator`; runs the same job on one machine; requires identical accuracy and
loss (`compare.py`); then starts a long job, `kill -9`s one agent and its training processes, and
requires the job to FAIL, that node to become LOST and the other to return to READY.

Measured on the development Mac (2026-11-16):

| Check | Result |
|-------|--------|
| Two agents join, become READY | yes |
| Reused join string | refused ("already been used") |
| 2 machines × 2 ranks via the CLI vs 1 machine × 4 ranks | identical final loss 0.101882 and validation accuracy **91.70%** |
| `kill -9` an agent mid-job | job FAILED ("node lost (no heartbeats)"), node LOST, survivor READY |

## Level 2b — minikube mesh (Phase 1)

```bash
make e2e-minikube-mesh  # = scripts/e2e/minikube.sh mesh (needs the cluster from `make e2e-minikube`)
```

Cross-compiles the Go binaries for the Docker VM's architecture, builds
[`deploy/docker/mesh.Dockerfile`](../deploy/docker/mesh.Dockerfile) on top of the runtime image,
and deploys `meshd` (Deployment + Service) and one `tm-agent` per node (StatefulSet with pod
anti-affinity; join string from a 2-use token in a Secret). Then: a 2-machine job through
`trainmesh run --coordinator` inside the `meshd` pod; the same job on one machine inside
`trainmesh-agent-0`; `compare.py`; and finally a forced delete of the node-rank-1 agent pod
mid-job. `make e2e-minikube-mesh` cleans up the mesh resources at the end
(`scripts/e2e/minikube.sh mesh-down` does it by hand).

Measured 2026-11-16 (`benchmark-results/e2e/minikube-20261116-233034/`):

| Check | Result |
|-------|--------|
| Agents | on 2 distinct kubernetes nodes (`trainmesh-m02`, `trainmesh`), both READY |
| Mesh job (2 machines × 1 rank) | SUCCEEDED; rendezvous at node rank 0's **pod IP** (10.244.1.8:29500), so traffic crossed nodes |
| Mesh vs 1 machine × 2 ranks | identical final loss 0.101872 and validation accuracy **91.67%**; 51,308 vs 53,465 samples/s |
| Agent pod force-deleted mid-job | job FAILED, deleted node LOST |

The forced delete still delivered SIGTERM, so the dying agent reported "agent stopped" itself;
the silent-death path (no report, LOST via missing heartbeats) is covered by level 1b's `kill -9`.

## Phase 1 acceptance on physical machines (manual, pending)

Follow "Operating it" in [modules/mesh.md](modules/mesh.md) on two real machines: both join with
single-use tokens and report READY; one job runs one rank per machine; final model hashes match;
validation accuracy matches the single-machine baseline; killing one machine fails the job and
marks it LOST. Not run yet: no second physical machine was available.

## Not covered yet

- Two physical machines (see above).
- Automatic recovery after a node is lost (the job fails; re-submit with `run.resume`; Phase 5).
- Resume across machines (checkpoints are written only on node 0's local disk).
- Multipass VMs: Multipass is not installed on the development Mac (its installer needs `sudo`).
  The manual script works inside Multipass VMs as it does on physical machines.
- GPU backends and WAN conditions (`tc netem` latency/bandwidth emulation is planned).
