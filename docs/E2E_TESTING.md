# End-to-end and multi-machine testing

End-to-end tests, from fastest to most realistic. All of them run real PyTorch training; none
use mocks.

| Level | Command | What it proves | Runs in CI |
|-------|---------|----------------|------------|
| 1. Local CLI | `make e2e-local` | The `trainmesh` binary works end to end on one machine | Yes (`e2e` job, Linux) |
| 1b. Local mesh | `make e2e-mesh-local` | `meshd` + 2 `tm-agent`s on one machine: join, train through `trainmesh run --coordinator`, same model as 1 machine, a killed agent fails the job and goes LOST | Yes (`e2e` job, Linux) |
| 1c. Mixed devices | `make e2e-devices-local` | An mps agent and a cpu agent on one Apple Silicon Mac: profiles and islands reported, one job across both devices reaches the same model as 1 CPU machine, `balance: throughput` ends balanced | No (needs MPS; skips elsewhere) |
| 2. minikube multi-node | `make e2e-minikube` | Training across 2 kubernetes nodes reaches the same model as 1 node (torchrun started by Kubernetes); measures overhead | No (needs Docker + minikube) |
| 2b. minikube mesh | `make e2e-minikube-mesh` | `meshd` + one `tm-agent` pod per kubernetes node; the Phase 1 path over a real pod network | No (needs Docker + minikube) |
| 4. WAN (DiLoCo) | `make e2e-wan` (Docker) or `scripts/e2e/wan.sh` (Linux with tc) | 3 islands over tc netem (50 ms RTT, 100 Mbit/s): data_parallel vs diloco at equal samples (accuracy, wall time, bytes, utilization), then the root island is killed and the job still succeeds | Yes (`e2e` job, `sudo tc` on the runner) |
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

## Level 1c — mixed devices on one Mac (Phase 2)

```bash
make e2e-devices-local  # = scripts/e2e/devices-local.sh; KEEP_WORK=1 keeps logs and state
```

Needs an Apple Silicon Mac (prints SKIP elsewhere). Starts `meshd` and two agents,
`tm-agent --device mps` and `tm-agent --device cpu`; each benchmarks its device at start. Checks
`trainmesh nodes` shows one `mps` and one `cpu` node, both in island `gloo-host`, each with a
measured throughput. Then runs one job across both (`device: auto`, `communication: auto`,
MLP 256-4096-10, global batch 1024, 4 epochs) twice, with `balance: uniform` and
`balance: throughput`, plus the same job on one CPU rank. Requires: the coordinator picked
`gloo`; mixed devices were detected; the coordinator's first split gave the mps rank the larger
batch (from the profiles); both mesh runs match the one-machine loss and accuracy
(`compare.py`); and the throughput run's last epoch is balanced to within 1.25×
(`MAX_FINAL_IMBALANCE`).

Measured on the development Mac (2026-11-17), two consecutive runs:

| | Run 1 | Run 2 |
|---|---|---|
| Probe (one process, staged step) | mps ≈ 150–185k, cpu ≈ 100k samples/s | same |
| 1 CPU rank: loss / accuracy | 0.060631 / 93.87% | 0.060631 / 93.87% |
| cpu+mps, uniform: loss / accuracy | 0.060632 / 93.82% | 0.060632 / 93.82% |
| cpu+mps, throughput: loss / accuracy | 0.060629 / 93.82% | 0.060631 / 93.85% |
| Throughput split per epoch (cpu, mps) | 457/567 → 506/518 → 506/518 → 506/518 | 453/571 → 496/528 → 496/528 → 496/528 |
| Step-time imbalance per epoch | 1.21 → 1.01 → 1.04 → 1.02 | 1.18 → 1.05 → 1.06 → 1.08 |
| Parameter drift before realignment | 3e-8 | 3e-8 |

What this shows: the probe ranked MPS ~1.7× faster, so the coordinator gave it more; on the real
job the two ranks were about equally fast (both agents share this Mac's CPU, and per-step host
staging costs MPS time), and after one measured epoch the runtime moved the split to match and
then held it. The mesh is slower than one machine here (≈ 55k vs 104k samples/s): of each
~4.7 s run, every rank spends ~3 s in the per-step all-reduce of 4.4 MB of gradients over
loopback, and both "machines" share one CPU. This level proves correctness and the balancing mechanism, not a speed-up; real gains need
machines with different hardware (pending, see "Not covered yet").

The first version of this test used the per-sample `DataLoader` and the probe's pipelined step.
It found that the loader took 3–4 ms per 1024-sample batch (more than the model step, hiding any
device difference) and that per-epoch totals were too noisy to re-split on. Both were fixed (see
[modules/devices.md](modules/devices.md)).

## Level 4 — DiLoCo over an emulated WAN (Phase 3)

```bash
make e2e-wan    # = scripts/e2e/wan-docker.sh: builds the image, runs wan.sh inside it
```

macOS has no `tc netem`, so `wan-docker.sh` runs everything in one Linux container
(`deploy/docker/wan.Dockerfile`: runtime image + tmcomm wheel built in a Rust stage + iproute2 +
cross-compiled Go binaries) started with `--cap-add NET_ADMIN`. It first checks the Docker VM has
room (the VM is shared with other containers) and caps the container at 1800 MiB, so an overrun
is killed inside the test container only. On Linux (and in CI) `wan.sh` runs directly with
`TC="sudo tc"`.

`wan.sh`: shapes loopback with `tc qdisc ... netem delay 25ms rate 100mbit` and measures the
result; starts `meshd` and 3 cpu agents (one island each) over the shaped link; trains the same
model (MLP 64-1024-10, 76,810 parameters) on the same 98,304 teacher samples × 3 epochs twice:
`data_parallel` (gradient all-reduce every step, 3 × 512 per step; `gradient_sync: allreduce` so
its waiting is measured rather than hidden inside DDP's backward) and `diloco` (H = 64, int8).
Then a 6-epoch diloco job loses island 0 — the tmcomm root — after its first round (agent and
training process `kill -9`). `wan_report.py` requires: both ≥ 85% accuracy; diloco within 2
points of data_parallel; identical model hashes on all islands; diloco faster; diloco compute
utilization ≥ 60%; the killed job SUCCEEDED with islands `[FAILED, SUCCEEDED, SUCCEEDED]` and
the last round's members `[1, 2]`.

Measured on the development Mac's Docker VM (2026-11-17, linux/arm64, 1 CPU thread per rank),
`benchmark-results/e2e/wan-20261117-113656/`:

| | data_parallel (sync every step) | diloco (H = 64, int8) |
|---|---|---|
| Link | measured RTT 52 ms, 72 Mbit/s | same |
| Wall time | 49.2 s | **1.3 s (37×)** |
| Bytes between machines | 236 MB (ring all-reduce, fp32) | **0.9 MB (255× less)** |
| Compute utilization | 3% | **69%** |
| Validation accuracy / final loss | 95.21% / 0.2116 | 96.19% / 0.1418 |

Root island killed mid-run: job SUCCEEDED, islands `[FAILED, SUCCEEDED, SUCCEEDED]`, islands 1
and 2 elected island 1 as root and finished with members `[1, 2]` at 97.27% accuracy.

How to read this: the model is tiny, so per-step all-reduce over a 50 ms link is almost all
waiting; DiLoCo's advantage grows with RTT and shrinks with compute per step. The accuracy here
is from 3 rounds on a simple task, so "DiLoCo matches data_parallel" holds for this workload, not
in general — the DiLoCo paper reports a small loss gap at large scale. Loopback shaping puts all
three islands behind one shared 100 Mbit/s queue, which is stricter than three separate links.

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

Re-run after Phase 2 (2026-11-17, agents now benchmark their CPU at start): same checks pass;
mesh and one machine both reach loss 0.101840 and accuracy 91.55% (float summation order changed
with the new sampler, so the value differs slightly from the Phase 1 run; the two topologies
still agree exactly). The memory preflight had been comparing a value with a trailing `\r` from
`minikube ssh`, so it errored and never refused a run; it now strips it and refuses correctly.

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
- CUDA, NCCL and ROCm: written and unit-tested, never run (no NVIDIA/AMD GPU available).
- Balancing between genuinely different machines (e.g. a GPU PC and a CPU box): level 1c runs
  both agents on one Mac.
- Real WAN and NAT: level 4 emulates latency and bandwidth on one host; machines behind NAT
  cannot reach each other's tmcomm port yet (no hole punching or relay).
