# End-to-end and multi-machine testing

Three levels of end-to-end test, from fastest to most realistic. All of them run real PyTorch
training; none use mocks.

| Level | Command | What it proves | Runs in CI |
|-------|---------|----------------|------------|
| 1. Local CLI | `make e2e-local` | The `trainmesh` binary works end to end on one machine | Yes (`e2e` job, Linux) |
| 2. minikube multi-node | `make e2e-minikube` | Training across 2 kubernetes nodes reaches the same model as 1 node; measures overhead | No (needs Docker + minikube) |
| 3. Physical machines | `scripts/e2e/manual-node.sh` on each machine | Real 1-machine vs multi-machine speed over your network | No (manual) |

> The trainmesh CLI cannot launch multi-machine jobs yet (that is Phase 1 in
> [TRAINMESH_ARCHITECTURE.md](TRAINMESH_ARCHITECTURE.md)). Levels 2 and 3 therefore call
> `torchrun` directly with the TrainMesh runtime. They test the runtime and the network path,
> not a TrainMesh agent.

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

Default workload: MLP 256 → 1024 → 10, 65 536 synthetic samples, 2 epochs, Adam, fp32,
1 rank per CPU, `OMP_NUM_THREADS=1`.

`compare.py` reports, relative to the first (baseline) scenario:

- **Speedup** = samples/s ÷ baseline samples/s.
- **Per-machine efficiency** = speedup ÷ (machines ÷ baseline machines). 100% means adding a
  machine gave a proportional gain; ≤ 50% for 2 machines means no gain at all.

## Level 1 — local CLI test

```bash
make e2e-local          # KEEP_WORK=1 scripts/e2e/local.sh keeps the work directory
```

Checks, in order: CLI builds; `doctor` reports Gloo; `plan` resolves world size and global batch;
`run` trains 2 epochs with consistent replicas and writes a v2 safetensors checkpoint (no `.pt`);
`run --resume` continues to epoch 3 with the right global step; `benchmark` writes throughput;
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
`MEMORY_PER_RANK_MI` (512), `TORCH_VERSION` (2.15.0), `JOB_TIMEOUT_SECONDS` (900).

**Limit:** both nodes are containers on one Docker host. The pod network is virtual and fast, and
the nodes share one CPU package and memory bus. This level proves multi-node *correctness* and
shows communication overhead; it does **not** predict the speed of real separate machines.

### Measured on the development Mac

First run (2026-11-16, `benchmark-results/e2e/minikube-20261116-214339/`, not committed):

- `single` (1 node, 2 CPUs, 2 ranks): **passed**, 29,385 samples/s, 4.46 s, final loss 2.274241,
  replicas consistent. A native macOS run of the same config gives 2.274259; the 1.8e-5 gap comes
  from Linux vs macOS math libraries and is within the 1e-3 tolerance.
- `single-2x` (1 node, 4 CPUs, 4 ranks): **OOMKilled** on node `trainmesh-m02`, so `multi` did
  not run. Peak memory of the same job measured with `docker run` and no limit: **888 MiB for 2
  ranks, 1,440 MiB for 4 ranks**, below the pod's 3 GiB limit. The likely cause is the 3 GB
  minikube node itself running short, not the pod limit.

Open: size `NODE_MEMORY` / `MEMORY_PER_RANK_MI` from those measurements (for example 4 ranks ≈
1.5 GiB request) and re-run before trusting any multi-node numbers from this level.

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

## Not covered yet

- Launching multi-machine jobs through the `trainmesh` CLI and agent (Phase 1).
- Resume across machines (checkpoints are written only on node 0's local disk).
- Multipass VMs: Multipass is not installed on the development Mac (its installer needs `sudo`).
  The manual script works inside Multipass VMs as it does on physical machines.
- GPU backends and WAN conditions (`tc netem` latency/bandwidth emulation is planned).
