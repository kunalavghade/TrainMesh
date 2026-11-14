# Manual multi-machine test (Multipass and physical Macs)

## Current support boundary

The current code does not implement remote worker enrollment, agent authentication, or multi-node process launch. Consequently, a successful local `torchrun` run **does not validate a multi-machine cluster**. `trainmesh run` rejects configs with `execution.workers > 1`. This document gives a manual protocol-level test for a future multi-node runtime and the Multipass smoke checks that can be performed now; it does not claim the missing functionality passed.

## Multipass availability on the development Mac

Check first:

```bash
command -v multipass
multipass version
```

On the development Mac used for this implementation, Multipass was absent. `brew install --cask multipass` downloaded the package, then stopped because the macOS package installer requires an interactive administrator password (`sudo: a terminal is required to read the password`). No VM test was run. After an administrator installs Multipass, create two isolated Ubuntu VMs (Multipass is useful for Linux networking checks, but does not emulate two physical Macs):

```bash
multipass launch 24.04 --name tm-controller --cpus 2 --memory 4G --disk 12G
multipass launch 24.04 --name tm-worker --cpus 2 --memory 4G --disk 12G
multipass list
multipass exec tm-controller -- bash -lc 'uname -a; ip -brief address'
multipass exec tm-worker -- bash -lc 'uname -a; ip -brief address'
```

Install the same Go/Python/PyTorch versions and a matching TrainMesh checkout in both VMs. Confirm mutual TCP reachability before any future cluster test. Do not expose rendezvous or agent ports to untrusted networks.

## Future end-to-end acceptance checklist

1. On the primary machine, initialize a cluster, start its controller and local worker, and create a short-lived single-use join token.
2. On the second machine, start the TrainMesh agent with the controller address and token. Verify distinct persistent worker IDs, READY states, matching Python/PyTorch/runtime versions, and reported CPU/memory/network capabilities.
3. Run the same commit and config on both hosts. Use two nodes, one process per node, Gloo, one stable LAN/VPN interface, and a routable rendezvous address. Open the rendezvous and peer-to-peer TCP ports between hosts.
4. Launch exactly one `torchrun --nnodes=2 --nproc-per-node=1` process per node with common `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE=2`, node ranks 0/1, and matching run/rendezvous IDs. Gloo traffic must be direct between Python workers.
5. Verify logs report ranks 0 and 1, world size 2, distinct sampler indices, matching final model checksums, sample-weighted loss, and one valid checkpoint written by rank zero.
6. Repeat a fixed-seed single-rank baseline with the same global batch and compare elapsed time, throughput, and loss. Preserve raw JSON; distributed execution may be slower.
7. Kill one rank during a later run; verify the entire group stops, the latest checkpoint checksum validates, and a restart uses a new process group and an explicitly compatible world size.

## Local DDP smoke test (not a multi-host substitute)

On a machine with CPU PyTorch/Gloo:

```bash
go build -o bin/trainmesh ./cmd/trainmesh
./bin/trainmesh doctor
./bin/trainmesh run --config examples/cpu-ddp.yaml
```

Expected output contains epoch events with `world_size: 2`; `checkpoints/latest/metadata.json` records the world size and step. Keep local results labeled `single_host` in any report.
