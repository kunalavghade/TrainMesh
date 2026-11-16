#!/usr/bin/env bash
# Phase 2 proof on one Apple Silicon Mac, through the real binaries:
#   1. Two tm-agents join, one offering mps and one offering cpu. Each measures its reference
#      throughput at start; `trainmesh nodes` must show the device, profile and island.
#   2. The same mixed-device job (device: auto, communication: auto) runs twice: balance=uniform
#      and balance=throughput. The coordinator must pick gloo for the group and, for throughput,
#      give the mps rank the larger micro-batch.
#   3. Both runs must reach the accuracy and loss of a one-machine CPU run (compare.py). With
#      balance=throughput the runtime re-splits each epoch from each rank's median step time, and
#      its last epoch must be balanced to within MAX_FINAL_IMBALANCE (slowest rank's step compute
#      over the fastest's). The uniform run is not the bar: on one shared Mac the two ranks can
#      already be about equally fast, which is exactly what the probe alone cannot know.
# Both agents share this Mac, so wall time is a demo, not a benchmark of two real machines.
#
# Usage: scripts/e2e/devices-local.sh          (KEEP_WORK=1 keeps logs and state)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${TRAINMESH_PYTHON:-$ROOT/.venv/bin/python}
BIN=$ROOT/bin
CLI=$BIN/trainmesh
WORK=$(mktemp -d "${TMPDIR:-/tmp}/trainmesh-e2e-devices.XXXXXX")
E2E_TAG=e2e-devices
# Wider than the other e2e runs: at hidden 1024 MPS and one CPU thread are about equally fast on
# the dev Mac, so there would be nothing to balance.
HIDDEN_SIZE=${HIDDEN_SIZE:-4096}
GLOBAL_BATCH=${GLOBAL_BATCH:-1024}
EPOCHS=${EPOCHS:-4}
MAX_FINAL_IMBALANCE=${MAX_FINAL_IMBALANCE:-1.25}
# shellcheck source=scripts/e2e/mesh-lib.sh
source "$ROOT/scripts/e2e/mesh-lib.sh"
trap cleanup EXIT

cd "$ROOT"
if ! "$PYTHON" -c 'import sys, torch; sys.exit(not torch.backends.mps.is_available())'; then
  log "SKIP: needs an Apple Silicon Mac with MPS (one agent offers mps, the other cpu)"
  exit 0
fi
build_binaries
start_meshd

log "joining an mps agent and a cpu agent (each benchmarks its device first)"
join_agent agent-mps --device mps
join_agent agent-cpu --device cpu
wait_for_nodes READY 2 "$READY_TIMEOUT_SECONDS"
"$CLI" nodes >"$WORK/nodes.json"
"$PYTHON" - "$WORK/nodes.json" <<'EOF' || fail "node capabilities are wrong (see nodes.json)"
import json, sys
nodes = json.load(open(sys.argv[1]))
devices = sorted(n["device"] for n in nodes)
assert devices == ["cpu", "mps"], f"devices {devices}"
for n in nodes:
    assert n["island"] == "gloo-host", n
    assert n["samples_per_second"].get(n["device"], 0) > 0, n
    print(f"[e2e-devices]   {n['id']}: {n['device']} {n['samples_per_second'][n['device']]:,.0f} samples/s, island {n['island']}")
EOF
log "ok: devices, profiles and islands reported"

run_job() {
  local name=$1 balance=$2
  "$PYTHON" scripts/e2e/bench_config.py --name "$name" --machines 2 --ranks-per-machine 1 \
    --global-batch "$GLOBAL_BATCH" --epochs "$EPOCHS" --hidden-size "$HIDDEN_SIZE" --device auto --communication auto \
    --balance "$balance" >"$WORK/$name.yaml"
  "$CLI" run --config "$WORK/$name.yaml" --coordinator "$TRAINMESH_COORDINATOR" --output "$WORK/$name.raw.json" \
    >"$WORK/$name-run.out" 2>"$WORK/$name-run.err" || { cat "$WORK/$name-run.err" >&2; fail "$name job failed"; }
  grep "\[job-" "$WORK/$name-run.err" | head -1 | sed "s/^/[$E2E_TAG]   /"
  with_topology "$WORK/$name.raw.json" "$WORK/$name.json" "$name" 2
}

log "mixed-device job, balance=uniform"
run_job mixed-uniform uniform
log "mixed-device job, balance=throughput"
run_job mixed-throughput throughput

log "same job on one machine, one cpu rank"
"$PYTHON" scripts/e2e/bench_config.py --name single-cpu --machines 1 --ranks-per-machine 1 \
  --global-batch "$GLOBAL_BATCH" --epochs "$EPOCHS" --hidden-size "$HIDDEN_SIZE" >"$WORK/single.yaml"
"$CLI" benchmark --config "$WORK/single.yaml" --output "$WORK/single.raw.json" >"$WORK/single.log" 2>&1 ||
  fail "single-machine run failed"
with_topology "$WORK/single.raw.json" "$WORK/single-cpu.json" single-cpu 1

python3 scripts/e2e/compare.py "$WORK/single-cpu.json" "$WORK/mixed-uniform.json" "$WORK/mixed-throughput.json" ||
  fail "mixed-device runs did not train the same model as one machine"
log "ok: cpu+mps trained the same model as one cpu machine"

"$PYTHON" - "$WORK/mixed-uniform.json" "$WORK/mixed-throughput.json" "$MAX_FINAL_IMBALANCE" <<'EOF' || fail "throughput balancing did not behave as expected"
import json, sys
uniform, balanced = (json.load(open(path)) for path in sys.argv[1:3])
max_final_imbalance = float(sys.argv[3])
for name, result in (("uniform", uniform), ("throughput", balanced)):
    ranks = ", ".join(
        f"rank {r['rank']} {r['device']} batch {r['batch_size']} compute {r['compute_seconds']:.2f}s sync {r['sync_seconds']:.2f}s"
        for r in result["ranks"]
    )
    print(f"[e2e-devices]   {name:>10}: {result['collective_backend']}/{result['gradient_sync']}, "
          f"imbalance {result['compute_imbalance']:.2f}, {result['samples_per_second']:,.0f} samples/s, "
          f"drift {result['max_parameter_drift']:.1e} | {ranks}")
for epoch in balanced["balance_history"]:
    print(f"[e2e-devices]   throughput epoch {epoch['epoch']}: split {epoch['rank_batch_sizes']}, "
          f"median step compute {epoch['step_compute_ms']} ms, imbalance {epoch['compute_imbalance']:.2f}")
assert uniform["collective_backend"] == balanced["collective_backend"] == "gloo"
assert uniform["mixed_devices"] and balanced["mixed_devices"]
devices = [r["device"] for r in balanced["ranks"]]
first = dict(zip(devices, balanced["balance_history"][0]["rank_batch_sizes"]))
assert first["mps"] > first["cpu"], f"the coordinator should give mps the larger batch: {first}"
last = balanced["balance_history"][-1]["compute_imbalance"]
assert last <= max_final_imbalance, f"last epoch imbalance {last:.2f} > {max_final_imbalance}"
EOF
log "ok: the coordinator split by profile and the runtime balanced the real job"
log "PASS"
