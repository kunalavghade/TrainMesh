#!/usr/bin/env bash
# Run one node of the 1-machine vs multi-machine benchmark on a real machine.
#
# The trainmesh CLI cannot launch multi-machine jobs yet (Phase 1), so this calls torchrun
# directly with the same config generator as the minikube test. Run it on every machine with the
# same --scenario/--machines/--ranks-per-machine; only --node-rank differs. Node 0 writes
# benchmark-results/e2e/manual/<scenario>.json, then compare with scripts/e2e/compare.py.
#
# Examples (machine A = 192.168.1.10 on en0, machine B on en0):
#   A: scripts/e2e/manual-node.sh --scenario single --machines 1 --node-rank 0 --master-addr 127.0.0.1 --iface lo0
#   A: scripts/e2e/manual-node.sh --scenario multi  --machines 2 --node-rank 0 --master-addr 192.168.1.10 --iface en0
#   B: scripts/e2e/manual-node.sh --scenario multi  --machines 2 --node-rank 1 --master-addr 192.168.1.10 --iface en0
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${TRAINMESH_PYTHON:-$ROOT/.venv/bin/python}
RESULTS=${RESULTS:-$ROOT/benchmark-results/e2e/manual}
PORT=29500
RANKS_PER_MACHINE=2
SCENARIO="" MACHINES="" NODE_RANK="" MASTER_ADDR="" IFACE=""

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  echo "Options: --scenario --machines --node-rank --master-addr --iface [--ranks-per-machine N] [--port P]"
  exit 2
}
fail() {
  printf '[e2e-manual] FAIL: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) SCENARIO=$2 ;;
    --machines) MACHINES=$2 ;;
    --node-rank) NODE_RANK=$2 ;;
    --master-addr) MASTER_ADDR=$2 ;;
    --iface) IFACE=$2 ;;
    --ranks-per-machine) RANKS_PER_MACHINE=$2 ;;
    --port) PORT=$2 ;;
    -h | --help) usage ;;
    *) fail "unknown option $1" ;;
  esac
  shift 2
done
[[ -n "$SCENARIO" && -n "$MACHINES" && -n "$NODE_RANK" && -n "$MASTER_ADDR" && -n "$IFACE" ]] || usage
[[ -x "$PYTHON" ]] || fail "python not found at $PYTHON (set TRAINMESH_PYTHON)"

mkdir -p "$RESULTS"
CONFIG="$RESULTS/$SCENARIO.yaml"
RAW_RESULT="$RESULTS/$SCENARIO.raw.json"
"$PYTHON" "$ROOT/scripts/e2e/bench_config.py" --name "$SCENARIO" --machines "$MACHINES" \
  --ranks-per-machine "$RANKS_PER_MACHINE" --results-path "$RAW_RESULT" >"$CONFIG"

echo "[e2e-manual] node $NODE_RANK/$MACHINES, $RANKS_PER_MACHINE rank(s), master $MASTER_ADDR:$PORT via $IFACE"
PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 USE_LIBUV=0 GLOO_SOCKET_IFNAME="$IFACE" \
  "$PYTHON" -m torch.distributed.run --nnodes="$MACHINES" --node_rank="$NODE_RANK" \
  --nproc_per_node="$RANKS_PER_MACHINE" --master_addr="$MASTER_ADDR" --master_port="$PORT" \
  -m trainmesh_runtime --config "$CONFIG"

[[ "$NODE_RANK" == "0" ]] || {
  echo "[e2e-manual] done (results are written on node 0)"
  exit 0
}
[[ -f "$RAW_RESULT" ]] || fail "no result written to $RAW_RESULT"
"$PYTHON" - "$RAW_RESULT" "$RESULTS/$SCENARIO.json" "$SCENARIO" "$MACHINES" "$RANKS_PER_MACHINE" <<'EOF'
import json, os, sys
raw, out, scenario, machines, ranks = sys.argv[1:]
result = json.load(open(raw))
result["topology"] = {
    "scenario": scenario,
    "machines": int(machines),
    "ranks_per_machine": int(ranks),
    "cpus_per_machine": os.cpu_count(),
    "cpus_total": int(machines) * (os.cpu_count() or 0),
    "environment": "physical machines (manual)",
}
json.dump(result, open(out, "w"), indent=2)
EOF
echo "[e2e-manual] result: $RESULTS/$SCENARIO.json"
echo "[e2e-manual] compare: python3 scripts/e2e/compare.py $RESULTS/single.json $RESULTS/multi.json"
