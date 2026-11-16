#!/usr/bin/env bash
# Phase 1 proof on one machine, through the real binaries:
#   1. meshd starts; two tm-agents join with single-use tokens and become READY.
#   2. `trainmesh run --coordinator` trains the teacher task across both agents.
#   3. The same job runs on one machine; compare.py requires the same accuracy and loss.
#   4. A job is started and one agent (with its training processes) is killed mid-run: the job
#      must FAIL, the dead node must become LOST and the survivor must return to READY.
# Both agents share this Mac, so this proves the control plane and correctness, not speed.
#
# Usage: scripts/e2e/mesh-local.sh            (KEEP_WORK=1 keeps logs and state)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${TRAINMESH_PYTHON:-$ROOT/.venv/bin/python}
BIN=$ROOT/bin
CLI=$BIN/trainmesh
WORK=$(mktemp -d "${TMPDIR:-/tmp}/trainmesh-e2e-mesh.XXXXXX")
RANKS_PER_MACHINE=${RANKS_PER_MACHINE:-2}
READY_TIMEOUT_SECONDS=60
LOST_TIMEOUT_SECONDS=30
LONG_JOB_EPOCHS=200
declare -a PIDS=()
declare -a AGENT_PIDS=()

log() { printf '[e2e-mesh] %s\n' "$*"; }
fail() {
  printf '[e2e-mesh] FAIL: %s (logs in %s)\n' "$*" >&2
  KEEP_WORK=1
  exit 1
}
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  pkill -f "$WORK/" 2>/dev/null || true
  if [[ "${KEEP_WORK:-0}" == "1" ]]; then
    echo "[e2e-mesh] work directory kept: $WORK"
  else
    rm -rf -- "$WORK"
  fi
}
trap cleanup EXIT

free_port() { "$PYTHON" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

# node_count STATE prints how many nodes are in STATE.
node_count() {
  "$CLI" nodes 2>/dev/null | "$PYTHON" -c 'import json, sys; print(sum(n["state"] == sys.argv[1] for n in json.load(sys.stdin)))' "$1"
}

wait_for_nodes() {
  local state=$1 count=$2 timeout=$3 waited=0
  until [[ "$(node_count "$state" || echo 0)" -ge "$count" ]]; do
    ((waited < timeout)) || fail "expected $count node(s) $state within ${timeout}s"
    sleep 1
    waited=$((waited + 1))
  done
  log "ok: $count node(s) $state"
}

# with_topology RAW OUT SCENARIO MACHINES adds the topology block compare.py reads.
with_topology() {
  "$PYTHON" - "$@" <<'EOF'
import json, sys
raw, out, scenario, machines = sys.argv[1:]
result = json.load(open(raw))
result["topology"] = {"scenario": scenario, "machines": int(machines),
                      "environment": "one Mac: meshd + tm-agents over loopback"}
json.dump(result, open(out, "w"), indent=2)
EOF
}

cd "$ROOT"
log "building binaries"
go build -o "$BIN/trainmesh" ./cmd/trainmesh
go build -o "$BIN/meshd" ./cmd/meshd
go build -o "$BIN/tm-agent" ./cmd/tm-agent

PORT=$(free_port)
RENDEZVOUS_PORT=$(free_port)
export TRAINMESH_COORDINATOR="127.0.0.1:$PORT" TRAINMESH_ADMIN_DIR="$WORK/meshd"

log "starting meshd on 127.0.0.1:$PORT"
"$BIN/meshd" --state-dir "$WORK/meshd" --listen "127.0.0.1:$PORT" --advertise "127.0.0.1:$PORT" \
  --rendezvous-port "$RENDEZVOUS_PORT" --heartbeat-interval 1s --suspect-after 3s --lost-after 6s \
  >"$WORK/meshd.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 20); do "$CLI" nodes >/dev/null 2>&1 && break || sleep 0.5; done
"$CLI" nodes >/dev/null 2>&1 || fail "meshd did not come up (see meshd.log)"

log "joining two agents"
for i in 1 2; do
  join=$("$CLI" token create --ttl 5m 2>/dev/null) || fail "token create failed"
  "$BIN/tm-agent" --state-dir "$WORK/agent$i" --join "$join" >"$WORK/agent$i.log" 2>&1 &
  PIDS+=($!)
  AGENT_PIDS+=($!)
  LAST_JOIN=$join
done
wait_for_nodes READY 2 "$READY_TIMEOUT_SECONDS"
"$CLI" nodes >"$WORK/nodes.json"

log "rejects reusing a single-use token"
if "$BIN/tm-agent" --state-dir "$WORK/agent-reuse" --join "$LAST_JOIN" >"$WORK/agent-reuse.log" 2>&1; then
  fail "a used token was accepted"
fi
grep -q "already been used" "$WORK/agent-reuse.log" || fail "reuse was not rejected as 'already been used'"
log "ok: token reuse rejected"

log "training across 2 machines through the CLI"
python3 scripts/e2e/bench_config.py --name mesh --machines 2 --ranks-per-machine "$RANKS_PER_MACHINE" >"$WORK/mesh.yaml"
"$CLI" run --config "$WORK/mesh.yaml" --coordinator "$TRAINMESH_COORDINATOR" --output "$WORK/mesh.raw.json" \
  >"$WORK/mesh-run.out" 2>"$WORK/mesh-run.err" || { cat "$WORK/mesh-run.err" >&2; fail "mesh job failed"; }
grep "\[job-" "$WORK/mesh-run.err" | sed 's/^/[e2e-mesh]   /'

log "same job on one machine"
python3 scripts/e2e/bench_config.py --name single --machines 1 --ranks-per-machine "$((2 * RANKS_PER_MACHINE))" >"$WORK/single.yaml"
"$CLI" benchmark --config "$WORK/single.yaml" --output "$WORK/single.raw.json" >"$WORK/single.log" 2>&1 ||
  fail "single-machine run failed"

with_topology "$WORK/single.raw.json" "$WORK/single.json" single 1
with_topology "$WORK/mesh.raw.json" "$WORK/mesh.json" mesh 2
python3 scripts/e2e/compare.py "$WORK/single.json" "$WORK/mesh.json" || fail "1-machine and 2-machine results differ"
log "ok: the mesh trained the same model as one machine"

log "killing one agent and its training processes mid-job"
python3 scripts/e2e/bench_config.py --name long --machines 2 --ranks-per-machine 1 --epochs "$LONG_JOB_EPOCHS" >"$WORK/long.yaml"
"$CLI" run --config "$WORK/long.yaml" --coordinator "$TRAINMESH_COORDINATOR" --poll-interval 1s \
  >"$WORK/long-run.out" 2>"$WORK/long-run.err" &
RUN_PID=$!
wait_for_nodes BUSY 2 "$READY_TIMEOUT_SECONDS"
sleep 3
victim_dir="$WORK/agent2"
victim_node=$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["node_id"])' "$victim_dir/node.json")
kill -9 "${AGENT_PIDS[1]}"
pkill -9 -f "$victim_dir/jobs" || true
if wait "$RUN_PID"; then fail "job succeeded although a machine died"; fi
grep -q "FAILED" "$WORK/long-run.err" || fail "job did not report FAILED"
log "ok: job failed: $(grep -o 'FAILED.*' "$WORK/long-run.err" | tail -1 | cut -c1-140)"

waited=0
until "$CLI" nodes | "$PYTHON" -c 'import json, sys; nodes = {n["id"]: n["state"] for n in json.load(sys.stdin)}; sys.exit(nodes.get(sys.argv[1]) != "LOST")' "$victim_node"; do
  ((waited < LOST_TIMEOUT_SECONDS)) || fail "killed node $victim_node not LOST within ${LOST_TIMEOUT_SECONDS}s"
  sleep 1
  waited=$((waited + 1))
done
log "ok: killed node $victim_node is LOST"
wait_for_nodes READY 1 "$READY_TIMEOUT_SECONDS"

log "PASS"
