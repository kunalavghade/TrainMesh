# Shared helpers for the local mesh e2e scripts (sourced, not run).
# The caller sets E2E_TAG, ROOT, PYTHON, BIN, CLI and WORK before sourcing, then calls
# build_binaries, start_meshd and join_agent.
# shellcheck shell=bash

READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-60}
declare -a PIDS=()
declare -a AGENT_PIDS=()
LAST_JOIN=""

log() { printf '[%s] %s\n' "$E2E_TAG" "$*"; }
fail() {
  printf '[%s] FAIL: %s (logs in %s)\n' "$E2E_TAG" "$*" "$WORK" >&2
  KEEP_WORK=1
  exit 1
}
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  pkill -f "$WORK/" 2>/dev/null || true
  if [[ "${KEEP_WORK:-0}" == "1" ]]; then
    echo "[$E2E_TAG] work directory kept: $WORK"
  else
    rm -rf -- "$WORK"
  fi
}

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

build_binaries() {
  log "building binaries"
  go build -o "$BIN/trainmesh" ./cmd/trainmesh
  go build -o "$BIN/meshd" ./cmd/meshd
  go build -o "$BIN/tm-agent" ./cmd/tm-agent
}

# start_meshd exports TRAINMESH_COORDINATOR/TRAINMESH_ADMIN_DIR and waits until it answers.
start_meshd() {
  local port rendezvous_port
  port=$(free_port)
  rendezvous_port=$(free_port)
  export TRAINMESH_COORDINATOR="127.0.0.1:$port" TRAINMESH_ADMIN_DIR="$WORK/meshd"
  log "starting meshd on 127.0.0.1:$port"
  "$BIN/meshd" --state-dir "$WORK/meshd" --listen "127.0.0.1:$port" --advertise "127.0.0.1:$port" \
    --rendezvous-port "$rendezvous_port" --heartbeat-interval 1s --suspect-after 3s --lost-after 6s \
    >"$WORK/meshd.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 1 20); do "$CLI" nodes >/dev/null 2>&1 && break || sleep 0.5; done
  "$CLI" nodes >/dev/null 2>&1 || fail "meshd did not come up (see meshd.log)"
}

# join_agent NAME [tm-agent flags...] starts an agent with a fresh single-use token.
join_agent() {
  local name=$1 join
  shift
  join=$("$CLI" token create --ttl 5m 2>/dev/null) || fail "token create failed"
  "$BIN/tm-agent" --state-dir "$WORK/$name" --join "$join" "$@" >"$WORK/$name.log" 2>&1 &
  PIDS+=($!)
  AGENT_PIDS+=($!)
  LAST_JOIN=$join
}
