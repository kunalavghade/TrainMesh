#!/usr/bin/env bash
# Phase 3 proof: DiLoCo across 3 islands over an emulated WAN, inside one Linux network namespace.
#   1. tc netem shapes loopback (default 25 ms each way = 50 ms RTT, 100 Mbit/s); measure it.
#   2. meshd + 3 tm-agents (one island each) join over the shaped link.
#   3. The same model and samples train twice: data_parallel (all-reduce every step) and diloco
#      (int8 pseudo-gradients every H steps through tmcomm over mTLS).
#   4. A longer diloco job loses its root island (agent and training killed mid-run): the other
#      two islands must elect a new root and finish, and the job must still succeed.
#   5. wan_report.py judges: accuracy (diloco within 2 points of data_parallel, both >= 85%),
#      identical island models, diloco faster, compute utilization >= 60%, quorum survival.
#
# Needs: tc with CAP_NET_ADMIN (root in a container started with --cap-add NET_ADMIN, or sudo on
# a CI host), meshd/tm-agent/trainmesh on TRAINMESH_BIN, Python with the runtime and tmcomm.
# On a Mac use scripts/e2e/wan-docker.sh, which builds the image and runs this inside it.
set -euo pipefail

E2E_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON=${TRAINMESH_PYTHON:-python3}
BIN=${TRAINMESH_BIN:-/usr/local/bin}
CLI=$BIN/trainmesh
WORK=${WAN_WORK:-$(mktemp -d "${TMPDIR:-/tmp}/trainmesh-e2e-wan.XXXXXX")}
E2E_TAG=e2e-wan
KEEP_WORK=1
DELAY_MS=${WAN_DELAY_MS:-25}
RATE=${WAN_RATE:-100mbit}
INNER_STEPS=${WAN_INNER_STEPS:-64}
KILL_EPOCHS=${WAN_KILL_EPOCHS:-6}
KILLED_ISLAND=0 # the root: the survivors must elect a new one
MIN_UTILIZATION=${WAN_MIN_UTILIZATION:-0.60}
TC=${TC:-tc}
# Same model, data and samples for both strategies: 3 machines x 512 per step.
COMMON=(--machines 3 --ranks-per-machine 1 --global-batch 1536 --samples 98304
  --input-size 64 --hidden-size 1024 --output-size 10 --validation-samples 4096)
# shellcheck source=scripts/e2e/mesh-lib.sh
source "$E2E_DIR/mesh-lib.sh"
mkdir -p "$WORK"

remove_netem() { $TC qdisc del dev lo root 2>/dev/null || true; }
trap 'remove_netem; cleanup' EXIT

log "shaping loopback: ${DELAY_MS} ms each way, $RATE"
remove_netem
$TC qdisc add dev lo root netem delay "${DELAY_MS}ms" rate "$RATE" limit 100000 ||
  fail "tc netem failed (needs CAP_NET_ADMIN and the sch_netem kernel module)"
"$PYTHON" - "$WORK/network.json" "$DELAY_MS" "$RATE" <<'EOF'
import json, socket, statistics, sys, threading, time
out, delay, rate = sys.argv[1:]
server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen()
def serve():
    conn, _ = server.accept()
    while data := conn.recv(1 << 20):
        if data == b"p":
            conn.sendall(b"p")
    conn.sendall(b"done")
threading.Thread(target=serve, daemon=True).start()
client = socket.create_connection(server.getsockname())
client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
rtts = []
for _ in range(10):
    start = time.perf_counter(); client.sendall(b"p"); client.recv(1); rtts.append(time.perf_counter() - start)
payload = 12_500_000
start = time.perf_counter(); client.sendall(b"x" * payload); client.shutdown(socket.SHUT_WR)
while client.recv(1 << 16):
    pass
seconds = time.perf_counter() - start
json.dump({"description": f"tc netem delay {delay}ms each way, rate {rate}",
           "rtt_ms": statistics.median(rtts) * 1e3, "mbit_per_s": payload * 8 / seconds / 1e6},
          open(out, "w"), indent=2)
print(f"[e2e-wan]   measured RTT {statistics.median(rtts) * 1e3:.1f} ms, {payload * 8 / seconds / 1e6:.0f} Mbit/s")
EOF

start_meshd
log "joining 3 cpu agents (one island each)"
for i in 0 1 2; do join_agent "agent$i" --device cpu; done
wait_for_nodes READY 3 "$READY_TIMEOUT_SECONDS"

# run_job NAME [bench_config args...] trains through the CLI and keeps the result JSON.
run_job() {
  local name=$1
  shift
  "$PYTHON" "$E2E_DIR/bench_config.py" --name "$name" "${COMMON[@]}" "$@" >"$WORK/$name.yaml"
  "$CLI" run --config "$WORK/$name.yaml" --coordinator "$TRAINMESH_COORDINATOR" --output "$WORK/$name.json" \
    --poll-interval 1s >"$WORK/$name-run.out" 2>"$WORK/$name-run.err" || {
    tail -20 "$WORK/$name-run.err" >&2
    fail "$name job failed"
  }
  "$PYTHON" -c "import json, sys; r = json.load(open(sys.argv[1])); print('[e2e-wan]   %s: %.1f s, accuracy %.2f%%, utilization %.0f%%' % (sys.argv[2], r['duration_seconds'], 100 * r['validation']['final_accuracy'], 100 * r['compute_utilization']))" "$WORK/$name.json" "$name"
}

# job_id NAME prints the job ID the CLI reported for that run.
job_id() { grep -o 'job-[0-9a-f]*' "$WORK/$1-run.err" | head -1; }

# island_results JOB prints every island leader's result file for that job.
island_results() { ls "$WORK"/agent*/jobs/"$1"/result.json; }

log "data_parallel: all-reduce every step over the WAN"
# allreduce instead of DDP: same traffic, but its wait is measured instead of hidden in backward.
run_job ddp --strategy data_parallel --gradient-sync allreduce --epochs 3
log "diloco: int8 pseudo-gradients every $INNER_STEPS steps"
run_job diloco --strategy diloco --inner-steps "$INNER_STEPS" --epochs 3
mapfile -t DILOCO_RESULTS < <(island_results "$(job_id diloco)")
((${#DILOCO_RESULTS[@]} == 3)) || fail "expected 3 island results, found ${#DILOCO_RESULTS[@]}"

log "diloco again; island $KILLED_ISLAND (the root) is killed after its first outer round"
"$PYTHON" "$E2E_DIR/bench_config.py" --name diloco-kill "${COMMON[@]}" --strategy diloco \
  --inner-steps "$INNER_STEPS" --epochs "$KILL_EPOCHS" >"$WORK/diloco-kill.yaml"
"$CLI" run --config "$WORK/diloco-kill.yaml" --coordinator "$TRAINMESH_COORDINATOR" --poll-interval 1s \
  --output "$WORK/diloco-kill.json" >"$WORK/diloco-kill-run.out" 2>"$WORK/diloco-kill-run.err" &
RUN_PID=$!
waited=0
until KILL_JOB=$(job_id diloco-kill) && [[ -n "$KILL_JOB" ]]; do
  ((waited++ < 60)) || fail "kill job never submitted"
  sleep 1
done
victim_node=$("$CLI" job get "$KILL_JOB" | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["islands"][int(sys.argv[1])]["nodes"][0])' "$KILLED_ISLAND")
victim=""
for i in 0 1 2; do
  [[ "$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["node_id"])' "$WORK/agent$i/node.json")" == "$victim_node" ]] && victim=$i
done
[[ -n "$victim" ]] || fail "no agent owns $victim_node"
waited=0
until grep -q '"event": "outer_round"' "$WORK/agent$victim/jobs/$KILL_JOB/runtime.log" 2>/dev/null; do
  ((waited++ < 300)) || fail "island $KILLED_ISLAND never finished a round"
  sleep 0.5
done
kill -9 "${AGENT_PIDS[$victim]}"
pkill -9 -f "$WORK/agent$victim/jobs" || true
log "killed agent$victim ($victim_node) and its training process"
wait "$RUN_PID" || {
  tail -20 "$WORK/diloco-kill-run.err" >&2
  fail "the job failed although a quorum of islands survived"
}
"$CLI" job get "$KILL_JOB" >"$WORK/diloco-kill-job.json"
grep "\[job-" "$WORK/diloco-kill-run.err" | tail -4 | sed 's/^/[e2e-wan]   /'

remove_netem
"$PYTHON" "$E2E_DIR/wan_report.py" --ddp "$WORK/ddp.json" --diloco "${DILOCO_RESULTS[@]}" \
  --killed-job "$WORK/diloco-kill-job.json" --killed-island "$KILLED_ISLAND" \
  --network "$WORK/network.json" --summary "$WORK/summary.md" --min-utilization "$MIN_UTILIZATION" ||
  fail "WAN checks failed"
log "PASS"
