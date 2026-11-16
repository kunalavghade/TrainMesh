#!/usr/bin/env bash
# Multi-node dev test on minikube: kubernetes nodes stand in for machines.
#
# Runs one training job in three topologies with the same global batch, then compares them:
#   single     1 machine,  POD_CPUS CPUs
#   single-2x  1 machine,  2 x POD_CPUS CPUs   (more cores, no network)
#   multi      2 machines, POD_CPUS CPUs each  (same cores as single-2x, over the pod network)
# Every topology must reach the same final loss; compare.py fails the run if not.
#
# All nodes share one Docker host, so this measures correctness and communication overhead,
# not the speed of real separate machines. Use manual-node.sh on physical machines for that.
#
# Usage: scripts/e2e/minikube.sh [up|build|bench|mesh|mesh-down|down|all]   (default: all = up build bench)
#   mesh: Phase 1 proof — meshd and one tm-agent per node; a job through `trainmesh run
#         --coordinator` must match one machine; deleting an agent pod must fail the job cleanly.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROFILE=${PROFILE:-trainmesh}
NODE_COUNT=2
NODE_CPUS=${NODE_CPUS:-5}
NODE_MEMORY=${NODE_MEMORY:-3g}
POD_CPUS=${POD_CPUS:-2}
RANKS_PER_CPU=1
# Measured 2026-11-16 (torch 2.15 CPU, MLP 256-1024-10): 888 MiB for 2 ranks, 1440 MiB for 4.
MEMORY_BASE_MI=${MEMORY_BASE_MI:-340}
MEMORY_PER_RANK_MI=${MEMORY_PER_RANK_MI:-280}
# Free memory the Docker VM must keep beyond the scenario's own estimate.
MEMORY_HEADROOM_MI=${MEMORY_HEADROOM_MI:-256}
IMAGE=${IMAGE:-trainmesh-runtime:dev}
TORCH_VERSION=${TORCH_VERSION:-2.15.0}
JOB_TIMEOUT_SECONDS=${JOB_TIMEOUT_SECONDS:-900}
POLL_SECONDS=5
RENDEZVOUS_PORT=29500
ENVIRONMENT="minikube, ${NODE_COUNT} kubernetes nodes on one Docker host"
RESULTS=${RESULTS:-$ROOT/benchmark-results/e2e/minikube-$(date +%Y%m%d-%H%M%S)}

log() { printf '[e2e-minikube] %s\n' "$*"; }
fail() {
  printf '[e2e-minikube] FAIL: %s\n' "$*" >&2
  exit 1
}
kube() { kubectl --context "$PROFILE" "$@"; }

require() {
  local tool
  for tool in "$@"; do
    command -v "$tool" >/dev/null || fail "$tool is not installed"
  done
}

cluster_up() {
  require minikube kubectl docker
  if minikube -p "$PROFILE" status >/dev/null 2>&1; then
    log "cluster '$PROFILE' already running"
  else
    log "starting $NODE_COUNT-node cluster '$PROFILE' ($NODE_CPUS CPUs, $NODE_MEMORY each)"
    minikube start -p "$PROFILE" --driver=docker --nodes="$NODE_COUNT" \
      --cpus="$NODE_CPUS" --memory="$NODE_MEMORY"
  fi
  kube wait --for=condition=Ready nodes --all --timeout=300s
  local ready
  ready=$(kube get nodes --no-headers | grep -c ' Ready ')
  [[ "$ready" -ge "$NODE_COUNT" ]] || fail "expected $NODE_COUNT ready nodes, found $ready"
}

build_image() {
  require docker minikube
  log "building $IMAGE (torch $TORCH_VERSION)"
  docker build -f "$ROOT/deploy/docker/runtime.Dockerfile" \
    --build-arg TORCH_VERSION="$TORCH_VERSION" -t "$IMAGE" "$ROOT"
  log "loading $IMAGE into every node"
  minikube -p "$PROFILE" image load --overwrite=true "$IMAGE"
}

# job_manifest JOB MACHINES CPUS RANKS prints a headless Service and an Indexed Job.
job_manifest() {
  local job=$1 machines=$2 cpus=$3 ranks=$4
  local master_addr="127.0.0.1"
  [[ "$machines" -gt 1 ]] && master_addr="$job-0.$job"
  local memory_request
  memory_request=$(pod_memory_mi "$ranks")
  local memory_limit=$((memory_request * 2))
  cat <<EOF
apiVersion: v1
kind: Service
metadata:
  name: $job
  labels: {app: trainmesh-e2e}
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector: {job-name: $job}
  ports: [{name: rendezvous, port: $RENDEZVOUS_PORT}]
---
apiVersion: batch/v1
kind: Job
metadata:
  name: $job
  labels: {app: trainmesh-e2e}
spec:
  completionMode: Indexed
  completions: $machines
  parallelism: $machines
  backoffLimit: 0
  template:
    metadata:
      labels: {app: trainmesh-e2e, trainmesh-run: $job}
    spec:
      subdomain: $job
      restartPolicy: Never
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector: {matchLabels: {trainmesh-run: $job}}
              topologyKey: kubernetes.io/hostname
      volumes:
        - name: config
          configMap: {name: $job-config}
      containers:
        - name: trainer
          image: $IMAGE
          imagePullPolicy: Never
          env:
            - {name: NNODES, value: "$machines"}
            - {name: NPROC, value: "$ranks"}
            - {name: MASTER_ADDR, value: "$master_addr"}
            - {name: GLOO_SOCKET_IFNAME, value: eth0}
            - {name: OMP_NUM_THREADS, value: "1"}
          command: ["/bin/sh", "-c"]
          args:
            - |
              set -eu
              python -m torch.distributed.run --nnodes="\$NNODES" --node_rank="\$JOB_COMPLETION_INDEX" \\
                --nproc_per_node="\$NPROC" --master_addr="\$MASTER_ADDR" --master_port=$RENDEZVOUS_PORT \\
                -m trainmesh_runtime --config /config/config.yaml
              if [ -f /tmp/result.json ]; then echo "TRAINMESH_RESULT \$(tr -d '\\n' < /tmp/result.json)"; fi
          resources:
            requests: {cpu: "$cpus", memory: "${memory_request}Mi"}
            limits: {cpu: "$cpus", memory: "${memory_limit}Mi"}
          volumeMounts:
            - {name: config, mountPath: /config}
EOF
}

pod_memory_mi() { echo $((MEMORY_BASE_MI + MEMORY_PER_RANK_MI * $1)); }

# check_memory_headroom MACHINES RANKS fails early when the shared Docker VM cannot fit the scenario.
# All nodes run on one Docker VM, so its free memory is the real limit (see docs/E2E_TESTING.md).
check_memory_headroom() {
  local machines=$1 ranks=$2 available required
  available=$(minikube -p "$PROFILE" ssh -- free -m | awk '/Mem:/ {print $7}')
  required=$((machines * $(pod_memory_mi "$ranks") + MEMORY_HEADROOM_MI))
  if [[ -z "$available" || "$available" -lt "$required" ]]; then
    fail "Docker VM has ${available:-?} MiB available, scenario needs ${required} MiB; free memory (stop other containers or raise Docker's memory limit) and retry"
  fi
  log "memory ok: ${available} MiB available, ${required} MiB needed"
}

wait_for_job() {
  local job=$1 machines=$2 waited=0 succeeded failed
  while ((waited < JOB_TIMEOUT_SECONDS)); do
    succeeded=$(kube get job "$job" -o jsonpath='{.status.succeeded}')
    failed=$(kube get job "$job" -o jsonpath='{.status.failed}')
    [[ "${succeeded:-0}" -ge "$machines" ]] && return 0
    [[ "${failed:-0}" -gt 0 ]] && return 1
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
  done
  log "timed out after ${JOB_TIMEOUT_SECONDS}s"
  return 1
}

delete_job() {
  local job=$1
  kube delete job "$job" --ignore-not-found --wait=false >/dev/null
  kube delete service "$job" --ignore-not-found --wait=false >/dev/null
  kube delete configmap "$job-config" --ignore-not-found --wait=false >/dev/null
}

# run_scenario NAME MACHINES CPUS_PER_MACHINE writes $RESULTS/NAME.json.
run_scenario() {
  local name=$1 machines=$2 cpus=$3
  local ranks=$((cpus * RANKS_PER_CPU))
  local job="tm-$name-$(date +%s)"
  log "scenario $name: $machines machine(s) x $cpus CPU(s), $ranks rank(s) each"
  check_memory_headroom "$machines" "$ranks"

  python3 "$ROOT/scripts/e2e/bench_config.py" --name "$name" --machines "$machines" \
    --ranks-per-machine "$ranks" --results-path /tmp/result.json >"$RESULTS/$name.yaml"
  kube create configmap "$job-config" --from-file=config.yaml="$RESULTS/$name.yaml" >/dev/null
  job_manifest "$job" "$machines" "$cpus" "$ranks" | kube apply -f - >/dev/null

  local ok=1
  wait_for_job "$job" "$machines" || ok=0
  kube logs -l job-name="$job" --prefix --tail=-1 >"$RESULTS/$name.log" 2>&1 || true
  local nodes
  nodes=$(kube get pods -l job-name="$job" -o jsonpath='{.items[*].spec.nodeName}')
  if [[ "$ok" -eq 0 ]]; then
    kube get pods -l job-name="$job" -o wide >&2 || true
    delete_job "$job"
    fail "scenario $name failed; see $RESULTS/$name.log"
  fi
  delete_job "$job"

  local result_line
  result_line=$(grep -h 'TRAINMESH_RESULT ' "$RESULTS/$name.log" | head -1 | sed 's/.*TRAINMESH_RESULT //')
  [[ -n "$result_line" ]] || fail "scenario $name printed no result; see $RESULTS/$name.log"
  python3 - "$RESULTS/$name.json" "$result_line" <<EOF
import json, sys
result = json.loads(sys.argv[2])
result["topology"] = {
    "scenario": "$name",
    "machines": $machines,
    "ranks_per_machine": $ranks,
    "cpus_per_machine": $cpus,
    "cpus_total": $machines * $cpus,
    "nodes": "$nodes".split(),
    "environment": "$ENVIRONMENT",
}
json.dump(result, open(sys.argv[1], "w"), indent=2)
EOF
  local distinct
  distinct=$(tr ' ' '\n' <<<"$nodes" | sort -u | grep -c . || true)
  [[ "$distinct" -eq "$machines" ]] || fail "scenario $name ran on $distinct node(s), expected $machines"
  log "scenario $name done on node(s): $nodes"
}

bench() {
  require kubectl python3
  mkdir -p "$RESULTS"
  run_scenario single 1 "$POD_CPUS"
  run_scenario single-2x 1 "$((POD_CPUS * 2))"
  run_scenario multi 2 "$POD_CPUS"
  log "comparing (results in $RESULTS)"
  python3 "$ROOT/scripts/e2e/compare.py" "$RESULTS/single.json" "$RESULTS/single-2x.json" "$RESULTS/multi.json"
}

# ── Phase 1 mesh: meshd + one tm-agent per kubernetes node ─────────────────────────────────
MESH_IMAGE=${MESH_IMAGE:-trainmesh-mesh:dev}
MESHD_STATE=/var/lib/trainmesh/meshd
AGENT_STATE=/var/lib/trainmesh/agent
MESHD_PORT=7443
MESH_TIMEOUT_SECONDS=${MESH_TIMEOUT_SECONDS:-180}
LONG_JOB_EPOCHS=200

build_mesh_image() {
  build_image
  local context arch binary
  context=$(mktemp -d "${TMPDIR:-/tmp}/trainmesh-mesh-image.XXXXXX")
  arch=$(docker version --format '{{.Server.Arch}}')
  log "cross-compiling meshd, tm-agent and trainmesh for linux/$arch"
  for binary in trainmesh meshd tm-agent; do
    (cd "$ROOT" && CGO_ENABLED=0 GOOS=linux GOARCH="$arch" go build -o "$context/$binary" "./cmd/$binary")
  done
  docker build -f "$ROOT/deploy/docker/mesh.Dockerfile" --build-arg BASE_IMAGE="$IMAGE" -t "$MESH_IMAGE" "$context"
  rm -rf -- "$context"
  minikube -p "$PROFILE" image load --overwrite=true "$MESH_IMAGE"
}

meshd_manifest() {
  cat <<EOF
apiVersion: v1
kind: Service
metadata: {name: trainmesh-meshd, labels: {app: trainmesh-mesh}}
spec:
  selector: {app: trainmesh-meshd}
  ports: [{name: grpc, port: $MESHD_PORT}]
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: trainmesh-meshd, labels: {app: trainmesh-mesh}}
spec:
  replicas: 1
  selector: {matchLabels: {app: trainmesh-meshd}}
  template:
    metadata: {labels: {app: trainmesh-meshd}}
    spec:
      containers:
        - name: meshd
          image: $MESH_IMAGE
          imagePullPolicy: Never
          command: [meshd, --state-dir, $MESHD_STATE, --listen, "0.0.0.0:$MESHD_PORT",
                    --advertise, "trainmesh-meshd:$MESHD_PORT", --heartbeat-interval, 1s,
                    --suspect-after, 3s, --lost-after, 6s]
          resources: {requests: {cpu: 100m, memory: 64Mi}, limits: {cpu: 500m, memory: 256Mi}}
          volumeMounts: [{name: state, mountPath: $MESHD_STATE}]
      volumes: [{name: state, emptyDir: {}}]
EOF
}

# agents_manifest: one agent per node; sized for 1 rank, limit high enough for a 2-rank baseline.
agents_manifest() {
  local request limit
  request=$(pod_memory_mi 1)
  limit=$(($(pod_memory_mi 2) * 2))
  cat <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata: {name: trainmesh-agent, labels: {app: trainmesh-mesh}}
spec:
  serviceName: trainmesh-agent
  replicas: $NODE_COUNT
  podManagementPolicy: Parallel
  selector: {matchLabels: {app: trainmesh-agent}}
  template:
    metadata: {labels: {app: trainmesh-agent}}
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector: {matchLabels: {app: trainmesh-agent}}
              topologyKey: kubernetes.io/hostname
      containers:
        - name: agent
          image: $MESH_IMAGE
          imagePullPolicy: Never
          command: ["/bin/sh", "-c", "exec tm-agent --state-dir $AGENT_STATE --join \"\$JOIN\""]
          env: [{name: JOIN, valueFrom: {secretKeyRef: {name: trainmesh-join, key: join}}}]
          resources:
            requests: {cpu: "$POD_CPUS", memory: "${request}Mi"}
            limits: {cpu: "$POD_CPUS", memory: "${limit}Mi"}
          volumeMounts: [{name: state, mountPath: /var/lib/trainmesh}]
      volumes: [{name: state, emptyDir: {}}]
EOF
}

mesh_cli() { kube exec deploy/trainmesh-meshd -- trainmesh "$@" --coordinator "localhost:$MESHD_PORT" --admin-dir "$MESHD_STATE"; }

mesh_node_count() {
  mesh_cli nodes 2>/dev/null | python3 -c 'import json, sys; print(sum(n["state"] == sys.argv[1] for n in json.load(sys.stdin)))' "$1"
}

wait_for_mesh_nodes() {
  local state=$1 count=$2 waited=0
  until [[ "$(mesh_node_count "$state" || echo 0)" -ge "$count" ]]; do
    ((waited < MESH_TIMEOUT_SECONDS)) || fail "expected $count node(s) $state within ${MESH_TIMEOUT_SECONDS}s"
    sleep 2
    waited=$((waited + 2))
  done
  log "mesh: $count node(s) $state"
}

# copy_into POD_OR_DEPLOY LOCAL REMOTE
copy_into() { kube exec -i "$1" -- sh -c "cat > $3" <"$2"; }

mesh_topology() {
  python3 - "$@" <<'EOF'
import json, sys
raw, out, scenario, machines, nodes = sys.argv[1:]
result = json.load(open(raw))
result["topology"] = {"scenario": scenario, "machines": int(machines), "nodes": nodes.split(),
                      "environment": "minikube: meshd + tm-agent pods on separate nodes"}
json.dump(result, open(out, "w"), indent=2)
EOF
}

mesh_up() {
  require kubectl python3
  mesh_down >/dev/null 2>&1 || true
  check_memory_headroom "$NODE_COUNT" 1
  log "deploying meshd"
  meshd_manifest | kube apply -f - >/dev/null
  kube rollout status deploy/trainmesh-meshd --timeout=120s >/dev/null
  local join
  join=$(mesh_cli token create --uses "$NODE_COUNT" --ttl 10m 2>/dev/null) || fail "token create failed"
  kube create secret generic trainmesh-join --from-literal=join="$join" >/dev/null
  log "deploying $NODE_COUNT agents (one per node)"
  agents_manifest | kube apply -f - >/dev/null
  wait_for_mesh_nodes READY "$NODE_COUNT"
  local nodes
  nodes=$(kube get pods -l app=trainmesh-agent -o jsonpath='{.items[*].spec.nodeName}')
  [[ $(tr ' ' '\n' <<<"$nodes" | sort -u | grep -c .) -eq "$NODE_COUNT" ]] || fail "agents are not on $NODE_COUNT distinct nodes: $nodes"
  log "agents run on nodes: $nodes"
}

mesh_bench() {
  mkdir -p "$RESULTS"
  local nodes
  nodes=$(kube get pods -l app=trainmesh-agent -o jsonpath='{.items[*].spec.nodeName}')
  mesh_cli nodes >"$RESULTS/mesh-nodes.json"

  log "mesh job: $NODE_COUNT machines x 1 rank through trainmesh run --coordinator"
  python3 "$ROOT/scripts/e2e/bench_config.py" --name mesh --machines "$NODE_COUNT" --ranks-per-machine 1 >"$RESULTS/mesh.yaml"
  copy_into deploy/trainmesh-meshd "$RESULTS/mesh.yaml" /tmp/mesh.yaml
  mesh_cli run --config /tmp/mesh.yaml >"$RESULTS/mesh.raw.json" 2>"$RESULTS/mesh-run.log" ||
    fail "mesh job failed; see $RESULTS/mesh-run.log"
  grep "\[job-" "$RESULTS/mesh-run.log" | sed 's/^/[e2e-minikube]   /'

  log "baseline: the same job on one machine (trainmesh-agent-0, $NODE_COUNT ranks)"
  python3 "$ROOT/scripts/e2e/bench_config.py" --name single --machines 1 --ranks-per-machine "$NODE_COUNT" >"$RESULTS/single.yaml"
  copy_into trainmesh-agent-0 "$RESULTS/single.yaml" /tmp/single.yaml
  kube exec trainmesh-agent-0 -- trainmesh benchmark --config /tmp/single.yaml --output /tmp/single.json >"$RESULTS/single.log" 2>&1 ||
    fail "baseline failed; see $RESULTS/single.log"
  kube exec trainmesh-agent-0 -- cat /tmp/single.json >"$RESULTS/single.raw.json"

  mesh_topology "$RESULTS/single.raw.json" "$RESULTS/single.json" single 1 "$(cut -d' ' -f1 <<<"$nodes")"
  mesh_topology "$RESULTS/mesh.raw.json" "$RESULTS/mesh.json" mesh "$NODE_COUNT" "$nodes"
  python3 "$ROOT/scripts/e2e/compare.py" "$RESULTS/single.json" "$RESULTS/mesh.json" || fail "mesh and single results differ"
}

mesh_failover() {
  log "failure: deleting the node-rank-1 agent pod mid-job"
  python3 "$ROOT/scripts/e2e/bench_config.py" --name long --machines "$NODE_COUNT" --ranks-per-machine 1 --epochs "$LONG_JOB_EPOCHS" >"$RESULTS/long.yaml"
  copy_into deploy/trainmesh-meshd "$RESULTS/long.yaml" /tmp/long.yaml
  mesh_cli run --config /tmp/long.yaml --poll-interval 1s >"$RESULTS/long-run.out" 2>"$RESULTS/long-run.log" &
  local run_pid=$!
  wait_for_mesh_nodes BUSY "$NODE_COUNT"
  sleep 5
  local victim_node victim_pod
  read -r victim_node victim_pod < <(mesh_cli nodes | python3 -c '
import json, sys
for n in json.load(sys.stdin):
    if n["state"] == "BUSY" and n["hostname"] != "trainmesh-agent-0":
        print(n["id"], n["hostname"]); break')
  [[ -n "$victim_pod" ]] || fail "could not find a busy agent other than trainmesh-agent-0"
  kube delete pod "$victim_pod" --grace-period=0 --force >/dev/null 2>&1
  if wait "$run_pid"; then fail "job succeeded although a machine was deleted"; fi
  grep -q "FAILED" "$RESULTS/long-run.log" || fail "job did not report FAILED"
  log "job failed as expected: $(grep -o 'FAILED.*' "$RESULTS/long-run.log" | tail -1 | cut -c1-140)"
  local waited=0
  until mesh_cli nodes | python3 -c 'import json, sys; nodes = {n["id"]: n["state"] for n in json.load(sys.stdin)}; sys.exit(nodes.get(sys.argv[1]) != "LOST")' "$victim_node"; do
    ((waited < MESH_TIMEOUT_SECONDS)) || fail "deleted node $victim_node not LOST"
    sleep 2
    waited=$((waited + 2))
  done
  log "deleted node $victim_node ($victim_pod) is LOST; mesh results in $RESULTS"
}

mesh_down() {
  kube delete statefulset/trainmesh-agent deploy/trainmesh-meshd service/trainmesh-meshd secret/trainmesh-join \
    --ignore-not-found --wait=false
}

mesh() {
  build_mesh_image
  mesh_up
  mesh_bench
  mesh_failover
  mesh_down >/dev/null
  log "mesh PASS"
}

cluster_down() {
  log "deleting cluster '$PROFILE'"
  minikube delete -p "$PROFILE"
}

case "${1:-all}" in
  up) cluster_up ;;
  build) build_image ;;
  bench) bench ;;
  down) cluster_down ;;
  mesh) mesh ;;
  mesh-down) mesh_down ;;
  all)
    cluster_up
    build_image
    bench
    ;;
  *) fail "unknown command '$1' (use up, build, bench, mesh, mesh-down, down or all)" ;;
esac
