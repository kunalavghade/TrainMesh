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
# Usage: scripts/e2e/minikube.sh [up|build|bench|down|all]    (default: all = up build bench)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PROFILE=${PROFILE:-trainmesh}
NODE_COUNT=2
NODE_CPUS=${NODE_CPUS:-5}
NODE_MEMORY=${NODE_MEMORY:-3g}
POD_CPUS=${POD_CPUS:-2}
RANKS_PER_CPU=1
MEMORY_PER_RANK_MI=${MEMORY_PER_RANK_MI:-512}
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
  minikube -p "$PROFILE" image load "$IMAGE"
}

# job_manifest JOB MACHINES CPUS RANKS prints a headless Service and an Indexed Job.
job_manifest() {
  local job=$1 machines=$2 cpus=$3 ranks=$4
  local master_addr="127.0.0.1"
  [[ "$machines" -gt 1 ]] && master_addr="$job-0.$job"
  local memory_request=$((ranks * MEMORY_PER_RANK_MI))
  local memory_limit=$((memory_request * 3 / 2))
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

cluster_down() {
  log "deleting cluster '$PROFILE'"
  minikube delete -p "$PROFILE"
}

case "${1:-all}" in
  up) cluster_up ;;
  build) build_image ;;
  bench) bench ;;
  down) cluster_down ;;
  all)
    cluster_up
    build_image
    bench
    ;;
  *) fail "unknown command '$1' (use up, build, bench, down or all)" ;;
esac
