#!/usr/bin/env bash
# Runs scripts/e2e/wan.sh inside one Linux container (macOS has no tc netem).
#   - Checks the Docker VM has room (it is shared with your other containers), then caps the
#     test container's memory so an overrun is killed inside it, never elsewhere.
#   - Cross-compiles meshd/tm-agent/trainmesh for the VM's architecture, builds
#     deploy/docker/wan.Dockerfile (runtime image + tmcomm wheel + iproute2).
#   - Runs wan.sh with CAP_NET_ADMIN; results land in benchmark-results/e2e/wan-<timestamp>/.
#
# Usage: scripts/e2e/wan-docker.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BASE_IMAGE=${BASE_IMAGE:-trainmesh-runtime:dev}
IMAGE=${WAN_IMAGE:-trainmesh-wan:dev}
MEMORY_LIMIT_MI=${WAN_MEMORY_MI:-1800}
MEMORY_HEADROOM_MI=256
OUT=$ROOT/benchmark-results/e2e/wan-$(date +%Y%m%d-%H%M%S)

log() { printf '[e2e-wan-docker] %s\n' "$*"; }
fail() {
  printf '[e2e-wan-docker] FAIL: %s\n' "$*" >&2
  exit 1
}

docker info >/dev/null 2>&1 || fail "Docker is not running"
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  log "building $BASE_IMAGE"
  docker build -f "$ROOT/deploy/docker/runtime.Dockerfile" -t "$BASE_IMAGE" "$ROOT"
fi

available=$(docker run --rm "$BASE_IMAGE" awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo | tr -d '\r')
needed=$((MEMORY_LIMIT_MI + MEMORY_HEADROOM_MI))
[[ -n "$available" && "$available" -ge "$needed" ]] ||
  fail "Docker VM has ${available:-?} MiB available, the test needs $needed MiB; stop other containers or raise Docker's memory"
log "memory ok: $available MiB available, container capped at $MEMORY_LIMIT_MI MiB"

arch=$(docker version --format '{{.Server.Arch}}')
log "cross-compiling Go binaries for linux/$arch"
mkdir -p "$ROOT/deploy/docker/.wan-bin"
for binary in trainmesh meshd tm-agent; do
  (cd "$ROOT" && CGO_ENABLED=0 GOOS=linux GOARCH="$arch" go build -o "deploy/docker/.wan-bin/$binary" "./cmd/$binary")
done

log "building $IMAGE (first build compiles tmcomm: a few minutes)"
docker build -q -f "$ROOT/deploy/docker/wan.Dockerfile" --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE" "$ROOT" >/dev/null

mkdir -p "$OUT"
log "running wan.sh; results in ${OUT#"$ROOT"/}"
docker run --rm --name "trainmesh-wan-e2e-$$" --cap-add NET_ADMIN --memory "${MEMORY_LIMIT_MI}m" \
  -v "$ROOT/scripts/e2e:/opt/trainmesh/e2e:ro" -v "$OUT:/work" -e WAN_WORK=/work/run \
  -e WAN_DELAY_MS -e WAN_RATE -e WAN_INNER_STEPS -e WAN_MIN_UTILIZATION \
  "$IMAGE" bash /opt/trainmesh/e2e/wan.sh
cp "$OUT/run/summary.md" "$OUT/summary.md" 2>/dev/null || true
