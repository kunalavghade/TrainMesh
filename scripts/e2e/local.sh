#!/usr/bin/env bash
# End-to-end test of the trainmesh CLI on this machine: build, doctor, plan, run, resume,
# benchmark and the rejection paths. Exits non-zero on the first failed check.
#
# Usage: scripts/e2e/local.sh            (set KEEP_WORK=1 to keep the work directory)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${TRAINMESH_PYTHON:-$ROOT/.venv/bin/python}
CLI=$ROOT/bin/trainmesh
WORK=$(mktemp -d "${TMPDIR:-/tmp}/trainmesh-e2e-local.XXXXXX")
RANKS=2
GLOBAL_BATCH=128
SAMPLES=4096
STEPS_PER_EPOCH=$((SAMPLES / GLOBAL_BATCH))

cleanup() {
  if [[ "${KEEP_WORK:-0}" == "1" ]]; then
    echo "[e2e-local] work directory kept: $WORK"
  else
    rm -rf -- "$WORK"
  fi
}
trap cleanup EXIT

log() { printf '[e2e-local] %s\n' "$*"; }
fail() {
  printf '[e2e-local] FAIL: %s (logs in %s)\n' "$*" >&2
  KEEP_WORK=1
  exit 1
}

# json_field FILE KEY prints a top-level field of a JSON file.
json_field() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }

expect_equal() {
  local label=$1 actual=$2 expected=$3
  [[ "$actual" == "$expected" ]] || fail "$label: expected '$expected', got '$actual'"
  log "ok: $label = $actual"
}

write_config() {
  local epochs=$1
  "$PYTHON" "$ROOT/scripts/e2e/bench_config.py" \
    --name e2e-local --machines 1 --ranks-per-machine "$RANKS" \
    --global-batch "$GLOBAL_BATCH" --samples "$SAMPLES" \
    --input-size 32 --hidden-size 64 --output-size 4 --epochs "$epochs" \
    --checkpoint-dir "$WORK/checkpoints" --results-path "$WORK/result.json" >"$WORK/config.yaml"
}

cd "$ROOT"

log "building CLI"
go build -o "$CLI" ./cmd/trainmesh

log "doctor"
"$CLI" doctor >"$WORK/doctor.json" 2>"$WORK/doctor.err" || fail "doctor failed"
expect_equal "doctor gloo" "$(json_field "$WORK/doctor.json" gloo)" "True"

log "plan"
write_config 2
"$CLI" plan --config "$WORK/config.yaml" >"$WORK/plan.json" || fail "plan failed"
expect_equal "plan world_size" "$(json_field "$WORK/plan.json" world_size)" "$RANKS"
expect_equal "plan global batch" "$(json_field "$WORK/plan.json" effective_global_batch)" "$GLOBAL_BATCH"

log "run (2 epochs)"
"$CLI" run --config "$WORK/config.yaml" >"$WORK/run.log" 2>&1 || fail "run failed"
expect_equal "epoch events" "$(grep -c '"event": "epoch"' "$WORK/run.log")" "2"
expect_equal "replicas consistent" "$(json_field "$WORK/result.json" parameters_consistent)" "True"
expect_equal "checkpoint format" "$(json_field "$WORK/checkpoints/latest/metadata.json" format_version)" "2"
expect_equal "checkpoint epoch" "$(json_field "$WORK/checkpoints/latest/metadata.json" epoch)" "2"
[[ -z "$(find "$WORK/checkpoints" -name '*.pt')" ]] || fail "pickle checkpoint written"

log "resume to epoch 3"
write_config 3
"$CLI" run --config "$WORK/config.yaml" --resume "$WORK/checkpoints/latest" >"$WORK/resume.log" 2>&1 ||
  fail "resume failed"
expect_equal "resumed epoch" "$(json_field "$WORK/checkpoints/latest/metadata.json" epoch)" "3"
expect_equal "resumed global_step" \
  "$(json_field "$WORK/checkpoints/latest/metadata.json" global_step)" "$((3 * STEPS_PER_EPOCH))"

log "benchmark"
"$CLI" benchmark --config "$WORK/config.yaml" --output "$WORK/bench.json" >"$WORK/bench.log" 2>&1 ||
  fail "benchmark failed"
"$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["samples_per_second"] > 0' \
  "$WORK/bench.json" || fail "benchmark reported no throughput"
log "ok: benchmark samples/s = $(json_field "$WORK/bench.json" samples_per_second)"

log "rejects a v1 (pickle) checkpoint"
mkdir -p "$WORK/v1"
printf '{"version": 1, "world_size": %s}\n' "$RANKS" >"$WORK/v1/metadata.json"
if "$CLI" run --config "$WORK/config.yaml" --resume "$WORK/v1" >"$WORK/v1.log" 2>&1; then
  fail "v1 checkpoint was accepted"
fi
grep -q CHECKPOINT_INVALID "$WORK/v1.log" || fail "v1 rejection did not report CHECKPOINT_INVALID"
log "ok: v1 checkpoint rejected"

log "rejects workers > 1 until the multi-machine launcher exists"
sed 's/workers: 1/workers: 2/' "$WORK/config.yaml" >"$WORK/multi.yaml"
if "$CLI" run --config "$WORK/multi.yaml" >"$WORK/multi.log" 2>&1; then
  fail "workers: 2 was accepted"
fi
grep -q "multi-machine execution is not yet implemented" "$WORK/multi.log" || fail "wrong rejection message"
log "ok: workers > 1 rejected"

log "PASS"
