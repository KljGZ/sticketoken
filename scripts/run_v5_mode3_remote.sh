#!/usr/bin/env bash
set -Eeuo pipefail

# Formal Mode 3 V5 orchestration only.  It deliberately never invokes a
# repository-wide test command or any Mode 1/2/V1--V4 entry point.
ROOT="${V5_ROOT:-/home/jkl/StickyToken-v5}"
PYTHON="${V5_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
CONFIG="${V5_CONFIG:-configs/v5_mode3.yaml}"
OUTPUT="${V5_OUTPUT:-results/sticky_lab/sentence_t5_base/mode3_v5}"
WORKERS="${V5_WORKERS:-8}"
LOG_ROOT="${V5_LOG_ROOT:-$OUTPUT/orchestration_logs}"
TASKS=(prefix suffix random conditional shared)

cd "$ROOT"
mkdir -p "$LOG_ROOT"
EVENTS="$LOG_ROOT/events.jsonl"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree is dirty; refusing formal V5 run" >&2
  exit 1
fi

"$PYTHON" -m pytest -q \
  tests/test_mode3_v5_core.py \
  tests/test_mode3_v5_search.py \
  tests/test_mode3_v5_scope.py \
  tests/test_mode3_v5_publication.py \
  >"$LOG_ROOT/v5_tests.log" 2>&1
"$PYTHON" scripts/audit_v5_mode3.py --root "$ROOT" --config "$CONFIG" \
  --output "$LOG_ROOT/preflight_scope_audit.json"

event() {
  local label="$1"
  local state="$2"
  "$PYTHON" - "$EVENTS" "$label" "$state" <<'PY'
import datetime, json, os, sys
path, label, state = sys.argv[1:]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": label,
        "state": state,
        "pid": os.getpid(),
    }, sort_keys=True) + "\n")
PY
}

run_one() {
  local label="$1"
  local gpu="$2"
  shift 2
  event "$label" start
  if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m sticky_lab.mode3_v5.run \
      --config "$CONFIG" --output "$OUTPUT" --device cuda:0 "$@" \
      >"$LOG_ROOT/$label.log" 2>&1; then
    event "$label" complete
  else
    event "$label" failed
    return 1
  fi
}

declare -a PIDS=()
declare -a LABELS=()

flush_jobs() {
  local failed=0
  local index
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$index]}"; then
      echo "V5 job failed: ${LABELS[$index]}" >&2
      failed=1
    fi
  done
  PIDS=()
  LABELS=()
  if [[ "$failed" -ne 0 ]]; then
    exit 1
  fi
}

submit() {
  local label="$1"
  local gpu="$2"
  shift 2
  run_one "$label" "$gpu" "$@" &
  PIDS+=("$!")
  LABELS+=("$label")
  if [[ "${#PIDS[@]}" -ge "$WORKERS" ]]; then
    flush_jobs
  fi
}

run_one prepare 0 prepare
for shard in $(seq 0 7); do
  submit "calibrate_${shard}" "$((shard % WORKERS))" calibrate --shard "$shard" --shards 8
done
flush_jobs
"$PYTHON" -m sticky_lab.mode3_v5.run --config "$CONFIG" --output "$OUTPUT" \
  merge-calibration --shards 8 >"$LOG_ROOT/merge_calibration.log" 2>&1
event merge_calibration complete
"$PYTHON" -m sticky_lab.mode3_v5.run --config "$CONFIG" --output "$OUTPUT" register \
  >"$LOG_ROOT/register.log" 2>&1
event register complete

for task in "${TASKS[@]}"; do
  for shard in $(seq 0 7); do
    submit "screen_${task}_${shard}" "$((shard % WORKERS))" screen --task "$task" --shard "$shard" --shards 8
  done
done
flush_jobs

for index in "${!TASKS[@]}"; do
  task="${TASKS[$index]}"
  submit "formalize_${task}" "$((index % WORKERS))" formalize-screen --task "$task"
done
flush_jobs

job_index=0
for task in "${TASKS[@]}"; do
  for length in $(seq 2 30); do
    for restart in $(seq 0 2); do
      submit "search_${task}_L${length}_R${restart}" "$((job_index % WORKERS))" \
        search --task "$task" --length "$length" --restart "$restart"
      job_index=$((job_index + 1))
    done
  done
done
flush_jobs

job_index=0
for task in "${TASKS[@]}"; do
  for length in $(seq 2 30); do
    submit "merge_${task}_L${length}" "$((job_index % WORKERS))" \
      merge-search --task "$task" --length "$length"
    job_index=$((job_index + 1))
  done
done
flush_jobs

job_index=0
for task in "${TASKS[@]}"; do
  for length in $(seq 1 30); do
    submit "validate_${task}_L${length}" "$((job_index % WORKERS))" \
      validate --task "$task" --length "$length"
    job_index=$((job_index + 1))
  done
done
flush_jobs

run_one freeze 0 freeze
run_one test 0 test
run_one ood 1 ood
run_one retrieval 0 retrieval
run_one finalize 0 finalize

"$PYTHON" scripts/audit_v5_mode3.py --root "$ROOT" --config "$CONFIG" --results "$OUTPUT" \
  --output "$OUTPUT/final_scope_and_result_audit.json"
"$PYTHON" scripts/inventory_remote_results.py --results "$OUTPUT" \
  --output "$OUTPUT/full_inventory.json" --csv-output "$OUTPUT/full_inventory.csv"
event audit_and_inventory complete
