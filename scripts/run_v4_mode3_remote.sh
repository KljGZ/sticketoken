#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jkl/StickyToken}"
PYTHON_BIN="${PYTHON_BIN:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/v4_mode3.yaml}"
GPU_COUNT="${GPU_COUNT:-8}"
LOG_ROOT="${LOG_ROOT:-results/sticky_lab/sentence_t5_base/mode3_v4/remote_logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/sticky_lab/sentence_t5_base/mode3_v4}"

cd "$REPO_ROOT"
mkdir -p "$LOG_ROOT"

run_job() {
  local slot="$1"
  shift
  local gpu=$((slot % GPU_COUNT))
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m sticky_lab.mode3_v4.run "$@" --config "$CONFIG_PATH" --device cuda:0
}
export -f run_job
export PYTHON_BIN CONFIG_PATH GPU_COUNT

if [[ ! -f "$OUTPUT_ROOT/support/support_model.npz" ]]; then
  run_job 0 prepare >"$LOG_ROOT/prepare.log" 2>&1
fi

screen_jobs="$LOG_ROOT/screen_jobs.txt"
: >"$screen_jobs"
slot=0
for task in prefix suffix random universal; do
  for shard in $(seq 0 7); do
    if [[ ! -f "$OUTPUT_ROOT/screen/$task/shard_$(printf '%02d' "$shard").csv" ]]; then
      printf '%s\t%s\t%s\n' "$slot" "$task" "$shard" >>"$screen_jobs"
    fi
    slot=$((slot + 1))
  done
done
if [[ -s "$screen_jobs" ]]; then
  xargs -P "$GPU_COUNT" -n 3 bash -c 'run_job "$0" screen --task "$1" --shard "$2" >"'"$LOG_ROOT"'/screen_${1}_${2}.log" 2>&1' <"$screen_jobs"
fi
"$PYTHON_BIN" -m sticky_lab.mode3_v4.run merge-screen --config "$CONFIG_PATH" >"$LOG_ROOT/merge_screen.log" 2>&1

search_jobs="$LOG_ROOT/search_jobs.txt"
: >"$search_jobs"
slot=0
for task in prefix suffix random universal; do
  for length in $(seq 2 30); do
    for restart in 0 1; do
      if [[ ! -f "$OUTPUT_ROOT/search/$task/length_$(printf '%02d' "$length")/restart_$(printf '%02d' "$restart")_archive.csv" ]]; then
        printf '%s\t%s\t%s\t%s\n' "$slot" "$task" "$length" "$restart" >>"$search_jobs"
      fi
      slot=$((slot + 1))
    done
  done
done
if [[ -s "$search_jobs" ]]; then
  xargs -P "$GPU_COUNT" -n 4 bash -c 'run_job "$0" search --task "$1" --length "$2" --restart "$3" >"'"$LOG_ROOT"'/search_${1}_L${2}_R${3}.log" 2>&1' <"$search_jobs"
fi

validation_jobs="$LOG_ROOT/validation_jobs.txt"
: >"$validation_jobs"
slot=0
for task in prefix suffix random universal; do
  for length in $(seq 1 30); do
    if [[ ! -f "$OUTPUT_ROOT/validation/$task/length_$(printf '%02d' "$length")/summary.json" ]]; then
      printf '%s\t%s\t%s\n' "$slot" "$task" "$length" >>"$validation_jobs"
    fi
    slot=$((slot + 1))
  done
done
if [[ -s "$validation_jobs" ]]; then
  xargs -P "$GPU_COUNT" -n 3 bash -c 'run_job "$0" validate --task "$1" --length "$2" >"'"$LOG_ROOT"'/validate_${1}_L${2}.log" 2>&1' <"$validation_jobs"
fi

if [[ ! -f "$OUTPUT_ROOT/final_status.json" ]]; then
  run_job 0 finalize >"$LOG_ROOT/finalize.log" 2>&1
fi
"$PYTHON_BIN" scripts/audit_v4_mode3.py --config "$CONFIG_PATH" --results results/sticky_lab/sentence_t5_base/mode3_v4 >"$LOG_ROOT/audit.log" 2>&1
