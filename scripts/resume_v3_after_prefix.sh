#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${STICKY_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

log_dir="results/sticky_lab/logs/v3"
mkdir -p "$log_dir"
v3="$python_bin -m sticky_lab.mode3_v3.run --config configs/v3_mode3.yaml"

run_group() {
  local label="$1"
  shift
  local pids=()
  local names=()
  while [[ "$#" -gt 0 ]]; do
    local name="$1"
    local command="$2"
    shift 2
    bash -lc "$command" >"$log_dir/${name}.log" 2>&1 &
    pids+=("$!")
    names+=("$name")
  done
  local failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "[$label] ${names[$index]} completed"
    else
      echo "[$label] ${names[$index]} failed; see $log_dir/${names[$index]}.log" >&2
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]]
}

# Only the two interrupted restart-0 prefix tasks are recomputed.  The six
# restart-1..3 tasks remain owned by the original launcher.
run_group prefix-r0-restart \
  prefix_sep_r0_bounded "$v3 --phase search --position prefix --subprotocol separator --restart 0 --device cuda:0" \
  prefix_blank_r0_bounded "$v3 --phase search --position prefix --subprotocol blank --restart 0 --device cuda:4"

# Wait for every independently running prefix task to publish its atomic phase
# summary before starting the next position wave.
while true; do
  complete=1
  for protocol in separator blank; do
    for restart in 0 1 2 3; do
      summary="results/sticky_lab/sentence_t5_base/mode3_v3/search_summary_prefix_${protocol}_$(printf '%02d' "$restart").json"
      [[ -f "$summary" ]] || complete=0
    done
  done
  [[ "$complete" -eq 1 ]] && break
  sleep 15
done

for position in suffix random universal; do
  run_group "mode3-search-${position}" \
    "${position}_sep_r0" "$v3 --phase search --position $position --subprotocol separator --restart 0 --device cuda:0" \
    "${position}_sep_r1" "$v3 --phase search --position $position --subprotocol separator --restart 1 --device cuda:1" \
    "${position}_sep_r2" "$v3 --phase search --position $position --subprotocol separator --restart 2 --device cuda:2" \
    "${position}_sep_r3" "$v3 --phase search --position $position --subprotocol separator --restart 3 --device cuda:3" \
    "${position}_blank_r0" "$v3 --phase search --position $position --subprotocol blank --restart 0 --device cuda:4" \
    "${position}_blank_r1" "$v3 --phase search --position $position --subprotocol blank --restart 1 --device cuda:5" \
    "${position}_blank_r2" "$v3 --phase search --position $position --subprotocol blank --restart 2 --device cuda:6" \
    "${position}_blank_r3" "$v3 --phase search --position $position --subprotocol blank --restart 3 --device cuda:7"
done

run_group finalize \
  mode2_finalize "$python_bin -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase finalize --device cuda:0" \
  mode3_finalize "$v3 --phase finalize --device cuda:4"

echo "Recovered V3 experiment completed after bounded prefix restart-0 recomputation."
