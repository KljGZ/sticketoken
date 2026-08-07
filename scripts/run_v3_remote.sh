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

gpu_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$gpu_count" -lt 8 ]]; then
  echo "V3 registered launcher requires all eight CUDA devices; found $gpu_count" >&2
  exit 1
fi

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

v3="$python_bin -m sticky_lab.mode3_v3.run --config configs/v3_mode3.yaml"
v2="$python_bin -m sticky_lab.v2 --config configs/v2_multi_booster.yaml"

$v3 --phase prepare-common --device cuda:0 >"$log_dir/mode3_prepare_common.log" 2>&1

run_group mode3-screen \
  screen0 "$v3 --phase screen-shard --shard-index 0 --shard-count 8 --device cuda:0" \
  screen1 "$v3 --phase screen-shard --shard-index 1 --shard-count 8 --device cuda:1" \
  screen2 "$v3 --phase screen-shard --shard-index 2 --shard-count 8 --device cuda:2" \
  screen3 "$v3 --phase screen-shard --shard-index 3 --shard-count 8 --device cuda:3" \
  screen4 "$v3 --phase screen-shard --shard-index 4 --shard-count 8 --device cuda:4" \
  screen5 "$v3 --phase screen-shard --shard-index 5 --shard-count 8 --device cuda:5" \
  screen6 "$v3 --phase screen-shard --shard-index 6 --shard-count 8 --device cuda:6" \
  screen7 "$v3 --phase screen-shard --shard-index 7 --shard-count 8 --device cuda:7"

$v3 --phase merge-prepare --shard-count 8 >"$log_dir/mode3_merge_prepare.log" 2>&1

# Re-run Mode 2 from its frozen V2 screen/pool while V3 establishes the
# continuous feasibility upper bound.  Both retain lengths 2..30, step 2.
run_group mode2-and-soft-prompt \
  mode2_r0 "$v2 --phase search --restart 0 --device cuda:0" \
  mode2_r1 "$v2 --phase search --restart 1 --device cuda:1" \
  mode2_r2 "$v2 --phase search --restart 2 --device cuda:2" \
  mode2_r3 "$v2 --phase search --restart 3 --device cuda:3" \
  soft_prefix_sep "$v3 --phase soft-prompt --position prefix --subprotocol separator --device cuda:4" \
  soft_prefix_blank "$v3 --phase soft-prompt --position prefix --subprotocol blank --device cuda:5" \
  soft_suffix_sep "$v3 --phase soft-prompt --position suffix --subprotocol separator --device cuda:6" \
  soft_suffix_blank "$v3 --phase soft-prompt --position suffix --subprotocol blank --device cuda:7"

for position in prefix suffix random universal; do
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
  mode2_finalize "$v2 --phase finalize --device cuda:0" \
  mode3_finalize "$v3 --phase finalize --device cuda:4"

echo "Mode 2 V2 rerun and Mode 3 V3 registered experiment completed."
