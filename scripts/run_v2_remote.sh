#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${STICKY_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

log_dir="results/sticky_lab/logs/v2"
mkdir -p "$log_dir"

gpu_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$gpu_count" -lt 8 ]]; then
  echo "V2 registered launcher requires all eight CUDA devices; found $gpu_count" >&2
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
  if [[ "$failed" -ne 0 ]]; then
    exit 1
  fi
}

run_group prepare \
  mode1_full "$python_bin -m sticky_lab.v2 --config configs/v2_single_sticky.yaml --phase full --device cuda:2" \
  mode2_prepare "$python_bin -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase prepare --device cuda:0" \
  mode3_prepare "$python_bin -m sticky_lab.v2 --config configs/v2_repulsive_attractor.yaml --phase prepare --device cuda:4"

run_group search \
  mode2_restart0 "$python_bin -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase search --restart 0 --device cuda:0" \
  mode2_restart1 "$python_bin -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase search --restart 1 --device cuda:1" \
  mode2_restart2 "$python_bin -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase search --restart 2 --device cuda:2" \
  mode2_restart3 "$python_bin -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase search --restart 3 --device cuda:3" \
  mode3_restart0 "$python_bin -m sticky_lab.v2 --config configs/v2_repulsive_attractor.yaml --phase search --restart 0 --device cuda:4" \
  mode3_restart1 "$python_bin -m sticky_lab.v2 --config configs/v2_repulsive_attractor.yaml --phase search --restart 1 --device cuda:5" \
  mode3_restart2 "$python_bin -m sticky_lab.v2 --config configs/v2_repulsive_attractor.yaml --phase search --restart 2 --device cuda:6" \
  mode3_restart3 "$python_bin -m sticky_lab.v2 --config configs/v2_repulsive_attractor.yaml --phase search --restart 3 --device cuda:7"

run_group finalize \
  mode2_finalize "$python_bin -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase finalize --device cuda:0" \
  mode3_finalize "$python_bin -m sticky_lab.v2 --config configs/v2_repulsive_attractor.yaml --phase finalize --device cuda:4"

echo "Sticky / Attractor V2 registered experiment completed."

