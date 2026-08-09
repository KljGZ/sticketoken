#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${STICKY_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

log_dir="results/sticky_lab/logs/v3/finalize_shards"
mkdir -p "$log_dir"
v3=("$python_bin" -m sticky_lab.mode3_v3.run --config configs/v3_mode3.yaml --phase finalize)

declare -a names=(
  suffix_blank
  random_separator
  random_blank
  universal_separator
  universal_blank
)
declare -a positions=(suffix random random universal universal)
declare -a protocols=(blank separator blank separator blank)
declare -a devices=(cuda:0 cuda:4 cuda:5 cuda:6 cuda:7)
declare -a pids=()

for index in "${!names[@]}"; do
  "${v3[@]}" \
    --position "${positions[$index]}" \
    --subprotocol "${protocols[$index]}" \
    --device "${devices[$index]}" \
    >"$log_dir/${names[$index]}.log" 2>&1 &
  pids+=("$!")
  echo "[finalize-shards] started ${names[$index]} pid=${pids[$index]} device=${devices[$index]}"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[finalize-shards] ${names[$index]} completed"
  else
    echo "[finalize-shards] ${names[$index]} failed; see $log_dir/${names[$index]}.log" >&2
    failed=1
  fi
done

exit "$failed"
