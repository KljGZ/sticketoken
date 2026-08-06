#!/usr/bin/env bash
# Reproducible remote launcher: modes 1 and 2 run concurrently; mode 3 starts
# only after mode 2 has produced the registered full-vocabulary component pool.

set -uo pipefail

repo_root="${1:-/home/jkl/StickyToken}"
python_bin="${STICKY_LAB_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
log_dir="${repo_root}/results/sticky_lab/logs"
mkdir -p "${log_dir}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "${repo_root}" || exit 2

"${python_bin}" -u -m sticky_lab.run \
  --config configs/single_sticky.yaml \
  >"${log_dir}/single_sticky_v1.log" 2>&1 &
single_pid=$!

"${python_bin}" -u -m sticky_lab.run \
  --config configs/multi_booster.yaml \
  >"${log_dir}/multi_booster_v1.log" 2>&1 &
booster_pid=$!

wait "${single_pid}"
single_status=$?
wait "${booster_pid}"
booster_status=$?

repulsive_status=99
if [[ "${booster_status}" -eq 0 ]]; then
  "${python_bin}" -u -m sticky_lab.run \
    --config configs/repulsive_attractor.yaml \
    >"${log_dir}/repulsive_attractor_v1.log" 2>&1
  repulsive_status=$?
fi

printf 'single_sticky=%s\nmulti_booster=%s\nrepulsive_attractor=%s\n' \
  "${single_status}" "${booster_status}" "${repulsive_status}" \
  >"${log_dir}/three_mode_status.txt"

if [[ "${single_status}" -ne 0 || "${booster_status}" -ne 0 || "${repulsive_status}" -ne 0 ]]; then
  exit 1
fi
