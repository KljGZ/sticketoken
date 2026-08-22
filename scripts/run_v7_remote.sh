#!/usr/bin/env bash
set -Eeuo pipefail

export V7_ROOT="${V7_ROOT:-/mnt/data/jkl/StickyToken-v7-occupancy-frontier}"
export V7_PYTHON="${V7_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
export V7_CONFIG="${V7_CONFIG:-configs/v7_mode3_occupancy_frontier.yaml}"
export V7_OUTPUT="${V7_OUTPUT:-/mnt/data/jkl/StickyToken-v7-results/sticky_lab/sentence_t5_base/mode3_v7_occupancy_frontier_r2_10g}"
export V7_PROFILE="${V7_PROFILE:-formal}"
export V7_GPUS="${V7_GPUS:-4,5,6,7}"

cd "$V7_ROOT"
exec "$V7_PYTHON" scripts/run_v7_orchestrator.py \
  --config "$V7_CONFIG" --output "$V7_OUTPUT" --profile "$V7_PROFILE" \
  --python "$V7_PYTHON" --gpus "$V7_GPUS"
