#!/usr/bin/env bash
set -Eeuo pipefail

V7_ROOT="${V7_ROOT:-/mnt/data/jkl/StickyToken-v7-occupancy-frontier}"
V7_PYTHON="${V7_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
V7_CONFIG="${V7_CONFIG:-configs/v7_mode3_occupancy_frontier.yaml}"
V7_OUTPUT="${V7_OUTPUT:-/mnt/data/jkl/StickyToken-v7-results/sticky_lab/sentence_t5_base/mode3_v7_occupancy_frontier}"
V7_UNIT="${V7_UNIT:-sticky-v7-occupancy-frontier.service}"

cd "$V7_ROOT"
exec "$V7_PYTHON" scripts/status_v7_mode3.py \
  --root "$V7_ROOT" --output "$V7_OUTPUT" --config "$V7_CONFIG" --unit "$V7_UNIT" "$@"
