#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${V6_3_ROOT:-/mnt/data/jkl/StickyToken-v6-3-light-formal}"
PYTHON="${V6_3_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
CONFIG="${V6_3_CONFIG:-configs/v6_3_mode3_light.yaml}"
OUTPUT="${V6_3_OUTPUT:-/mnt/data/jkl/StickyToken-v6-3-results/sticky_lab/sentence_t5_base/mode3_v6_3_light}"
GPUS="${V6_3_GPUS:-4,5,6,7}"

cd "$ROOT"
V6_3_PROFILE=formal V6_3_ROOT="$ROOT" V6_3_PYTHON="$PYTHON" V6_3_CONFIG="$CONFIG" V6_3_OUTPUT="$OUTPUT" V6_3_GPUS="$GPUS" scripts/run_v6_3_preflight.sh
"$PYTHON" scripts/run_v6_3_orchestrator.py --config "$CONFIG" --output "$OUTPUT" \
  --profile formal --mode search --gpus "$GPUS" --shards 32 --cpu-workers 8
