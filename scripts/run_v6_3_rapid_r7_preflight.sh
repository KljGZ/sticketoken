#!/usr/bin/env bash
set -Eeuo pipefail

export V6_3_ROOT="${V6_3_ROOT:-/mnt/data/jkl/StickyToken-v6-3-rapid-r7}"
export V6_3_PYTHON="${V6_3_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
export V6_3_CONFIG="${V6_3_CONFIG:-configs/v6_3_mode3_rapid_r7.yaml}"
export V6_3_OUTPUT="${V6_3_OUTPUT:-/mnt/data/jkl/StickyToken-v6-3-results/sticky_lab/sentence_t5_base/mode3_v6_3_rapid_r7}"
export V6_3_PROFILE="${V6_3_PROFILE:-formal}"
export V6_3_GPUS="${V6_3_GPUS:-0,1,2,3,4,5,6,7}"

cd "$V6_3_ROOT"
scripts/run_v6_3_preflight.sh
