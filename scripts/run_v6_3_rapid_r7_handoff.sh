#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${V6_3_ROOT:-/mnt/data/jkl/StickyToken-v6-3-rapid-r7}"
OUTPUT="${V6_3_OUTPUT:-/mnt/data/jkl/StickyToken-v6-3-results/sticky_lab/sentence_t5_base/mode3_v6_3_rapid_r7}"
PREFLIGHT_PID="${V6_3_PREFLIGHT_PID:-}"

[[ "$PREFLIGHT_PID" =~ ^[1-9][0-9]*$ ]] || {
  echo "V6_3_PREFLIGHT_PID must identify the registered r7 preflight" >&2
  exit 64
}

while [[ ! -f "$OUTPUT/registration/COMPLETE.json" ]]; do
  command_line="$(ps -p "$PREFLIGHT_PID" -o args= 2>/dev/null || true)"
  [[ "$command_line" == *"sticky_lab.mode3_v6_3.cli"* ]] || {
    echo "r7 preflight ended before registration completed" >&2
    exit 75
  }
  [[ "$command_line" == *"--output $OUTPUT --profile formal prepare"* ]] || {
    echo "r7 preflight PID identity drift" >&2
    exit 76
  }
  sleep 15
done

cd "$ROOT"
exec scripts/run_v6_3_rapid_r7_remote.sh
