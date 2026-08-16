#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${V6_3_ROOT:-/mnt/data/jkl/StickyToken-v6-3-light-formal}"
cd "$ROOT"
scripts/run_v6_3_search.sh
scripts/run_v6_3_confirm.sh
scripts/run_v6_3_followups.sh
