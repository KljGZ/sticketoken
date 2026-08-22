#!/usr/bin/env bash
set -Eeuo pipefail

V7_ROOT="${V7_ROOT:-/mnt/data/jkl/StickyToken-v7-occupancy-frontier}"
V7_UNIT="${V7_UNIT:-sticky-v7-occupancy-frontier.service}"
V7_USER_CONFIG="${XDG_CONFIG_HOME:-${HOME}/.config}"
unit_source="$V7_ROOT/deploy/$V7_UNIT"
unit_target="$V7_USER_CONFIG/systemd/user/$V7_UNIT"

[[ -f "$unit_source" ]] || { echo "V7 service template not found: $unit_source" >&2; exit 66; }
mkdir -p "$(dirname "$unit_target")"
install -m 0644 "$unit_source" "$unit_target"
systemctl --user daemon-reload
systemctl --user enable --now "$V7_UNIT"
systemctl --user show "$V7_UNIT" --no-pager \
  --property=ActiveState,SubState,MainPID,ExecMainStatus,Result,ExecStart
