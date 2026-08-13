#!/usr/bin/env bash
set -Eeuo pipefail

OUTPUT="${V6_COMPACT_OUTPUT:-/mnt/data/jkl/StickyToken-v6-compact-results/sticky_lab/sentence_t5_base/mode3_v6_compact}"
UNIT="${V6_COMPACT_UNIT:-}"
GRACE_SECONDS="${V6_COMPACT_STOP_GRACE:-30}"

if [[ -n "$UNIT" ]]; then
  systemctl --user stop "$UNIT"
fi

pid_file="$OUTPUT/.orchestrator.pid"
if [[ -s "$pid_file" ]]; then
  pid="$(tr -cd '0-9' <"$pid_file")"
  if [[ -n "$pid" && -r "/proc/$pid/cmdline" ]]; then
    command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
    if [[ "$command_line" != *run_v6_mode3_compact_remote.sh* && "$command_line" != *sticky_lab.mode3_v6_compact* ]]; then
      echo "refusing to stop unrelated PID $pid: $command_line" >&2
      exit 64
    fi
    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
    kill -TERM -- "-$pgid"
    for ((index=0; index<GRACE_SECONDS; index++)); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pgid"
    fi
  fi
fi

matches="$(python3 - <<'PY'
import os,pathlib
me=os.getpid(); rows=[]
for item in pathlib.Path('/proc').iterdir():
    if not item.name.isdigit() or int(item.name)==me: continue
    try:
        cmd=(item/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        status=(item/'status').read_text(errors='replace')
        uid=int(next(line.split(':',1)[1].split()[0] for line in status.splitlines() if line.startswith('Uid:')))
    except Exception: continue
    if uid==os.getuid() and ('mode3_v6_compact' in cmd or 'run_v6_mode3_compact_remote.sh' in cmd):
        rows.append(f'{item.name}\t{cmd}')
print('\n'.join(rows))
PY
)"
if [[ -n "$matches" ]]; then
  echo "Compact processes remain after exact stop:" >&2
  echo "$matches" >&2
  exit 1
fi
echo "V6 Compact stopped; no exact process matches remain."
