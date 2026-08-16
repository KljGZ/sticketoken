#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${V6_2_ROOT:-/mnt/data/jkl/StickyToken-v6-2-formal}"
OUTPUT="${V6_2_OUTPUT:-/mnt/data/jkl/StickyToken-v6-2-results/sticky_lab/sentence_t5_base/mode3_v6_2}"
UNIT="${V6_2_UNIT:-sticky-v6-2}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
AUDIT="${V6_2_STOP_AUDIT:-$ROOT/audit/v6_3_stop_$STAMP}"

[[ "$ROOT" == /mnt/data/jkl/StickyToken-v6-2-formal ]] || {
  echo "refusing an unregistered V6.2 root: $ROOT" >&2; exit 64;
}
[[ "$OUTPUT" == /mnt/data/jkl/StickyToken-v6-2-results/sticky_lab/sentence_t5_base/mode3_v6_2 ]] || {
  echo "refusing an unregistered V6.2 output: $OUTPUT" >&2; exit 64;
}
mkdir -p "$AUDIT"

git -C "$ROOT" rev-parse HEAD >"$AUDIT/git_commit.txt"
git -C "$ROOT" branch --show-current >"$AUDIT/git_branch.txt"
git -C "$ROOT" status --porcelain=v2 >"$AUDIT/git_status.txt"
ps -eo user,pid,ppid,pgid,sid,lstart,cmd --sort=pid >"$AUDIT/processes_before.txt"
nvidia-smi >"$AUDIT/nvidia_smi_before.txt" 2>&1 || true
systemctl --user show "$UNIT" --no-pager >"$AUDIT/systemd_before.txt" 2>&1 || true
journalctl --user -u "$UNIT" -n 200 --no-pager >"$AUDIT/journal_before.txt" 2>&1 || true

python - "$OUTPUT" "$AUDIT/result_manifest_before.json" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); rows=[]
for path in sorted(root.rglob('*')) if root.is_dir() else []:
    if path.is_file():
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''): digest.update(block)
        rows.append({'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
pathlib.Path(sys.argv[2]).write_text(json.dumps({'files':rows,'file_count':len(rows),'total_bytes':sum(row['bytes'] for row in rows)},indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY

# Stop only the exact registered user unit. No broad process-name command is used.
systemctl --user stop "$UNIT"

mapfile -t exact_pids < <(python - "$ROOT" "$OUTPUT" <<'PY'
import os,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(); output=pathlib.Path(sys.argv[2]).resolve()
for entry in pathlib.Path('/proc').iterdir():
    if not entry.name.isdigit(): continue
    try:
        pid=int(entry.name); cmd=entry.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        cwd=entry.joinpath('cwd').resolve()
        status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in entry.joinpath('status').read_text().splitlines() if ':' in line}
        if int(status['Uid'].split()[0])!=os.getuid(): continue
        owned=('mode3_v6_2' in cmd or 'run_v6_2' in cmd) and (cwd==root or root in cwd.parents or output==cwd or output in cwd.parents or str(output) in cmd)
        if owned: print(pid)
    except (OSError,PermissionError,ValueError,KeyError): pass
PY
)
if [[ "${#exact_pids[@]}" -gt 0 ]]; then
  kill -TERM "${exact_pids[@]}"
  for _ in $(seq 1 30); do
    remaining=0
    for pid in "${exact_pids[@]}"; do kill -0 "$pid" 2>/dev/null && remaining=$((remaining+1)); done
    [[ "$remaining" -eq 0 ]] && break
    sleep 1
  done
fi

ps -eo user,pid,ppid,pgid,sid,lstart,cmd --sort=pid >"$AUDIT/processes_after.txt"
nvidia-smi >"$AUDIT/nvidia_smi_after.txt" 2>&1 || true
systemctl --user show "$UNIT" --no-pager >"$AUDIT/systemd_after.txt" 2>&1 || true
python - "$AUDIT/STOP_CURRENT_V6_2_AUDIT.json" "$ROOT" "$OUTPUT" "$UNIT" "${#exact_pids[@]}" <<'PY'
import json,pathlib,subprocess,sys,time
path,root,output,unit,count=sys.argv[1:]
show=subprocess.run(['systemctl','--user','show',unit,'--no-pager','--property=ActiveState,SubState,MainPID'],text=True,stdout=subprocess.PIPE,check=False).stdout
inactive='ActiveState=inactive' in show and 'MainPID=0' in show
value={'schema_version':'mode3-v6-3-stop-v6-2-v1','status':'STOPPED_AND_ARCHIVED' if inactive else 'INCONCLUSIVE_RUNTIME_FAILURE','root':root,'output':output,'unit':unit,'exact_remaining_processes_signalled':int(count),'old_outputs_deleted':False,'other_gpu_jobs_interfered':False,'systemd_after':show,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
pathlib.Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if not inactive: raise SystemExit(1)
PY
