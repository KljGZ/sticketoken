#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${V6_3_ROOT:-/mnt/data/jkl/StickyToken-v6-3-light-formal}"
PYTHON="${V6_3_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
CONFIG="${V6_3_CONFIG:-configs/v6_3_mode3_light.yaml}"
OUTPUT="${V6_3_OUTPUT:-/mnt/data/jkl/StickyToken-v6-3-results/sticky_lab/sentence_t5_base/mode3_v6_3_light}"
PROFILE="${V6_3_PROFILE:-formal}"
GPUS="${V6_3_GPUS:-4,5,6,7}"

IFS=',' read -r -a gpu_array <<<"$GPUS"
[[ "${#gpu_array[@]}" -ge 1 ]] || { echo "no V6.3 GPU selected" >&2; exit 64; }
for gpu in "${gpu_array[@]}"; do
  [[ "$gpu" =~ ^[4-7]$ ]] || {
    echo "V6.3 hard-forbids physical GPUs 0-3 and permits only 4-7: $GPUS" >&2
    exit 64
  }
done

cd "$ROOT"
mkdir -p "$OUTPUT/orchestration_logs"
if [[ "$PROFILE" == formal && -n "$(git status --porcelain)" ]]; then
  echo "formal V6.3 worktree is dirty" >&2
  exit 65
fi

python_log="$OUTPUT/orchestration_logs/code_and_synthetic_tests.log"
{
  "$PYTHON" -m compileall -q sticky_lab/mode3_v6_3
  "$PYTHON" -m ruff check sticky_lab/mode3_v6_3 tests/test_mode3_v6_3_*.py scripts/run_v6_3_orchestrator.py scripts/status_v6_3_mode3.py
  "$PYTHON" -m mypy sticky_lab/mode3_v6_3
  "$PYTHON" -m pytest -q -p no:cacheprovider tests/test_mode3_v6_3_*.py
} >"$python_log" 2>&1

CUDA_VISIBLE_DEVICES="" "$PYTHON" -m sticky_lab.mode3_v6_3.cli \
  --config "$CONFIG" --output "$OUTPUT" --profile "$PROFILE" prepare

"$PYTHON" - "$OUTPUT/orchestration_logs/CODE_AND_SYNTHETIC_TESTS_PASSED.json" \
  "$(git rev-parse HEAD)" "$(sha256sum "$CONFIG" | awk '{print $1}')" "$PROFILE" <<'PY'
import json,os,sys,tempfile,time
path,commit,config_sha,profile=sys.argv[1:]
value={"schema_version":"mode3-v6-3-test-gate-v1","status":"CODE_AND_SYNTHETIC_TESTS_PASSED","code_commit":commit,"source_config_sha256":config_sha,"profile":profile,"utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
directory=os.path.dirname(path); os.makedirs(directory,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix='.tests.',dir=directory)
with os.fdopen(fd,'w',encoding='utf-8') as handle:
    json.dump(value,handle,indent=2,sort_keys=True); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp,path)
PY
