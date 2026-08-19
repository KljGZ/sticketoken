#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${V6_3_ROOT:-/mnt/data/jkl/StickyToken-v6-3-light-formal}"
PYTHON="${V6_3_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
CONFIG="${V6_3_CONFIG:-configs/v6_3_mode3_light.yaml}"
OUTPUT="${V6_3_OUTPUT:-/mnt/data/jkl/StickyToken-v6-3-results/sticky_lab/sentence_t5_base/mode3_v6_3_light}"
PROFILE="${V6_3_PROFILE:-formal}"
GPUS="${V6_3_GPUS:-4,5,6,7}"
RUFF="${V6_3_RUFF:-$(command -v ruff || true)}"
MYPY="${V6_3_MYPY:-$(command -v mypy || true)}"

[[ -n "$RUFF" && -x "$RUFF" ]] || { echo "ruff executable not found" >&2; exit 69; }
[[ -n "$MYPY" && -x "$MYPY" ]] || { echo "mypy executable not found" >&2; exit 69; }

IFS=',' read -r -a gpu_array <<<"$GPUS"
[[ "${#gpu_array[@]}" -ge 1 ]] || { echo "no V6.3 GPU selected" >&2; exit 64; }

cd "$ROOT"
"$PYTHON" - "$CONFIG" "$GPUS" <<'PY'
import pathlib
import sys

from sticky_lab.mode3_v6_3.config import load_config

config = load_config(pathlib.Path(sys.argv[1]))
requested = [int(value) for value in sys.argv[2].split(",") if value.strip()]
allowed = set(map(int, config["resources"]["allowed_physical_gpus"]))
forbidden = set(map(int, config["resources"]["forbidden_physical_gpus"]))
if not requested or len(requested) != len(set(requested)):
    raise SystemExit(f"invalid GPU list: {requested}")
if any(gpu not in allowed or gpu in forbidden for gpu in requested):
    raise SystemExit(
        f"GPU policy mismatch: requested={requested} "
        f"allowed={sorted(allowed)} forbidden={sorted(forbidden)}"
    )
PY

mkdir -p "$OUTPUT/orchestration_logs"
if [[ "$PROFILE" == formal && -n "$(git status --porcelain)" ]]; then
  echo "formal V6.3 worktree is dirty" >&2
  exit 65
fi

python_log="$OUTPUT/orchestration_logs/code_and_synthetic_tests.log"
{
  "$PYTHON" -m compileall -q sticky_lab/mode3_v6_3
  "$RUFF" check sticky_lab/mode3_v6_3 tests/test_mode3_v6_3_*.py scripts/run_v6_3_orchestrator.py scripts/status_v6_3_mode3.py
  "$MYPY" sticky_lab/mode3_v6_3
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
