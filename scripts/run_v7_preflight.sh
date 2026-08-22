#!/usr/bin/env bash
set -Eeuo pipefail

V7_ROOT="${V7_ROOT:-/mnt/data/jkl/StickyToken-v7-occupancy-frontier}"
V7_PYTHON="${V7_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
V7_CONFIG="${V7_CONFIG:-configs/v7_mode3_occupancy_frontier.yaml}"
V7_OUTPUT="${V7_OUTPUT:-/mnt/data/jkl/StickyToken-v7-results/sticky_lab/sentence_t5_base/mode3_v7_occupancy_frontier_r3_priority}"
V7_PROFILE="${V7_PROFILE:-formal}"
V7_GPUS="${V7_GPUS:-4,5,6,7}"
V7_RUFF="${V7_RUFF:-$(command -v ruff || true)}"
V7_MYPY="${V7_MYPY:-$(command -v mypy || true)}"
V7_ENV_BIN="$(dirname "$V7_PYTHON")"
[[ -x "$V7_RUFF" ]] || V7_RUFF="$V7_ENV_BIN/ruff"
[[ -x "$V7_MYPY" ]] || V7_MYPY="$V7_ENV_BIN/mypy"

[[ -x "$V7_PYTHON" ]] || { echo "V7 Python executable not found: $V7_PYTHON" >&2; exit 69; }
[[ -n "$V7_RUFF" && -x "$V7_RUFF" ]] || { echo "ruff executable not found" >&2; exit 69; }
[[ -n "$V7_MYPY" && -x "$V7_MYPY" ]] || { echo "mypy executable not found" >&2; exit 69; }

cd "$V7_ROOT"
"$V7_PYTHON" - "$V7_CONFIG" "$V7_OUTPUT" "$V7_PROFILE" "$V7_GPUS" <<'PY'
import pathlib
import sys

from sticky_lab.mode3_v7.config import assert_output_leaf, config_for_profile, load_config

config_path, output, profile, raw_gpus = sys.argv[1:]
config = config_for_profile(load_config(pathlib.Path(config_path)), profile)
assert_output_leaf(pathlib.Path(output), config)
requested = [int(value) for value in raw_gpus.split(",") if value.strip()]
allowed = list(map(int, config["resources"]["allowed_physical_gpus"]))
forbidden = set(map(int, config["resources"]["forbidden_physical_gpus"]))
if profile == "formal" and requested != allowed:
    raise SystemExit(f"formal V7 requires exact GPU order {allowed}, observed {requested}")
if not requested or len(requested) != len(set(requested)):
    raise SystemExit(f"invalid V7 GPU list: {requested}")
if not set(requested).issubset(set(allowed)) or set(requested).intersection(forbidden):
    raise SystemExit(
        f"V7 GPU policy mismatch: requested={requested} "
        f"allowed={allowed} forbidden={sorted(forbidden)}"
    )
source = pathlib.Path(config["reuse"]["source_output"])
if not source.is_dir():
    raise SystemExit(f"V6.3 r5 reuse source is unavailable: {source}")
PY

if [[ "$V7_PROFILE" == formal && -n "$(git status --porcelain)" ]]; then
  echo "formal V7 worktree is dirty" >&2
  exit 65
fi

mkdir -p "$V7_OUTPUT/orchestration_logs"
test_log="$V7_OUTPUT/orchestration_logs/code_and_synthetic_tests.log"
{
  "$V7_PYTHON" -m compileall -q sticky_lab/mode3_v7 scripts/run_v7_orchestrator.py scripts/status_v7_mode3.py
  "$V7_RUFF" check sticky_lab/mode3_v7 tests/test_mode3_v7_*.py scripts/run_v7_orchestrator.py scripts/status_v7_mode3.py
  "$V7_MYPY" sticky_lab/mode3_v7
  "$V7_PYTHON" -m pytest -q -p no:cacheprovider tests/test_mode3_v7_*.py
} >"$test_log" 2>&1

"$V7_PYTHON" -m sticky_lab.mode3_v7.priority audit \
  --config "$V7_CONFIG" --output "$V7_OUTPUT"

CUDA_VISIBLE_DEVICES="" "$V7_PYTHON" -m sticky_lab.mode3_v7.cli \
  --config "$V7_CONFIG" --output "$V7_OUTPUT" --profile "$V7_PROFILE" prepare

"$V7_PYTHON" - "$V7_OUTPUT/orchestration_logs/CODE_AND_SYNTHETIC_TESTS_PASSED.json" \
  "$(git rev-parse HEAD)" "$(sha256sum "$V7_CONFIG" | awk '{print $1}')" "$V7_PROFILE" <<'PY'
import json
import os
import sys
import tempfile
import time

path, commit, config_sha, profile = sys.argv[1:]
value = {
    "schema_version": "mode3-v7-test-gate-v1",
    "status": "CODE_AND_SYNTHETIC_TESTS_PASSED",
    "code_commit": commit,
    "source_config_sha256": config_sha,
    "profile": profile,
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
directory = os.path.dirname(path)
os.makedirs(directory, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".tests.", dir=directory)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
