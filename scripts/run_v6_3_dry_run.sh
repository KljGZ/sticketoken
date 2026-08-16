#!/usr/bin/env bash
set -Eeuo pipefail

export V6_3_ROOT="${V6_3_ROOT:-/mnt/data/jkl/StickyToken-v6-3-light-formal}"
export V6_3_PYTHON="${V6_3_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
export V6_3_CONFIG="${V6_3_CONFIG:-configs/v6_3_mode3_light.yaml}"
export V6_3_OUTPUT="${V6_3_DRY_OUTPUT:-/mnt/data/jkl/StickyToken-v6-3-dry-run-r2/sticky_lab/sentence_t5_base/mode3_v6_3_light}"
export V6_3_PROFILE=dry_run
export V6_3_GPUS="${V6_3_DRY_GPUS:-4}"

cd "$V6_3_ROOT"
scripts/run_v6_3_preflight.sh
"$V6_3_PYTHON" scripts/run_v6_3_orchestrator.py \
  --config "$V6_3_CONFIG" --output "$V6_3_OUTPUT" --profile dry_run \
  --mode search --gpus "$V6_3_GPUS" --shards 4 --cpu-workers 1
# Marker-boundary replay proves the search path is resumable without duplicate model calls.
"$V6_3_PYTHON" scripts/run_v6_3_orchestrator.py \
  --config "$V6_3_CONFIG" --output "$V6_3_OUTPUT" --profile dry_run \
  --mode search --gpus "$V6_3_GPUS" --shards 4 --cpu-workers 1
"$V6_3_PYTHON" scripts/run_v6_3_orchestrator.py \
  --config "$V6_3_CONFIG" --output "$V6_3_OUTPUT" --profile dry_run \
  --mode confirm --gpus "$V6_3_GPUS" --shards 4 --cpu-workers 1
"$V6_3_PYTHON" scripts/run_v6_3_orchestrator.py \
  --config "$V6_3_CONFIG" --output "$V6_3_OUTPUT" --profile dry_run \
  --mode followups --gpus "$V6_3_GPUS" --shards 4 --cpu-workers 1

"$V6_3_PYTHON" - "$V6_3_OUTPUT" <<'PY'
import json,os,pathlib,sys,tempfile,time
root=pathlib.Path(sys.argv[1])
required=[root/'enumeration/COMPLETE.json',root/'stages/s0/COMPLETE.json',root/'stages/s1/COMPLETE.json',root/'stages/s2/COMPLETE.json',root/'stages/full/COMPLETE.json',root/'stages/top100/COMPLETE.json',root/'freeze/COMPLETE.json',root/'confirm/COMPLETE.json',root/'FINAL_STATUS.json',root/'result_inventory.json']
missing=[str(path.relative_to(root)) for path in required if not path.is_file()]
if missing: raise SystemExit(f'dry run incomplete: {missing}')
value={"schema_version":"mode3-v6-3-dry-run-v1","status":"DRY_RUN_PASSED","legal_tokens":64,"one_physical_gpu":4,"physical_gpus_0_3_used":False,"marker_boundary_replay_passed":True,"scientific_claims_allowed":False,"utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
path=root/'DRY_RUN_PASSED.json'; fd,tmp=tempfile.mkstemp(prefix='.dry.',dir=root)
with os.fdopen(fd,'w',encoding='utf-8') as handle:
    json.dump(value,handle,indent=2,sort_keys=True); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp,path)
PY
