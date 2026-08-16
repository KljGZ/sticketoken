#!/usr/bin/env bash
set -Eeuo pipefail

export V6_3_ROOT="${V6_3_ROOT:-/mnt/data/jkl/StickyToken-v6-3-light-formal}"
export V6_3_PYTHON="${V6_3_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
export V6_3_CONFIG="${V6_3_CONFIG:-configs/v6_3_mode3_light.yaml}"
export V6_3_OUTPUT="${V6_3_PILOT_OUTPUT:-/mnt/data/jkl/StickyToken-v6-3-pilot-r4/sticky_lab/sentence_t5_base/mode3_v6_3_light}"
export V6_3_PROFILE=pilot
export V6_3_GPUS="${V6_3_PILOT_GPUS:-4,5,6,7}"

cd "$V6_3_ROOT"
scripts/run_v6_3_preflight.sh
"$V6_3_PYTHON" scripts/run_v6_3_orchestrator.py \
  --config "$V6_3_CONFIG" --output "$V6_3_OUTPUT" --profile pilot \
  --mode all --gpus "$V6_3_GPUS" --shards 8 --cpu-workers 1
"$V6_3_PYTHON" - "$V6_3_OUTPUT" "$V6_3_GPUS" <<'PY'
import json,os,pathlib,sys,tempfile,time
root=pathlib.Path(sys.argv[1])
gpus=[int(value) for value in sys.argv[2].split(',') if value]
if not gpus or any(gpu not in range(4, 8) for gpu in gpus): raise SystemExit(f'invalid pilot GPU binding: {gpus}')
required=[root/'enumeration/COMPLETE.json',root/'stages/top100/COMPLETE.json',root/'freeze/COMPLETE.json',root/'confirm/COMPLETE.json',root/'FINAL_STATUS.json',root/'result_inventory.json']
missing=[str(path.relative_to(root)) for path in required if not path.is_file()]
if missing: raise SystemExit(f'pilot incomplete: {missing}')
value={"schema_version":"mode3-v6-3-pilot-v1","status":"PILOT_PASSED","legal_tokens":512,"authorized_physical_gpus":gpus,"scientific_claims_allowed":False,"physical_gpus_0_3_used":False,"utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
path=root/'PILOT_PASSED.json'; fd,tmp=tempfile.mkstemp(prefix='.pilot.',dir=root)
with os.fdopen(fd,'w',encoding='utf-8') as handle:
    json.dump(value,handle,indent=2,sort_keys=True); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp,path)
PY
