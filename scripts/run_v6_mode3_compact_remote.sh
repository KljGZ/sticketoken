#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${V6_COMPACT_ROOT:-/home/jkl/StickyToken-v6-compact}"
PYTHON="${V6_COMPACT_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
CONFIG="${V6_COMPACT_CONFIG:-configs/v6_mode3_compact.yaml}"
OUTPUT="${V6_COMPACT_OUTPUT:-/mnt/data/jkl/StickyToken-v6-compact-results/sticky_lab/sentence_t5_base/mode3_v6_compact}"
WORKERS="${V6_COMPACT_WORKERS:-8}"
SHARDS="${V6_COMPACT_SHARDS:-32}"
V5_HISTORY="${V6_COMPACT_V5_HISTORY:-/mnt/data/jkl/StickyToken-v6-prerequisites/346feae/v5_single_token_history.json}"
LOGS="$OUTPUT/orchestration_logs"
export NLTK_DATA="${V6_COMPACT_NLTK_DATA:-/mnt/data/jkl/StickyToken-v6-resources/nltk_data}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

cd "$ROOT"
mkdir -p "$LOGS"
exec 9>"$OUTPUT/.orchestrator.lock"
if ! flock -n 9; then
  echo "another Compact orchestrator owns $OUTPUT" >&2
  exit 75
fi
RUN_COMMIT="$(git rev-parse HEAD)"
CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print $1}')"
printf '%s\n' "$$" >"$OUTPUT/.orchestrator.pid"

atomic_status() {
  local state="$1" code="$2" stage="$3"
  "$PYTHON" - "$LOGS/status.json" "$state" "$code" "$stage" "$RUN_COMMIT" "$CONFIG_SHA" "$$" <<'PY'
import json,os,sys,tempfile,time
path,state,code,stage,commit,config,pid=sys.argv[1:]
value={"schema_version":"mode3-v6-compact-orchestrator-status-v1","state":state,"exit_code":int(code),"stage":stage,"run_commit":commit,"config_sha256":config,"pid":int(pid),"updated_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
os.makedirs(os.path.dirname(path),exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix='.status.',dir=os.path.dirname(path))
with os.fdopen(fd,'w',encoding='utf-8') as handle:
    json.dump(value,handle,indent=2,sort_keys=True); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp,path)
PY
}
STAGE=starting
on_exit() {
  local code="$?"
  trap - EXIT
  if [[ "$code" -eq 0 ]]; then atomic_status complete "$code" "$STAGE"; else atomic_status stopped "$code" "$STAGE"; fi
  exit "$code"
}
trap on_exit EXIT
atomic_status running 0 "$STAGE"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Compact worktree is dirty; refusing formal execution" >&2
  exit 1
fi

STAGE=tests
atomic_status running 0 "$STAGE"
"$PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_mode3_v6_compact_core.py \
  tests/test_mode3_v6_compact_data.py \
  tests/test_mode3_v6_compact_scope.py \
  tests/test_mode3_v6_compact_pipeline.py >"$LOGS/tests.log" 2>&1
"$PYTHON" scripts/audit_v6_mode3_compact.py --config "$CONFIG" --output "$LOGS/scope_audit.json"
"$PYTHON" scripts/budget_v6_mode3_compact.py --config "$CONFIG" --output "$OUTPUT/budget/planned.json" >"$LOGS/budget.log"

STAGE=prepare
atomic_status running 0 "$STAGE"
"$PYTHON" -m sticky_lab.mode3_v6_compact.run --config "$CONFIG" --output "$OUTPUT" preflight
"$PYTHON" -m sticky_lab.mode3_v6_compact.run --config "$CONFIG" --output "$OUTPUT" prepare

if [[ ! -s "$OUTPUT/enumeration/COMPLETE.json" ]]; then
  STAGE=enumeration
  atomic_status running 0 "$STAGE"
  "$PYTHON" -m sticky_lab.mode3_v6_compact.workers --config "$CONFIG" --output "$OUTPUT" enumerate-vocab
fi

run_parallel() {
  local -a commands=("$@") pids=()
  local command pid failed=0
  for command in "${commands[@]}"; do
    bash -c "$command" & pids+=("$!")
    if [[ "${#pids[@]}" -ge "$WORKERS" ]]; then
      for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
      pids=()
      [[ "$failed" -eq 0 ]] || return 1
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}

STAGE=base_embeddings
atomic_status running 0 "$STAGE"
mapfile -t roles < <("$PYTHON" - "$CONFIG" <<'PY'
import sys,yaml
c=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
roles=list(c['data']['roles'])
for index in range(int(c['data']['ood_domains'])):
    roles.extend([f'ood_{index}_trigger',f'ood_{index}_benign'])
print('\n'.join(roles))
PY
)
commands=()
for index in "${!roles[@]}"; do
  role="${roles[$index]}"
  [[ -s "$OUTPUT/base_embeddings/$role.npy" && -s "$OUTPUT/base_embeddings/$role.json" ]] && continue
  gpu=$((index % WORKERS))
  commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6_compact.workers --config '$CONFIG' --output '$OUTPUT' --device cuda:0 precompute-role --role '$role' >'$LOGS/base_${role}.log' 2>&1")
done
[[ "${#commands[@]}" -eq 0 ]] || run_parallel "${commands[@]}"
"$PYTHON" - "$OUTPUT/base_embeddings/COMPLETE.json" "${#roles[@]}" <<'PY'
import json,os,sys,tempfile
p,n=sys.argv[1],int(sys.argv[2]); d=os.path.dirname(p); os.makedirs(d,exist_ok=True)
manifests=list(__import__('pathlib').Path(d).glob('*.json'))
if len(manifests)!=n: raise SystemExit(f'base embedding manifest count {len(manifests)}/{n}')
fd,t=tempfile.mkstemp(dir=d,prefix='.complete.'); os.close(fd)
open(t,'w').write(json.dumps({'roles':n,'encoded_once':True,'reused_across_shards':True},sort_keys=True,indent=2)+'\n'); os.replace(t,p)
PY

STAGE=tracks
atomic_status running 0 "$STAGE"
commands=()
if [[ ! -s "$OUTPUT/tracks/whitebox/COMPLETE.json" ]]; then
  commands+=("CUDA_VISIBLE_DEVICES=0 '$PYTHON' -m sticky_lab.mode3_v6_compact.track_whitebox --config '$CONFIG' --output '$OUTPUT' --device cuda:0 >'$LOGS/whitebox.log' 2>&1")
fi
for restart in $(seq 0 7); do
  [[ -s "$OUTPUT/tracks/blackbox/restart_$(printf '%02d' "$restart")/COMPLETE.json" ]] && continue
  gpu=$((restart % WORKERS))
  commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6_compact.track_blackbox --config '$CONFIG' --output '$OUTPUT' --device cuda:0 --restart-offset '$restart' --restarts 1 >'$LOGS/blackbox_${restart}.log' 2>&1")
done
[[ "${#commands[@]}" -eq 0 ]] || run_parallel "${commands[@]}"
if [[ ! -s "$OUTPUT/tracks/blackbox/COMPLETE.json" ]]; then
  "$PYTHON" -m sticky_lab.mode3_v6_compact.track_blackbox --config "$CONFIG" --output "$OUTPUT" --merge-only
fi

STAGE=s0
atomic_status running 0 "$STAGE"
commands=()
for shard in $(seq 0 $((SHARDS-1))); do
  target="$OUTPUT/s0/shard_$(printf '%02d' "$shard")/COMPLETE.json"
  [[ -s "$target" ]] && continue
  gpu=$((shard % WORKERS))
  commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6_compact.workers --config '$CONFIG' --output '$OUTPUT' --device cuda:0 s0-shard --shard '$shard' --shards '$SHARDS' >'$LOGS/s0_${shard}.log' 2>&1")
done
[[ "${#commands[@]}" -eq 0 ]] || run_parallel "${commands[@]}"
if [[ ! -s "$OUTPUT/funnel/s0/COMPLETE.json" ]]; then
  "$PYTHON" -m sticky_lab.mode3_v6_compact.workers --config "$CONFIG" --output "$OUTPUT" merge-s0 --shards "$SHARDS" \
    --whitebox "$OUTPUT/tracks/whitebox/candidates.json" \
    --blackbox "$OUTPUT/tracks/blackbox/candidates.json" \
    --v5-history "$V5_HISTORY"
fi

for stage in s1 s2 s3; do
  STAGE="$stage"
  atomic_status running 0 "$STAGE"
  commands=()
  for shard in $(seq 0 $((SHARDS-1))); do
    target="$OUTPUT/funnel/$stage/shard_$(printf '%02d' "$shard")/COMPLETE.json"
    [[ -s "$target" ]] && continue
    gpu=$((shard % WORKERS))
    commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6_compact.workers --config '$CONFIG' --output '$OUTPUT' --device cuda:0 stage-shard --stage '$stage' --shard '$shard' --shards '$SHARDS' >'$LOGS/${stage}_${shard}.log' 2>&1")
  done
  [[ "${#commands[@]}" -eq 0 ]] || run_parallel "${commands[@]}"
  if [[ ! -s "$OUTPUT/funnel/$stage/COMPLETE.json" ]]; then
    "$PYTHON" -m sticky_lab.mode3_v6_compact.workers --config "$CONFIG" --output "$OUTPUT" merge-stage --stage "$stage" --shards "$SHARDS"
  fi
done

STAGE=validation
atomic_status running 0 "$STAGE"
commands=()
for shard in $(seq 0 $((SHARDS-1))); do
  target="$OUTPUT/validation/shard_$(printf '%02d' "$shard")/COMPLETE.json"
  [[ -s "$target" ]] && continue
  gpu=$((shard % WORKERS))
  commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6_compact.workers --config '$CONFIG' --output '$OUTPUT' --device cuda:0 validation-shard --shard '$shard' --shards '$SHARDS' >'$LOGS/validation_${shard}.log' 2>&1")
done
[[ "${#commands[@]}" -eq 0 ]] || run_parallel "${commands[@]}"
if [[ ! -s "$OUTPUT/validation/COMPLETE.json" ]]; then
  "$PYTHON" -m sticky_lab.mode3_v6_compact.workers --config "$CONFIG" --output "$OUTPUT" merge-validation --shards "$SHARDS"
fi

gate_open="$($PYTHON - "$OUTPUT/validation/COMPLETE.json" <<'PY'
import json,sys
print('yes' if json.load(open(sys.argv[1],encoding='utf-8'))['gate_open'] else 'no')
PY
)"
if [[ "$gate_open" == no ]]; then
  mkdir -p "$OUTPUT/sealed"
  "$PYTHON" - "$OUTPUT/sealed/NOT_RUN.json" <<'PY'
import json,sys
json.dump({'reason':'validation gate closed after one-cap and finalist-only two-cap rescue','test_ood_encoded':False,'compliant_negative_endpoint':True},open(sys.argv[1],'w'),indent=2,sort_keys=True); open(sys.argv[1],'a').write('\n')
PY
  STAGE=negative_endpoint
  "$PYTHON" -m sticky_lab.mode3_v6_compact.run --config "$CONFIG" --output "$OUTPUT" finalize
  exit 0
fi

STAGE=sealed_confirmation
atomic_status running 0 "$STAGE"
commands=()
phases=(test replication_0 replication_1 ood_0 ood_1 ood_2)
for index in "${!phases[@]}"; do
  phase="${phases[$index]}"
  [[ -s "$OUTPUT/sealed/$phase/COMPLETE.json" ]] && continue
  gpu=$((index % WORKERS))
  commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6_compact.sealed --config '$CONFIG' --output '$OUTPUT' --device cuda:0 confirm --phase '$phase' >'$LOGS/sealed_${phase}.log' 2>&1")
done
[[ "${#commands[@]}" -eq 0 ]] || run_parallel "${commands[@]}"
"$PYTHON" - "$OUTPUT/sealed/COMPLETE.json" <<'PY'
import json,os,sys,tempfile,pathlib
root=pathlib.Path(sys.argv[1]).parent; phases=['test','replication_0','replication_1','ood_0','ood_1','ood_2']
if not all((root/p/'COMPLETE.json').is_file() for p in phases): raise SystemExit('sealed phase incomplete')
fd,t=tempfile.mkstemp(dir=root,prefix='.complete.'); os.close(fd)
open(t,'w').write(json.dumps({'phases':phases,'refit_performed':False,'complete':True},indent=2,sort_keys=True)+'\n'); os.replace(t,sys.argv[1])
PY

STAGE=semantic_controls
atomic_status running 0 "$STAGE"
if [[ ! -s "$OUTPUT/semantic/COMPLETE.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m sticky_lab.mode3_v6_compact.sealed --config "$CONFIG" --output "$OUTPUT" --device cuda:0 build-semantic-metadata
fi
if [[ ! -s "$OUTPUT/semantic-controls/COMPLETE.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m sticky_lab.mode3_v6_compact.sealed --config "$CONFIG" --output "$OUTPUT" --device cuda:0 semantic-controls
fi

STAGE=mechanism
atomic_status running 0 "$STAGE"
if [[ ! -s "$OUTPUT/mechanism/COMPLETE.json" ]]; then
  "$PYTHON" -m sticky_lab.mode3_v6_compact.sealed --config "$CONFIG" --output "$OUTPUT" mechanism
fi

STAGE=retrieval
atomic_status running 0 "$STAGE"
if [[ ! -s "$OUTPUT/retrieval/COMPLETE.json" && ! -s "$OUTPUT/retrieval/SKIPPED.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m sticky_lab.mode3_v6_compact.sealed --config "$CONFIG" --output "$OUTPUT" --device cuda:0 retrieval
fi

STAGE=finalize
atomic_status running 0 "$STAGE"
"$PYTHON" -m sticky_lab.mode3_v6_compact.run --config "$CONFIG" --output "$OUTPUT" finalize
