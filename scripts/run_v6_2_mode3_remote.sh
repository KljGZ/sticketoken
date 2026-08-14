#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${V6_2_ROOT:-/home/jkl/StickyToken-v6-2}"
PYTHON="${V6_2_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
CONFIG="${V6_2_CONFIG:-configs/v6_2_mode3.yaml}"
OUTPUT="${V6_2_OUTPUT:-/mnt/data/jkl/StickyToken-v6-2-results/sticky_lab/sentence_t5_base/mode3_v6_2}"
WORKERS="${V6_2_WORKERS:-8}"
SHARDS="${V6_2_SHARDS:-32}"
ENUM_WORKERS="${V6_2_ENUM_WORKERS:-16}"
LOGS="$OUTPUT/orchestration_logs"
export NLTK_DATA="${V6_2_NLTK_DATA:-/mnt/data/jkl/StickyToken-v6-resources/nltk_data}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false

cd "$ROOT"
mkdir -p "$LOGS"
exec 9>"$OUTPUT/.orchestrator.lock"
flock -n 9 || { echo "another V6.2 orchestrator owns $OUTPUT" >&2; exit 75; }
RUN_COMMIT="$(git rev-parse HEAD)"; CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print $1}')"
printf '%s\n' "$$" >"$OUTPUT/.orchestrator.pid"

atomic_status() {
  local state="$1" code="$2" stage="$3"
  "$PYTHON" - "$LOGS/status.json" "$state" "$code" "$stage" "$RUN_COMMIT" "$CONFIG_SHA" "$$" <<'PY'
import json,os,sys,tempfile,time
path,state,code,stage,commit,config,pid=sys.argv[1:]
value={"schema_version":"mode3-v6-2-orchestrator-status-v1","state":state,"exit_code":int(code),"stage":stage,"run_commit":commit,"config_sha256":config,"pid":int(pid),"updated_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
os.makedirs(os.path.dirname(path),exist_ok=True); fd,tmp=tempfile.mkstemp(prefix='.status.',dir=os.path.dirname(path))
with os.fdopen(fd,'w',encoding='utf-8') as h: json.dump(value,h,indent=2,sort_keys=True); h.write('\n'); h.flush(); os.fsync(h.fileno())
os.replace(tmp,path)
PY
}
STAGE=starting
on_exit() { local code="$?"; trap - EXIT; [[ "$code" -eq 0 ]] && atomic_status complete "$code" "$STAGE" || atomic_status stopped "$code" "$STAGE"; exit "$code"; }
trap on_exit EXIT
atomic_status running 0 "$STAGE"

[[ -z "$(git status --porcelain)" ]] || { echo "formal V6.2 requires a clean worktree" >&2; exit 1; }

run_parallel() {
  local -a commands=("$@") pids=(); local command pid failed=0
  for command in "${commands[@]}"; do
    bash -c "$command" & pids+=("$!")
    if [[ "${#pids[@]}" -ge "$WORKERS" ]]; then
      for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); [[ "$failed" -eq 0 ]] || return 1
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}

run_stage() {
  local stage="$1"; local -a commands=(); STAGE="$stage"; atomic_status running 0 "$STAGE"
  for shard in $(seq 0 $((SHARDS-1))); do
    local target="$OUTPUT/funnel/$stage/shard_$(printf '%02d' "$shard")/COMPLETE.json"
    [[ -s "$target" ]] && continue
    local gpu=$((shard % WORKERS))
    commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6_2.workers --config '$CONFIG' --output '$OUTPUT' --device cuda:0 stage-shard --stage '$stage' --shard '$shard' --shards '$SHARDS' >'$LOGS/${stage}_${shard}.log' 2>&1")
  done
  [[ "${#commands[@]}" -eq 0 ]] || run_parallel "${commands[@]}"
  [[ -s "$OUTPUT/funnel/$stage/COMPLETE.json" ]] || "$PYTHON" -m sticky_lab.mode3_v6_2.workers --config "$CONFIG" --output "$OUTPUT" merge-stage --stage "$stage" --shards "$SHARDS"
}

precompute_roles() {
  local phase="$1"; shift; local -a roles=("$@") commands=(); STAGE="base_embeddings_$phase"; atomic_status running 0 "$STAGE"
  for index in "${!roles[@]}"; do
    local role="${roles[$index]}"
    [[ -s "$OUTPUT/base_embeddings/$role.npy" && -s "$OUTPUT/base_embeddings/$role.json" ]] && continue
    local gpu=$((index % WORKERS))
    commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6_2.workers --config '$CONFIG' --output '$OUTPUT' --device cuda:0 precompute-role --role '$role' >'$LOGS/base_${role}.log' 2>&1")
  done
  [[ "${#commands[@]}" -eq 0 ]] || run_parallel "${commands[@]}"
}

STAGE=tests; atomic_status running 0 "$STAGE"
"$PYTHON" -m pytest -q -p no:cacheprovider tests/test_mode3_v6_2_core.py tests/test_mode3_v6_2_data.py tests/test_mode3_v6_2_pipeline.py tests/test_mode3_v6_2_scope.py >"$LOGS/tests.log" 2>&1
"$PYTHON" scripts/audit_v6_2_mode3.py --config "$CONFIG" --output "$LOGS/scope_audit.json"
"$PYTHON" scripts/budget_v6_2_mode3.py --config "$CONFIG" --output "$OUTPUT/budget/planned.json" >"$LOGS/budget.log"

STAGE=prepare; atomic_status running 0 "$STAGE"
"$PYTHON" -m sticky_lab.mode3_v6_2.run --config "$CONFIG" --output "$OUTPUT" preflight
"$PYTHON" -m sticky_lab.mode3_v6_2.run --config "$CONFIG" --output "$OUTPUT" prepare

if [[ ! -s "$OUTPUT/enumeration/COMPLETE.json" ]]; then
  STAGE=enumeration; atomic_status running 0 "$STAGE"
  commands=()
  for shard in $(seq 0 $((SHARDS-1))); do
    [[ -s "$OUTPUT/enumeration/shard_$(printf '%02d' "$shard")/COMPLETE.json" ]] && continue
    commands+=("'$PYTHON' -m sticky_lab.mode3_v6_2.workers --config '$CONFIG' --output '$OUTPUT' enumerate-vocab --shard '$shard' --shards '$SHARDS' >'$LOGS/enumeration_${shard}.log' 2>&1")
  done
  original_workers="$WORKERS"; WORKERS="$ENUM_WORKERS"
  [[ "${#commands[@]}" -eq 0 ]] || run_parallel "${commands[@]}"
  WORKERS="$original_workers"
  "$PYTHON" -m sticky_lab.mode3_v6_2.workers --config "$CONFIG" --output "$OUTPUT" merge-enumeration --shards "$SHARDS" >"$LOGS/enumeration_merge.log" 2>&1
fi

mapfile -t discovery_roles < <("$PYTHON" - <<'PY'
from sticky_lab.mode3_v6_2.roles import DISCOVERY_ORDER
print('\n'.join(DISCOVERY_ORDER))
PY
)
precompute_roles discovery "${discovery_roles[@]}"

# Exhaustive enumeration is the formal search. Optional black/white-box tracks
# are the first registered budget reduction and never seed this funnel.
run_stage s0
run_stage s1
run_stage s2
run_stage full
run_stage stability

STAGE=semantic_metadata; atomic_status running 0 "$STAGE"
[[ -s "$OUTPUT/semantic/METADATA_COMPLETE.json" ]] || CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m sticky_lab.mode3_v6_2.semantic --config "$CONFIG" --output "$OUTPUT" --device cuda:0 build-metadata >"$LOGS/semantic_metadata.log" 2>&1

STAGE=semantic_discovery; atomic_status running 0 "$STAGE"; commands=()
for shard in $(seq 0 $((SHARDS-1))); do
  [[ -s "$OUTPUT/semantic/shard_$(printf '%02d' "$shard")/COMPLETE.json" ]] && continue
  gpu=$((shard % WORKERS)); commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6_2.semantic --config '$CONFIG' --output '$OUTPUT' --device cuda:0 discovery-shard --shard '$shard' --shards '$SHARDS' >'$LOGS/semantic_${shard}.log' 2>&1")
done
[[ "${#commands[@]}" -eq 0 ]] || run_parallel "${commands[@]}"
[[ -s "$OUTPUT/semantic/COMPLETE.json" ]] || "$PYTHON" -m sticky_lab.mode3_v6_2.semantic --config "$CONFIG" --output "$OUTPUT" merge-discovery --shards "$SHARDS"

STAGE=protocol_selection; atomic_status running 0 "$STAGE"; commands=()
for shard in $(seq 0 $((SHARDS-1))); do
  [[ -s "$OUTPUT/protocol_selection/shard_$(printf '%02d' "$shard")/COMPLETE.json" ]] && continue
  gpu=$((shard % WORKERS)); commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6_2.selection --config '$CONFIG' --output '$OUTPUT' --device cuda:0 position-shard --shard '$shard' --shards '$SHARDS' >'$LOGS/protocol_${shard}.log' 2>&1")
done
[[ "${#commands[@]}" -eq 0 ]] || run_parallel "${commands[@]}"
[[ -s "$OUTPUT/freezes/COMPLETE.json" ]] || "$PYTHON" -m sticky_lab.mode3_v6_2.selection --config "$CONFIG" --output "$OUTPUT" merge-and-freeze --shards "$SHARDS"

mapfile -t sealed_roles < <("$PYTHON" - "$CONFIG" <<'PY'
import sys,yaml
c=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); roles=['confirm_trigger','confirm_benign','semantic_confirm','iid_replication_0','iid_replication_1','iid_replication_2','retrieval_probe']
for i in range(int(c['data']['ood_domains'])): roles += [f'ood_{i}_trigger',f'ood_{i}_benign']
print('\n'.join(roles))
PY
)
precompute_roles sealed "${sealed_roles[@]}"

STAGE=core_confirmation; atomic_status running 0 "$STAGE"
[[ -s "$OUTPUT/confirmation/COMPLETE.json" ]] || CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m sticky_lab.mode3_v6_2.sealed --config "$CONFIG" --output "$OUTPUT" --device cuda:0 confirm-core >"$LOGS/core_confirmation.log" 2>&1

STAGE=semantic_confirmation; atomic_status running 0 "$STAGE"
[[ -s "$OUTPUT/semantic_confirmation/COMPLETE.json" ]] || CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m sticky_lab.mode3_v6_2.sealed --config "$CONFIG" --output "$OUTPUT" --device cuda:0 semantic-confirmation >"$LOGS/semantic_confirmation.log" 2>&1

gate_open="$("$PYTHON" - "$OUTPUT/confirmation/COMPLETE.json" <<'PY'
import json,sys
print('yes' if json.load(open(sys.argv[1],encoding='utf-8'))['any_core_certified'] else 'no')
PY
)"
if [[ "$gate_open" == yes ]]; then
  STAGE=sealed_followups; atomic_status running 0 "$STAGE"; commands=(); phases=(iid_replication_0 iid_replication_1 iid_replication_2 ood_0 ood_1 ood_2 ood_3)
  for index in "${!phases[@]}"; do
    phase="${phases[$index]}"; [[ -s "$OUTPUT/sealed_followups/$phase/COMPLETE.json" ]] && continue
    gpu=$((index % WORKERS)); commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6_2.sealed --config '$CONFIG' --output '$OUTPUT' --device cuda:0 confirm-followup --phase '$phase' >'$LOGS/followup_${phase}.log' 2>&1")
  done
  [[ "${#commands[@]}" -eq 0 ]] || run_parallel "${commands[@]}"
  "$PYTHON" - "$OUTPUT/sealed_followups/COMPLETE.json" <<'PY'
import json,os,sys,tempfile,pathlib
p=pathlib.Path(sys.argv[1]); phases=['iid_replication_0','iid_replication_1','iid_replication_2','ood_0','ood_1','ood_2','ood_3']
if not all((p.parent/x/'COMPLETE.json').is_file() for x in phases): raise SystemExit('sealed followup incomplete')
p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent,prefix='.complete.'); os.close(fd); open(t,'w').write(json.dumps({'phases':phases,'refit_performed':False},indent=2)+'\n'); os.replace(t,p)
PY
  STAGE=retrieval; atomic_status running 0 "$STAGE"
  [[ -s "$OUTPUT/retrieval/COMPLETE.json" ]] || CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m sticky_lab.mode3_v6_2.sealed --config "$CONFIG" --output "$OUTPUT" --device cuda:0 retrieval >"$LOGS/retrieval.log" 2>&1
else
  mkdir -p "$OUTPUT/sealed_followups"
  "$PYTHON" - "$OUTPUT/sealed_followups/COMPLETE.json" <<'PY'
import json,sys
json.dump({'phases':[],'reason':'independent core gate closed','sealed_roles_encoded_for_core_only':True,'refit_performed':False},open(sys.argv[1],'w'),indent=2); open(sys.argv[1],'a').write('\n')
PY
fi

STAGE=finalize; atomic_status running 0 "$STAGE"
"$PYTHON" -m sticky_lab.mode3_v6_2.run --config "$CONFIG" --output "$OUTPUT" finalize
