#!/usr/bin/env bash
set -Eeuo pipefail

# Mode 3 V6 only. This script never invokes repository-wide tests or any V1--V5
# entry point. Formal execution is bound to one clean commit and fails before
# model loading if the document/source/capacity contract is unmet.
ROOT="${V6_ROOT:-/home/jkl/StickyToken-v6}"
PYTHON="${V6_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
CONFIG="${V6_CONFIG:-configs/v6_mode3.yaml}"
OUTPUT="${V6_OUTPUT:-results/sticky_lab/sentence_t5_base/mode3_v6}"
WORKERS="${V6_WORKERS:-8}"
V5_HISTORY="${V6_V5_HISTORY:-results/sticky_lab/sentence_t5_base/mode3_v5/publication/v5_single_token_history.json}"
V5_RESULTS="${V6_V5_RESULTS:-results/sticky_lab/sentence_t5_base/mode3_v5}"
LOGS="$OUTPUT/orchestration_logs"
export NLTK_DATA="${V6_NLTK_DATA:-/mnt/data/jkl/StickyToken-v6-resources/nltk_data}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$ROOT"
mkdir -p "$LOGS"
exec 9>"$OUTPUT/.orchestrator.lock"
if ! flock -n 9; then
  echo "another V6 orchestrator owns $OUTPUT" >&2
  exit 75
fi
RUN_COMMIT="$(git rev-parse HEAD)"
CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print $1}')"
printf '%s\n' "$$" >"$OUTPUT/.orchestrator.pid"

write_status() {
  local state="$1" code="$2"
  "$PYTHON" - "$LOGS/status.json" "$state" "$code" "$RUN_COMMIT" "$CONFIG_SHA" "$$" <<'PY'
import json,os,sys,tempfile,time
path,state,code,commit,config,pid=sys.argv[1:]
value={"state":state,"exit_code":int(code),"run_commit":commit,"config_sha256":config,"pid":int(pid),"updated_unix":time.time()}
os.makedirs(os.path.dirname(path),exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix='.status.',dir=os.path.dirname(path))
with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(value,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp,path)
PY
}
on_exit() {
  local code="$?"
  trap - EXIT
  if [[ "$code" -eq 0 ]]; then write_status complete "$code"; else write_status stopped "$code"; fi
  exit "$code"
}
trap on_exit EXIT
write_status running 0

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree is dirty; refusing formal V6" >&2
  exit 1
fi

"$PYTHON" -m pytest -q \
  tests/test_mode3_v6_core.py tests/test_mode3_v6_scope.py \
  tests/test_mode3_v6_data.py tests/test_mode3_v6_search.py \
  tests/test_mode3_v6_publication.py >"$LOGS/tests.log" 2>&1
"$PYTHON" scripts/audit_v6_mode3.py --config "$CONFIG" --output "$LOGS/scope_audit.json"

# This is intentionally hard-fail, not report-only.
"$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" preflight
if [[ -s "$OUTPUT/registration/run_contract.json" ]]; then
  "$PYTHON" - "$OUTPUT/registration/run_contract.json" "$RUN_COMMIT" "$CONFIG_SHA" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding='utf-8'))
if value.get('run_code_commit') != sys.argv[2] or value.get('config_sha256') != sys.argv[3]:
    raise SystemExit('existing V6 output is bound to a different code/config contract')
PY
else
  "$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" prepare
fi
if [[ ! -s "$OUTPUT/enumeration/COMPLETE.json" ]]; then
  "$PYTHON" -m sticky_lab.mode3_v6.workers --config "$CONFIG" --output "$OUTPUT" enumerate-vocab
fi

run_parallel() {
  local -a pids=()
  local failed=0
  while [[ "$#" -gt 0 ]]; do
    bash -c "$1" & pids+=("$!"); shift
    if [[ "${#pids[@]}" -ge "$WORKERS" ]]; then
      local pid; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=()
    fi
  done
  local pid; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}

commands=()
for shard in $(seq 0 31); do
  if [[ -s "$OUTPUT/screen/shard_$(printf '%02d' "$shard")/COMPLETE.json" && -s "$OUTPUT/screen/shard_$(printf '%02d' "$shard")/metrics.jsonl" ]]; then continue; fi
  gpu=$((shard % WORKERS))
  commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6.workers --config '$CONFIG' --output '$OUTPUT' --device cuda:0 screen-shard --shard $shard --shards 32 >'$LOGS/screen_$shard.log' 2>&1")
done
if [[ "${#commands[@]}" -gt 0 ]]; then run_parallel "${commands[@]}"; fi
if [[ ! -s "$OUTPUT/screen/COMPLETE.json" ]]; then
  "$PYTHON" -m sticky_lab.mode3_v6.workers --config "$CONFIG" --output "$OUTPUT" merge-screen --shards 32
fi

# Physically separated discovery tracks. White-box outputs never seed the
# black-box process; both enter only the later union/re-evaluation stage.
track_pids=()
if [[ ! -s "$OUTPUT/tracks/whitebox/COMPLETE.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m sticky_lab.mode3_v6.track_whitebox --config "$CONFIG" --output "$OUTPUT" --device cuda:0 >"$LOGS/whitebox.log" 2>&1 & track_pids+=("$!")
fi
if [[ ! -s "$OUTPUT/tracks/blackbox/restart_00/COMPLETE.json" ]]; then
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m sticky_lab.mode3_v6.track_blackbox --config "$CONFIG" --output "$OUTPUT" --device cuda:0 >"$LOGS/blackbox.log" 2>&1 & track_pids+=("$!")
fi
track_failed=0
for pid in "${track_pids[@]}"; do wait "$pid" || track_failed=1; done
if [[ "$track_failed" -ne 0 ]]; then
  echo "a V6 discovery track failed; inspect the preserved track logs" >&2
  exit 1
fi
if [[ ! -s "$OUTPUT/semantic/COMPLETE.json" ]]; then
  CUDA_VISIBLE_DEVICES=2 "$PYTHON" -m sticky_lab.mode3_v6.build_semantic_metadata --config "$CONFIG" --output "$OUTPUT" --device cuda:0
fi

if [[ ! -f "$V5_HISTORY" ]]; then
  V5_HISTORY="$OUTPUT/candidate_union/v5_history.json"
  "$PYTHON" scripts/extract_v5_single_token_history.py --v5-results "$V5_RESULTS" --output "$V5_HISTORY"
fi
if [[ ! -s "$OUTPUT/candidate_union/COMPLETE.json" ]]; then
  "$PYTHON" -m sticky_lab.mode3_v6.workers --config "$CONFIG" --output "$OUTPUT" build-union \
    --whitebox "$OUTPUT/tracks/whitebox/candidates.json" \
    --blackbox "$OUTPUT/tracks/blackbox/restart_00/candidates.json" \
    --v5-history "$V5_HISTORY"
fi

commands=()
for shard in $(seq 0 31); do
  if [[ -s "$OUTPUT/full_search/shard_$(printf '%02d' "$shard")/COMPLETE.json" && -s "$OUTPUT/full_search/shard_$(printf '%02d' "$shard")/metrics.jsonl" ]]; then continue; fi
  gpu=$((shard % WORKERS))
  commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6.workers --config '$CONFIG' --output '$OUTPUT' --device cuda:0 full-search-shard --shard $shard --shards 32 >'$LOGS/full_$shard.log' 2>&1")
done
if [[ "${#commands[@]}" -gt 0 ]]; then run_parallel "${commands[@]}"; fi
if [[ ! -s "$OUTPUT/full_search/COMPLETE.json" ]]; then
  "$PYTHON" -m sticky_lab.mode3_v6.workers --config "$CONFIG" --output "$OUTPUT" merge-full-search --shards 32
fi

commands=()
for shard in $(seq 0 31); do
  if [[ -s "$OUTPUT/validation/shard_$(printf '%02d' "$shard")/COMPLETE.json" && -s "$OUTPUT/validation/shard_$(printf '%02d' "$shard")/results.jsonl" ]]; then continue; fi
  gpu=$((shard % WORKERS))
  commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6.workers --config '$CONFIG' --output '$OUTPUT' --device cuda:0 validate-shard --shard $shard --shards 32 --validation-candidates 5000 >'$LOGS/validation_$shard.log' 2>&1")
done
if [[ "${#commands[@]}" -gt 0 ]]; then run_parallel "${commands[@]}"; fi
if [[ ! -s "$OUTPUT/validation/COMPLETE.json" ]]; then
  "$PYTHON" -m sticky_lab.mode3_v6.workers --config "$CONFIG" --output "$OUTPUT" merge-validation --shards 32
fi

if "$PYTHON" - "$OUTPUT/validation/gate_summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8"))["gate_open"] else 1)
PY
then
  if [[ ! -s "$OUTPUT/validation/frozen_cap.json" ]]; then
    "$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" freeze --freeze-source "$OUTPUT/validation/selected_freeze_source.json"
  fi
else
  if [[ ! -s "$OUTPUT/multicap_rescue/COMPLETE.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m sticky_lab.mode3_v6.multicap_rescue --config "$CONFIG" --output "$OUTPUT" --device cuda:0
  fi
  echo "No one-cap ST-FCA candidate certified; sealed one-cap Test/OOD remain unencoded." >&2
  "$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" inventory
  exit 2
fi

run_sealed() {
  local phase="$1"; shift
  local suffix="base"
  if [[ "$phase" == "replication" || "$phase" == "ood" ]]; then suffix="$(printf '%02d' "${2:-0}")"; fi
  local result="$OUTPUT/$phase/result.json"
  if [[ "$phase" == "replication" || "$phase" == "ood" ]]; then result="$OUTPUT/$phase/${phase}_${suffix}/result.json"; fi
  if [[ -s "$result" ]]; then return; fi
  local payload="$OUTPUT/.payload_${phase}_${suffix}.json"
  "$PYTHON" -m sticky_lab.mode3_v6.sealed_worker --config "$CONFIG" --output "$OUTPUT" --device cuda:0 --phase "$phase" "$@" --payload-output "$payload"
  "$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" "$phase" --payload "$payload"
}
run_sealed test
run_sealed replication --index 1
run_sealed replication --index 2
for index in 0 1 2 3; do run_sealed ood --index "$index"; done

semantic_payload="$OUTPUT/.payload_semantic.json"
if [[ ! -s "$OUTPUT/semantic-controls/result.json" ]]; then
  "$PYTHON" -m sticky_lab.mode3_v6.semantic_worker --config "$CONFIG" --output "$OUTPUT" --device cuda:0 \
    --metadata "$OUTPUT/semantic/token_metadata.jsonl" --payload-output "$semantic_payload"
  "$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" semantic-controls --payload "$semantic_payload"
fi

mechanism_payload="$OUTPUT/.payload_mechanism.json"
if [[ ! -s "$OUTPUT/mechanism/result.json" ]]; then
"$PYTHON" - "$OUTPUT/tracks/whitebox/candidates.json" "$mechanism_payload" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8")); value.update({"refit_performed":False,"search_feedback":False})
json.dump(value,open(sys.argv[2],"w",encoding="utf-8"),sort_keys=True,indent=2)
PY
"$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" mechanism --payload "$mechanism_payload"
fi

retrieval_payload="$OUTPUT/.payload_retrieval.json"
if [[ ! -s "$OUTPUT/retrieval/result.json" ]]; then
  "$PYTHON" -m sticky_lab.mode3_v6.retrieval_worker --config "$CONFIG" --output "$OUTPUT" --device cuda:0 --payload-output "$retrieval_payload"
  "$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" retrieval --payload "$retrieval_payload"
fi
"$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" finalize
