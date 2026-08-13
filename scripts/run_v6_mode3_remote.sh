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

cd "$ROOT"
mkdir -p "$LOGS"
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
"$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" prepare
"$PYTHON" -m sticky_lab.mode3_v6.workers --config "$CONFIG" --output "$OUTPUT" enumerate-vocab

run_parallel() {
  local -a pids=()
  while [[ "$#" -gt 0 ]]; do
    bash -c "$1" & pids+=("$!"); shift
    if [[ "${#pids[@]}" -ge "$WORKERS" ]]; then
      local pid; for pid in "${pids[@]}"; do wait "$pid"; done; pids=()
    fi
  done
  local pid; for pid in "${pids[@]}"; do wait "$pid"; done
}

commands=()
for shard in $(seq 0 31); do
  gpu=$((shard % WORKERS))
  commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6.workers --config '$CONFIG' --output '$OUTPUT' --device cuda:0 screen-shard --shard $shard --shards 32 >'$LOGS/screen_$shard.log' 2>&1")
done
run_parallel "${commands[@]}"
"$PYTHON" -m sticky_lab.mode3_v6.workers --config "$CONFIG" --output "$OUTPUT" merge-screen --shards 32

# Physically separated discovery tracks. White-box outputs never seed the
# black-box process; both enter only the later union/re-evaluation stage.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m sticky_lab.mode3_v6.track_whitebox --config "$CONFIG" --output "$OUTPUT" --device cuda:0 >"$LOGS/whitebox.log" 2>&1 & whitebox_pid=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m sticky_lab.mode3_v6.track_blackbox --config "$CONFIG" --output "$OUTPUT" --device cuda:0 >"$LOGS/blackbox.log" 2>&1 & blackbox_pid=$!
wait "$whitebox_pid"; wait "$blackbox_pid"
"$PYTHON" -m sticky_lab.mode3_v6.build_semantic_metadata --config "$CONFIG" --output "$OUTPUT" --device cuda:0

if [[ ! -f "$V5_HISTORY" ]]; then
  V5_HISTORY="$OUTPUT/candidate_union/v5_history.json"
  "$PYTHON" scripts/extract_v5_single_token_history.py --v5-results "$V5_RESULTS" --output "$V5_HISTORY"
fi
"$PYTHON" -m sticky_lab.mode3_v6.workers --config "$CONFIG" --output "$OUTPUT" build-union \
  --whitebox "$OUTPUT/tracks/whitebox/candidates.json" \
  --blackbox "$OUTPUT/tracks/blackbox/restart_00/candidates.json" \
  --v5-history "$V5_HISTORY"

commands=()
for shard in $(seq 0 31); do
  gpu=$((shard % WORKERS))
  commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6.workers --config '$CONFIG' --output '$OUTPUT' --device cuda:0 full-search-shard --shard $shard --shards 32 >'$LOGS/full_$shard.log' 2>&1")
done
run_parallel "${commands[@]}"
"$PYTHON" -m sticky_lab.mode3_v6.workers --config "$CONFIG" --output "$OUTPUT" merge-full-search --shards 32

commands=()
for shard in $(seq 0 31); do
  gpu=$((shard % WORKERS))
  commands+=("CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' -m sticky_lab.mode3_v6.workers --config '$CONFIG' --output '$OUTPUT' --device cuda:0 validate-shard --shard $shard --shards 32 --validation-candidates 5000 >'$LOGS/validation_$shard.log' 2>&1")
done
run_parallel "${commands[@]}"
"$PYTHON" -m sticky_lab.mode3_v6.workers --config "$CONFIG" --output "$OUTPUT" merge-validation --shards 32

if "$PYTHON" - "$OUTPUT/validation/gate_summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8"))["gate_open"] else 1)
PY
then
  "$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" freeze --freeze-source "$OUTPUT/validation/selected_freeze_source.json"
else
  "$PYTHON" -m sticky_lab.mode3_v6.multicap_rescue --config "$CONFIG" --output "$OUTPUT" --device cuda:0
  echo "No one-cap ST-FCA candidate certified; sealed one-cap Test/OOD remain unencoded." >&2
  "$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" inventory
  exit 2
fi

run_sealed() {
  local phase="$1"; shift
  local payload="$OUTPUT/.payload_${phase}_$RANDOM.json"
  "$PYTHON" -m sticky_lab.mode3_v6.sealed_worker --config "$CONFIG" --output "$OUTPUT" --device cuda:0 --phase "$phase" "$@" --payload-output "$payload"
  "$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" "$phase" --payload "$payload"
}
run_sealed test
run_sealed replication --index 1
run_sealed replication --index 2
for index in 0 1 2 3; do run_sealed ood --index "$index"; done

semantic_payload="$OUTPUT/.payload_semantic.json"
"$PYTHON" -m sticky_lab.mode3_v6.semantic_worker --config "$CONFIG" --output "$OUTPUT" --device cuda:0 \
  --metadata "$OUTPUT/semantic/token_metadata.jsonl" --payload-output "$semantic_payload"
"$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" semantic-controls --payload "$semantic_payload"

mechanism_payload="$OUTPUT/.payload_mechanism.json"
"$PYTHON" - "$OUTPUT/tracks/whitebox/candidates.json" "$mechanism_payload" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8")); value.update({"refit_performed":False,"search_feedback":False})
json.dump(value,open(sys.argv[2],"w",encoding="utf-8"),sort_keys=True,indent=2)
PY
"$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" mechanism --payload "$mechanism_payload"

retrieval_payload="$OUTPUT/.payload_retrieval.json"
"$PYTHON" -m sticky_lab.mode3_v6.retrieval_worker --config "$CONFIG" --output "$OUTPUT" --device cuda:0 --payload-output "$retrieval_payload"
"$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" retrieval --payload "$retrieval_payload"
"$PYTHON" -m sticky_lab.mode3_v6.run --config "$CONFIG" --output "$OUTPUT" finalize
