#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${V6_3_ROOT:-/mnt/data/jkl/StickyToken-v6-3-light-formal}"
PYTHON="${V6_3_PYTHON:-/home/jkl/anaconda3/envs/StickyToken/bin/python}"
OUTPUT="${V6_3_OUTPUT:-/mnt/data/jkl/StickyToken-v6-3-results/sticky_lab/sentence_t5_base/mode3_v6_3_light}"
RELEASE_DIR="${V6_3_RELEASE_DIR:-/mnt/data/jkl/StickyToken-v6-3-release}"

cd "$ROOT"
[[ -s "$OUTPUT/FINAL_STATUS.json" ]] || { echo "V6.3 FINAL_STATUS.json is absent" >&2; exit 66; }
mkdir -p "$RELEASE_DIR/assets"
"$PYTHON" - "$OUTPUT" <<'PY'
import json,pathlib,sys
from sticky_lab.mode3_v6_3.report import atomic_json,result_inventory
root=pathlib.Path(sys.argv[1]); atomic_json(root/'result_inventory.json',result_inventory(root))
PY
"$PYTHON" scripts/package_result_release_shards.py \
  --results "$OUTPUT" --inventory "$OUTPUT/result_inventory.json" \
  --output-dir "$RELEASE_DIR/assets" --asset-index "$RELEASE_DIR/asset_index.json" \
  --prefix mode3_v6_3_light --asset-stem mode3-v6-3-light-full-results \
  --schema-version mode3-v6-3-release-shards-v1 --maximum-uncompressed-bytes 1900000000
cp "$OUTPUT/result_inventory.json" "$RELEASE_DIR/result_inventory.json"
"$PYTHON" scripts/recover_v6_3_results.py \
  --asset-index "$RELEASE_DIR/asset_index.json" \
  --inventory "$RELEASE_DIR/result_inventory.json" \
  --archive-dir "$RELEASE_DIR/assets" \
  --destination "$RELEASE_DIR/fresh-clone-restore" \
  --audit-output "$RELEASE_DIR/FRESH_CLONE_RESTORE_AUDIT.json"
