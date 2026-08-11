# Mode 3 V4 complete result publication

This publication closes the V4 reproducibility gap before Mode 3 V5 begins. The V4 code and experiment are frozen; no V4 model query, search, validation, test, or OOD stage was rerun.

## Content identity

- Remote result root: `/home/jkl/StickyToken-v4/results/sticky_lab/sentence_t5_base/mode3_v4`
- Local audit root: `results/sticky_lab/sentence_t5_base/mode3_v4`
- Files: 2,856
- Uncompressed bytes: 157,863,998
- Content root SHA-256: `b4b871109499d8fede3396f776b119682db6fe0f874037b93baba81e0d667bd1`
- Original V4 manifest rows: 2,854
- Single-token completed shards: 32
- CEM search archives: 232
- Task-by-length validations: 120
- Query ledgers: 385
- Length-frontier rows: 120

`results_publication/v4/remote_inventory.json` and `local_inventory.json` independently inventory the remote and local trees. `inventory_diff.json` records no missing, extra, byte-mismatched, or hash-mismatched files. `complete_file_manifest.csv` is the release reconstruction contract.

## Release asset

- GitHub tag: `mode3-v4-full-results`
- Asset: `mode3-v4-full-results.tar.gz`
- Asset bytes: 94,014,838
- Asset SHA-256: `30eb4a2a41ff0d2b4101faa5c032f1ce6187d65cd7039be906bdcee7560004d0`
- Archive prefix: `mode3_v4/`

The archive is deterministic: entries follow the complete manifest order, store regular files only, and normalize ownership, mode, and timestamp metadata. Large raw artifacts stay out of ordinary Git history.

## Recovery

Use `scripts/recover_v4_results.py` with the asset hash and `complete_file_manifest.csv`. The script rejects path traversal and non-regular archive members, verifies the compressed asset, restores atomically per file, and then verifies every restored file's path, byte count, and SHA-256.

The publication is accepted only after a fresh clone downloads the GitHub Release asset and reconstructs the same 2,856-file, 157,863,998-byte content root.

## Scientific status

V4 completed all registered searches and validations but found no validation-certified universal candidate. Prefix-specific certificates exist, with the shortest at actual token length 10. Because the universal gate stayed closed, V4 correctly did not open the one-time IID test, OOD trigger evaluation, or one-poison retrieval stage. V5 uses fresh splits and a new protocol; this V4 package is read-only audit evidence.
