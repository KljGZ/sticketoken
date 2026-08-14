# V6 Compact publication index

This directory is the readable and cryptographic index for the formal
StickyToken Mode 3 V6 Compact result. The registered Validation gate closed, so
the run ended compliantly without encoding sealed Test/OOD or downstream stages.

Key files:

- `summary.json`: machine-readable endpoint, funnel, best near miss, data, and budget.
- `validation_one_cap_candidates.csv`: all 20 one-cap Validation candidates.
- `validation_two_cap_rescues.csv`: finalist-only two-cap rescue results.
- `exemplar_evidence.json` and `exemplar_radial_shells.csv`: compact evidence for the closest registered near miss.
- `complete_file_manifest.csv` and `remote_inventory.json`: authoritative post-final per-file hashes and content root.
- `release_asset_index.json`: names, sizes, and SHA-256 hashes of the three raw-result release shards.
- `PACKAGE_COMPLETE.json`: remote packaging completion marker.
- `run_contract.json`, `v6_mode3_compact.yaml`, and budget/status files: frozen run lineage.
- `blackbox_restart_00_attempt_01_cuda_oom.log`: preserved resource-error evidence.

The full raw result identity is 764 files, 3,446,888,256 bytes, and content root
`9dcac6ead43975c71ebe7db14ff3539e8bc0c1766ad0ae4c8b950fd6114b400b`.

Release: [mode3-v6-compact-full-results](https://github.com/KljGZ/sticketoken/releases/tag/mode3-v6-compact-full-results)

After cloning the publication branch, restore and verify the release with:

```powershell
python scripts\recover_v6_compact_results.py `
  --asset-index results_publication\v6_compact\release_asset_index.json `
  --manifest results_publication\v6_compact\complete_file_manifest.csv `
  --destination D:\StickyToken-v6-compact-restored `
  --audit-output D:\StickyToken-v6-compact-restored\fresh_clone_release_audit.json
```

The recovery is accepted only if file count, total bytes, every file SHA-256,
and the deterministic content root all match. The completed independent recovery
is recorded in `fresh_clone_release_audit.json`: 764 files and 3,446,888,256
bytes were restored with zero missing, extra, size-mismatched, or hash-mismatched
files, and `triple_identity_ready` is true.
