# StickyToken Mode 3 V6 Compact full results

This release contains the complete raw output for formal commit
`dc2d9b9e4250954c0b651eb7f207282f3ee57d20` and config SHA-256
`31845bf2fc12ec8ae0b285d5c20e9833bb2529ab3bf9a896af0cacbeeb3e2597`.

The run reached a compliant negative endpoint: 20 finalists were evaluated, but
none passed every registered Validation confidence-bound gate. Consequently no
candidate was frozen and sealed Test, replication, OOD, semantic, mechanism, and
retrieval stages were not encoded.

Raw-result identity:

- 764 files
- 3,446,888,256 bytes
- content root SHA-256 `9dcac6ead43975c71ebe7db14ff3539e8bc0c1766ad0ae4c8b950fd6114b400b`
- three deterministic, content-addressed tar.gz shards

The Git branch includes the authoritative manifest, release index, readable
report, plots, and `scripts/recover_v6_compact_results.py`. Recovery is valid
only when all per-file hashes and the content root match.

The post-publication fresh-clone audit completed successfully from commit
`245fa08aa10621c357e8f3cfc2738fefdb7a0aff`. It downloaded all three release
shards, restored 764 files totaling 3,446,888,256 bytes, reproduced the registered
content root, and found zero missing, extra, size-mismatched, or hash-mismatched
files. The release includes `fresh_clone_release_audit.json` as independent
machine-readable evidence.
