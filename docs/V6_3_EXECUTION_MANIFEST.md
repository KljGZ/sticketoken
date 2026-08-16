# V6.3 execution manifest

Branch: `codex/mode3-v6.3-light`
Base commit: `fa0ccb773ef1174112b77181f0769c7e3d0ff273`
Formal code tree: `/mnt/data/jkl/StickyToken-v6-3-light-formal`
Formal output: `/mnt/data/jkl/StickyToken-v6-3-results/sticky_lab/sentence_t5_base/mode3_v6_3_light`
Service: `sticky-v6-3-light.service`

V6.2 was stopped and archived before this worktree was created. Its immutable audit is
`/mnt/data/jkl/StickyToken-v6-2-formal/audit/v6_3_stop_20260816T075049Z`; the audit JSON
SHA-256 is `76e1bd9192be6c4700f4c9f4446e6983a9efd48b122196ab02ff9e9fff54dab8`.
The archived code/config identities are
`01308cbf655f5763ed7fe6afd2951adc3f1f0aaa` and
`46a2ca7b2ccb1172c7382e0ca5858523a2665a757b73d0fe02180a70f772adf5`, and the preserved
branch is `archive/mode3-v6.2-interrupted-20260816T075049Z`. No old output was deleted.

The formal run must be a clean detached checkout of the pushed V6.3 commit. It must bind
the source config hash, resolved config hash, protocol hash, code commit, model revision,
tokenizer hash, data manifests and role/call-space hashes. Dry-run and pilot use separate
parents with the same required output leaf and are explicitly non-scientific.

Execution order:

1. preflight and physical role sealing;
2. exhaustive legal-token enumeration and merge;
3. discovery clean-base precompute;
4. S0, S1, S2 and Full stage shards and complete merges;
5. all-position top-100 refit and merge;
6. primary/secondary freeze while confirm remains sealed;
7. hash-bound access grant and confirm call-space creation;
8. independent clean-base precompute and fixed-cap confirm;
9. Core-gated semantic/IID/OOD/retrieval follow-ups;
10. final report, result inventory and release shards;
11. fresh-clone restore verification.

Workers may use only physical GPUs 4–7. The orchestrator must inspect those devices and
start at most one V6.3 worker per safe device. It must not kill, pause or modify another
user's job. If no authorized GPU is safe, status is waiting rather than fallback to 0–3.

Recovery is marker-based and fail closed. A `COMPLETE.json` may be reused only after its
identity and hashes validate. `FAILED.json`, partial output, hash drift, budget hard stop or
unknown service exit blocks automatic restart. No partial shard may be merged.
