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

Protocol revision 5 uses run ID `mode3_v6_3_light_r5`. It retains the tokenizer
identity, NumPy 2.x and runtime-GPU marker repairs. Revision 5 adds a dynamic GPU
safety state machine: a worker launches only with at least 12 GiB free; if its runtime
reserve falls below 8 GiB, the orchestrator requests a cooperative yield. Encoder
calls are split into registered 1,024-text chunks, and a worker observes the request
only after its budget reservation, call-registry bits and cache chunk are durable.
The tokenizer hash contract is
`sorted_token_id_nul_text_lf_v1`; backend-tokenizer JSON hashes are diagnostic only.
Identity checks run before any corpus scan or role write. Superseded dry-runs are
preserved and marked `INVALIDATED_PROTOCOL_CHANGE`; revision-5 dry-run and pilot use new
parent directories and never reuse partial artifacts from earlier revisions.

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

Workers may use only physical GPUs 4–7. The orchestrator inspects those devices every
second and starts at most one V6.3 worker per safe device. It never signals, kills,
pauses or modifies another user's process. If no authorized GPU is safe, status is
`waiting_gpu` rather than fallback to GPUs 0–3. A yield timeout blocks automatic replay;
the orchestrator may terminate only its exact child process and must preserve all output
for diagnosis.

Recovery is marker-based and fail closed. A `COMPLETE.json` may be reused only after its
identity and hashes validate. `FAILED.json`, partial output, hash drift, budget hard stop or
unknown service exit blocks automatic restart. No partial shard may be merged.
