# Mode 3 V4 audited results

## Registered question and outcome

Mode 3 V4 asked whether a discrete trigger of actual tokenizer length 1--30, inserted exactly once, could move heterogeneous texts into one fixed, compact, support-interior, low-normal-occupancy representation region across prefix, suffix, and deterministic-random insertion positions.

The registered search and validation grid completed, but no universal candidate was validation-certified. The result is therefore negative under the registered model, data, thresholds, search budget, and candidate family:

> No cross-position V4 attractor was discovered. This bounded negative result is not a proof that no such trigger exists.

Because no universal validation certificate existed, V4 correctly did not freeze a final trigger, did not open the one-time IID test or fixed-center OOD evaluation, and did not run the gated one-poison retrieval experiment. Those stages are inapplicable rather than missing.

## Reproducibility envelope

- Encoder: `sentence-transformers/sentence-t5-base`
- Observed model revision: `fc5d4628481afbbaaacd7af6bb07cf9d3865f781`
- Embedding dimension: 768
- Threat model: final embedding-output query-only black box
- Registered V1--V3 baseline: `b78bb21693e87a62287929683f575eb2a1b89be1`
- V4 random seed: 44042
- Legal non-special, exact-roundtrip single tokens: 21,984
- Length schedule: every actual tokenizer length 1--30, step 1
- Insertion protocol: shared literal inserted once at prefix, suffix, deterministic-random, and the universal union of all three positions

Tokenizer access was restricted to legal candidate construction, exact-length auditing, round-trip verification, and contextual realizability. The search used no gradients, parameters, input embeddings, hidden states, HotFlip, soft prompts, continuous prompts, or retrieval feedback.

## Data audit

The IID pool contained 37,000 input rows from 37 source files. Global normalization and deduplication produced 17,943 unique texts before the token-length filter and 17,621 after it. The fixed roles were:

| Role | Texts |
|---|---:|
| Search trigger | 3,000 |
| Search benign support | 3,000 |
| Validation trigger | 1,000 |
| Validation benign support | 3,000 |
| Reserved one-time test trigger | 1,000 |
| Reserved one-time test benign support | 3,000 |
| Reserved source-disjoint OOD | 1,000 |

All pairwise IID role overlaps were zero at both sentence and fallback-group level. OOD overlap with all IID roles was zero. Original document provenance was unavailable, so the registered conservative fallback was one group per globally unique sentence; this limitation must accompany any claim about group independence.

## Search and validation completion

- All 21,984 legal single tokens were screened for each of four tasks in 32 completed shards.
- For each task and every length 2--30, two independent categorical CEM restarts completed: 4 x 29 x 2 = 232 search archives.
- No warm start from another length or from Modes 1/2 was used.
- All 4 x 30 = 120 task-by-length validations completed.
- Each validation compared the eight archived candidates with 64 same-length random triggers at the registered 0.99 quantile.
- Confidence bounds used 500 grouped bootstrap replicates and 20,000 sampled pairs. The exact pairwise lookups were block-vectorized without changing the registered sample count.

The query ledger contains 385 process records:

| Phase | Ledgers | Encode calls | Requested texts | Submitted texts |
|---|---:|---:|---:|---:|
| Prepare | 1 | 7 | 15,000 | 15,000 |
| Exhaustive single-token screen | 32 | 11,008 | 6,331,392 | 6,331,392 |
| Independent CEM search | 232 | 13,920 | 8,017,920 | 7,960,272 |
| Validation | 120 | 1,080 | 12,960,000 | 12,960,000 |
| **Total** | **385** | **26,015** | **27,324,312** | **27,266,664** |

The difference of 57,648 texts is recorded as cache hits.

The per-process ledgers also preserve the implementation commit used for each query. Preparation, exhaustive screening, and all CEM searches submitted 14,306,664 texts under `3ae2128`. One 72,000-text validation ran under `9d20db9`; the remaining 12,888,000 validation texts ran under `0be6cc9`, which block-vectorized the same exact bootstrap lookups. V4-only tests compare the vectorized confidence bounds with the scalar reference. Later commits `5ee9a75` and `03182ea` change the result audit and certificate-aligned visualization/report only; they do not alter the recorded search or validation outputs.

## Certification frontier

| Task | Certified lengths | Count |
|---|---|---:|
| Prefix | 10, 11, 14, 15, 16, 18--30 | 18 |
| Suffix | None | 0 |
| Deterministic-random | None | 0 |
| Universal fixed-center cross-position | None | 0 |

The shortest position-specific certificate was prefix length 10:

- Token IDs: `12697,2732,21762,5510,20076,570,28051,5088,16929,26178`
- Decoded literal: `priorities atunci Pokémon ultimately celebrities list cherish Rose eaten Holocaust`
- Context realizability: 1.0, with actual realized length exactly 10
- Displacement q05 95% lower bound: 0.51008 (threshold >= 0.02)
- Compact radius q95 95% upper bound: 0.32832 (threshold <= 0.40)
- Relative contraction q95 95% upper bound: 0.48612 (threshold <= 0.60)
- Support-interior margin: 0.04535 (threshold >= 0)
- Normal occupancy 95% upper bound: 0.000998 at 1x and 0.003049 at 2x radius (thresholds <= 0.001 and <= 0.01)
- Same-length random-baseline exceedance: passed

This is a prefix-only phenomenon and is not the V4 target. It was not frozen or sent to test.

Across the selected best universal candidate at each of 30 lengths, the gate pass counts were:

| Gate | Passed lengths |
|---|---:|
| Exact realizability | 30/30 |
| Non-trivial displacement | 30/30 |
| Support-interior center | 30/30 |
| Same-length random-baseline exceedance | 30/30 |
| Absolute compactness | 16/30 |
| Relative semantic contraction | 18/30 |
| Multi-scale low normal occupancy | **0/30** |

Thus low occupancy was the universal bottleneck. The nearest candidate was universal length 23. It passed every other registered gate, but failed multi-scale occupancy:

- Token IDs: `191,26297,17310,9909,1088,13155,3273,28648,26979,955,27601,19741,10110,9701,14566,7554,803,11269,12010,27500,7081,15108,31330`
- Actual tokenizer length and contextual realized length: exactly 23
- Displacement q05 95% lower bound: 0.46847
- Compact radius q95 95% upper bound: 0.36254
- Relative contraction q95 95% upper bound: 0.53611
- Support-interior margin: 0.05374
- Maximum position-center drift: 0.05712
- At 1x and 1.5x radius: 0/3,000 normal points; Clopper--Pearson upper bound 0.000998
- At 2x radius: 706/3,000 normal points (0.23533); upper bound 0.24843, versus the registered maximum 0.01

This multi-scale result prevents a false positive in which an empty tiny core is surrounded by a densely occupied normal neighborhood. The candidate induces cross-position compactness inside empirical support, but not a low-density attractor region as defined by V4.

## Downstream gate status

The final status is:

```json
{
  "encoder_attractor_discovered": false,
  "reason": "No universal validation-certified length; test and retrieval were not opened"
}
```

Consequently:

- no shortest universal candidate was frozen;
- no frozen center or radius exists for a one-time test;
- no triggered IID test or fixed-center test decision was run;
- no triggered OOD evaluation or fixed-center OOD decision was run;
- no poison entry was selected;
- no Top-K, PoisonRank, or retrieval margin was computed.

This preserves the registered separation between encoder discovery and downstream retrieval effects.

The untriggered base embeddings for all roles, including the reserved test and OOD texts, were precomputed once during `prepare` and included in the 15,000 preparation queries. Search and validation load only their registered search/validation roles; the test/OOD base embeddings were not combined with a candidate or consumed by a test decision because the universal gate stayed closed. A future protocol can strengthen operational sealing further by deferring even these untriggered base encodings until the gate opens.

## Integrity and scope audit

- V4-only tests: 9 passed locally and 9 passed on the remote compute host.
- Result completeness audit: 120 summaries and 120 frontier rows.
- SHA-256 manifest: 2,854 experiment artifacts, all verified.
- The manifest file itself and the post-manifest audit log are the only two result-package files intentionally outside the manifest rows.
- Static threat-model audit: no forbidden gradient, parameter, hidden-state, soft-prompt, legacy Mode 3 V3, or retrieval-feedback path.
- Scope audit: all committed changes since the registered V3 baseline are confined to the new V4 config, implementation, scripts, tests, and documents. Modes 1/2 and V1--V3 were not modified or executed.

## Interpretation and next route

The experiment supports a narrower result than the original hypothesis: this encoder admits strong prefix-specific compact attractor behavior at length 10, but the behavior did not generalize to suffix and random insertion, and the cross-position candidates that did compactify representations landed in normally occupied regions at the registered 2x scale.

The next V4 iteration should keep the same one-time-test gate and query-only threat model while treating multi-scale occupancy and position robustness as first-class search objectives rather than late validation filters. Any such iteration must be registered as a new experiment; the completed V4 frontier and reserved test/OOD sets must not be reused for adaptive selection.
