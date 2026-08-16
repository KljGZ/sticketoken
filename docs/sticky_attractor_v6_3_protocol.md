# StickyToken Mode 3 V6.3 protocol

Status: preregistered implementation specification. The formal run is valid only when its
`run_manifest.json` binds this document, the resolved configuration, code commit, model,
tokenizer, data-role manifest, position manifest and call-space manifest by SHA-256.

Protocol revision 2 binds tokenizer identity with
`sorted_token_id_nul_text_lf_v1`: every complete-vocabulary `(token_id, token_text)` pair
is sorted by integer ID and token text, then hashed as the UTF-8 byte stream
`token_id + NUL + token_text + LF`. Fast-tokenizer backend JSON is recorded only as an
environment diagnostic because its serialization can change across library versions
without changing token identity. Tokenizer, model-manifest, data-manifest and environment
lock identities are checked before corpus registration or sealed-role writes begin.

## Primary question

Does there exist a legal, non-special tokenizer item whose realized contextual length is
exactly one token, such that one insertion attracts heterogeneous source texts into one
shared, frozen angular cap under `sentence-transformers/sentence-t5-base` revision
`fc5d4628481afbbaaacd7af6bb07cf9d3865f781`?

Only P3 ST-FCA-Core is the primary claim. P1/P2 are descriptive diagnostics. Multicap,
CEM, GA, HotFlip, continuous-token search, white-box seeds and historical-candidate quotas
are disabled. All geometry is computed on normalized final embeddings in the original
high-dimensional space.

For a registered insertion position `p`, token `x`, source text `q_i` and encoder `E`,

```text
z_i,p(x) = normalize(E(I_p(q_i, x)))
d_angle(z,c) = arccos(clip(z dot c, -1, 1))
C(c,rho) = {z: d_angle(z,c) <= rho}
```

The P3 cap has one shared robust center and one shared radius across prefix, suffix and
random positions. The center is a 10% source- and position-balanced trimmed spherical
center. Radius data are independent of fit and score data; the frozen design radius is the
0.92 quantile and may not exceed 35 degrees.

## Data and observation units

The fit, radius and score chains are mutually document-disjoint and nested by stage within
their own chain: S0 `1024/512/512`, S1 `1536/768/1024`, S2 `3072/1536/2048`, Full
`8000/4000/6000`. Discovery benign has 50,000 independent records. Confirm, paired audit,
semantic, IID, OOD and retrieval roles are physically sealed until a hash-bound freeze is
written and a separate access grant is created.

S0/S1 use a deterministic source-balanced 1-of-3 incomplete block. S2/Full use a
deterministic source-balanced 2-of-3 block. The top 100 are re-evaluated at all three
positions before freeze. Confirm assigns exactly one position to each independent text;
only the separate paired audit evaluates all positions for the same text. Random-position
vectors are never averaged.

Every source is pre-truncated before insertion. Every realized insertion must contain the
registered target token exactly once, with a one-token offset span that remains attended.
The same pre-truncated source is used for its clean and triggered pair. A candidate-level
realization or radius rejection is allowed; any unexpected exception fails the whole shard.

## Candidate funnel

The exact legal vocabulary enters S0. Deterministic retention is:

```text
all legal -> 12,000 -> 8,000 -> 5,000 -> 100 -> primary 1 + secondary 4
```

Each stage refits center and radius from scratch on its registered data while reusing only
previously computed exact embedding calls. Retention is 70% Pareto/composite, 20%
near-threshold uncertainty, 5% diversity and 5% deterministic random audit. No history
quota exists. All candidates, rejections and selection reasons are retained.

## Freeze and independent confirmation

Primary and four secondary candidates are selected from the complete-position top-100
archive before sealed data can be read. `confirm.py` consumes only the hash-verified freeze
artifact and cannot import fit functions. It may not refit, resize or select a cap.

The primary confirmation roles contain 50,000 trigger texts and 150,000 benign texts.
Familywise alpha is 0.05 with Bonferroni correction across observed source-position cells.
ST-FCA-Core requires all of:

- source-balanced trigger coverage LCB > 0.90;
- worst-position coverage LCB > 0.85;
- worst-source coverage LCB > 0.80;
- source-balanced independent benign core occupancy UCB < 0.01;
- clean-outside to triggered-inside migration LCB >= 0.85;
- clean-outside conditional on triggered-inside LCB >= 0.95;
- frozen radius <= 35 degrees.

Levels are reported independently: A radial shift, B Core, C Moat, D Basin and E central
collapse. Semantic matched controls, three IID replications, four OOD domains and a real
single-poison retrieval evaluation run only after Level B passes. Their results can extend
the evidence chain but cannot retroactively alter discovery or the frozen primary.

## Budget, devices and stopping

One budget unit is one V5 submitted-text equivalent; the V5 baseline is 83,605,976. The
core target is 3.2x, the registered complete plan is at most 4.5x, warning is 5.5x and no
new model call may begin at 6.5x. Fifteen times V5 is a forbidden ceiling, not a target.
Budget is reserved before each model call and duplicate exact calls are protocol errors.

Only physical GPUs 4, 5, 6 and 7 are authorized. Physical GPUs 0, 1, 2 and 3 are hard
forbidden. Each worker sees exactly one authorized physical GPU through
`CUDA_VISIBLE_DEVICES` and uses logical `cuda:0`. Lack of safe capacity causes waiting or
reduced concurrency; it never authorizes fallback or interference with another process.

A run with data/hash/role/protocol corruption has no scientific endpoint. A valid search
whose frozen primary fails confirm is reported as `VALID_PRIMARY_NOT_CERTIFIED`, not as a
universal absence claim. Only independent formal confirmation can issue a V6.3 certificate.
