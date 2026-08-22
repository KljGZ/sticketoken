# StickyToken Mode 3 V7 protocol

Status: preregistered implementation specification. A formal result is valid only when
`V7_PROTOCOL_LOCK.json`, the resolved configuration, code commit, model/tokenizer identity,
data-role manifest, call-space manifest, and immutable freeze hashes all agree.

## Primary question

Does one legal tokenizer item, inserted exactly once, attract heterogeneous source texts at
both the prefix and suffix into one shared angular cap whose independent benign occupancy is
bounded by a registered budget? V7 calls this prefix/suffix occupancy-constrained single-token
frozen-cap attraction (`PS_OC_ST_FCA`). It uses normalized final embeddings in the original
high-dimensional space.

For token `x`, source text `q_i`, position `p`, encoder `E`, center `c`, and radius `rho`:

```text
z_i,p(x) = normalize(E(I_p(q_i, x)))
d_angle(z,c) = arccos(clip(z dot c, -1, 1))
C(c,rho) = {z: d_angle(z,c) <= rho}
```

Only `prefix` and `suffix` are registered. Random insertion, multicap, CEM, genetic search,
HotFlip, continuous-token search, and beta-specific center refits are forbidden. Each source
is pre-truncated before insertion; runtime offsets must prove that the target appears exactly
once as one attended tokenizer item.

## Shared center and occupancy frontier

Each candidate receives one triggered-only center shared by prefix and suffix. Fitting gives
equal total mass to every source-position stratum and trims the farthest 10% within each
stratum. The center is fitted once per candidate and cannot change across occupancy budgets.

The immutable occupancy grid is:

```text
0.1%, 0.3%, 0.5%, 0.8%, 1%, 2%, 3%, 4%, 5%, 6%, 7%, 8%, 9%,
10%, 11%, 12%, 13%, 14%, 15%
```

For each beta, calibration selects the largest closed angular radius no greater than 35
degrees whose source-balanced one-sided benign occupancy UCB is at most beta. The V6.3 q92
radius is retained only as a labeled diagnostic and can never select a V7 operating point.

On the disjoint select role, prefix and suffix each receive a source-balanced one-sided 95%
coverage LCB. They are separate gates and may not be averaged. `beta80_ps` is the smallest
registered beta where both LCBs are at least 0.80. The secondary curve metric is trapezoidal
AUC over log(beta) of `min(prefix LCB, suffix LCB)`.

Evidence grades are beta80 at or below 1%, at or below 3%, at or below 5%, above 5% through
15%, or no feasible PS-80 point. Migration statistics are report-only: capture conditional on
clean-outside, outside-to-inside mass, conditional clean-outside origin, inside retention, and
net gain. The independent benign direction `e*`, center-to-axis angle, exclusion margin, and
`beta_axis` are also report-only. Neither migration nor axis geometry enters ranking.

## Registered flow

1. Lock code, config, model, tokenizer, corpus, roles, call space, and this protocol.
2. Audit V6.3 r5 S0 read-only. Legal tokens, manifests, and aggregates may be reused. Exact
   prefix/suffix caches are reusable only if every required ordinal and identity is complete.
   Old q92 radii, random-influenced centers, rankings, and confirmation are never formal V7.
3. If exact raw reuse is complete, retain 256 FULL proposals; otherwise use a deliberately
   wide 512-token aggregate fallback. In either route every retained token is refitted and
   rescored from scratch under V7.
4. V7 r3 has registered scheduling priority over V6.3 r5. After V7 registration and the
   read-only S0 reuse audit, V7 freezes only the exact identity-checked r5 orchestrator and
   writes r5's registered cooperative-yield requests for its current workers. Each r5 worker
   must leave at its next durable cache boundary; hard termination is forbidden. If every
   worker does not yield within 300 seconds, V7 resumes r5 and fails closed. Once acquired,
   V7 uses only physical GPUs 4, 5, 6, and 7; GPUs 0 through 3 remain forbidden. V7 resumes
   the exact r5 orchestrator only after `V7_FINAL_STATUS.json` records a terminal endpoint.
   V7 r3 keeps the explicitly operator-authorized 10 GB free-space gate before registration,
   before every new model-work launch, and during FULL scheduling. The original 67.5 GB peak
   reference remains a diagnostic; lowering the gate changes no scientific rule and is not a
   capacity guarantee.
5. Precompute shared clean calibration/select vectors and fit independent `e*`.
6. Evaluate every retained token on FULL roles: fit 8,000, benign calibration 50,000, and
   paired select 6,000. Merge all 32 shards or fail closed.
7. Rank token-beta pairs by minimum beta80, maximum minimum-position LCB, maximum mean
   position coverage, minimum radius, then minimum token ID. Retain the top 20.
8. Run center-drift and `e*` displacement diagnostics after selection. These cannot change
   the selected set. After the diagnostics are durable, remove only V7's non-selected raw
   embedding caches while retaining every FULL frontier/rejection/audit and the top-20 caches.
   Freeze one primary and four secondaries before confirmation is opened.
9. Confirm only the frozen primary with 25,000 independent prefix texts, 25,000 independent
   suffix texts, and 150,000 independent benign texts. A separate 2,000-text paired
   prefix/suffix role is diagnostic and adds no unit to the primary tests. No refit, reselection,
   radius change, beta change, or secondary substitution is allowed.

## Confirmation and terminal states

The primary is certified only if the independent source-balanced benign occupancy UCB is at
most the frozen beta, prefix coverage LCB is at least 0.80, suffix coverage LCB is at least
0.80, radius is at most 35 degrees, and all identity/isolation/no-refit gates pass.

`CERTIFIED_V7_OCFCA_80` is the positive endpoint. A frozen primary that misses any independent
gate is `VALID_PRIMARY_NOT_CERTIFIED`. If fewer than five FULL candidates have a PS-80 point,
the sealed confirm roles remain unopened and the endpoint is
`VALID_NO_OCCUPANCY_FEASIBLE_CANDIDATE`. Hash, role, budget, tokenizer, model, or shard
corruption has no scientific endpoint and must fail closed.

## Required artifacts

The durable output includes `V7_PROTOCOL.md`, `V7_PROTOCOL_LOCK.json`,
`V7_S0_REUSE_AUDIT.json`, `V7_OCCUPANCY_GRID.json`,
`v7_s0_frontier_all_tokens.parquet`, `v7_full_frontier_all_candidates.parquet`,
`v7_top20_token_beta_pairs.json`, occupancy/migration/axis/center-drift diagnostics,
`v7_primary_freeze.json`, `v7_secondary_freeze.jsonl`, `v7_confirm_certificate.json`,
`V7_FINAL_STATUS.json`, and a hash inventory. An aggregate-only S0 table is explicitly marked
proposal-only and is not represented as a formal V7 frontier.
