# StickyToken V6.3 Rapid Positive Track (r7 high-priority 8-GPU amendment)

## Registered identity

- Protocol revision: `7`
- Run ID: `mode3_v6_3_rapid_r7`
- Amendment: `V6_3_RAPID_POSITIVE_TRACK_A2_8GPU_HIGH_PRIORITY`
- Output leaf: `mode3_v6_3_rapid_r7`
- Source discovery run: `mode3_v6_3_light_r5`
- Source code commit: `6f9195aa747434fc595468dd7d8dddd727c56967`
- Source config SHA-256: `aa5294c3852fbf134ed30a73f7f16e738acb612cfbaeeab491904f27982194c1`

This amendment is an independently registered positive-existence route. It
does not overwrite r5 and it does not replace the r5 exhaustive funnel. It may
share GPUs with r5 at higher admission priority, but it never sends a signal to
r5 or any other process. Its valid positive claim is limited to a frozen primary
token that passes the existing sealed independent confirmation gate. Failure of r7 does
not support a negative claim over all 21,984 legal tokens.

## Handoff and reuse gate

r7 may start only after the r5 merged S0 `COMPLETE.json` exists, all 32 r5 S0
shards are complete, no S0 `FAILED.json` exists, and merged metrics plus
candidate-level rejections cover exactly 21,984 unique legal tokens. The import
also requires exact agreement on source run ID, source commit, source config
file hash, model revision, tokenizer hash, and data-role manifest hash.

Only discovery artifacts are reused: the legal-token enumeration, tokenizer
audit, S0 metrics, S0 candidate rejections, and the S0 fitted models needed to
construct a deterministic 200-token shortlist. S0 centers and radii are not
certification artifacts and are never reused as FULL fits or radii.

## Frozen rapid funnel

1. `21,984 -> 200`: deterministically select from the completed r5 S0 results.
   The quotas are Pareto composite 120, worst-position coverage 20,
   lowest-occupancy 20, migration 15, compact radius 10, stability 10, and
   deterministic random audit 5. Overlap is filled deterministically from the
   source Pareto order.
2. `200 -> 20`: refit every candidate from scratch on the registered FULL views
   (`fit=8,000`, `radius=4,000`, `score=6,000`) at prefix, suffix, and one
   candidate-independent random insertion position. The single robust cap uses
   source-equal and position-equal weighting, the 0.92 design radius quantile,
   and a maximum radius of 35 degrees.
3. `20 -> 1`: complete the all-position discovery evaluation, rank only on
   discovery data, and freeze one primary plus four diagnostic secondaries.
4. Confirm the one frozen primary on sealed independent roles. The primary
   confirmation roles and thresholds are unchanged from r5: 50,000 trigger
   texts, 150,000 benign texts, the registered paired-position audit, and the
   familywise-corrected `P3_ST_FCA_CORE` criteria.
5. Run semantic, IID, OOD, and retrieval follow-ups only if the core certificate
   passes, exactly as in r5.

The top-20 discovery comparison is not an independent confirm-A screen. Only
the already frozen primary is exposed to sealed confirmation data.

## Resource and coexistence policy

- r7 requests physical GPUs 0-7 with exactly one model replica per GPU.
- A worker starts only with at least 12,288 MiB free and keeps an 8,192 MiB
  runtime reserve.
- r7 has higher admission priority than r5: it does not treat r5 as a blocking
  priority peer and does not yield merely because an r5 worker is present.
- Higher priority is non-preemptive. r7 never stops, pauses, modifies, or sends
  a signal to r5 or another user's process. Existing processes may continue to
  share compute and memory.
- `waiting_gpu` due to a failed memory reserve is a normal state and is not
  restart evidence. No automatic replay occurs after an unexplained failure or
  yield timeout.
- Eight-way launch is conditional on every GPU meeting the registered memory
  reserve. Sharing can reduce other jobs' throughput, so zero-impact operation
  is not claimed.

## Budget and interpretation

The registered core estimate is 11,124,000 V5 submitted-text equivalents:
10,800,000 for 200 FULL candidates at three positions, 68,000 for the discovery
clean base, 100,000 for primary trigger/clean confirmation, 150,000 independent
benign texts, and 6,000 paired-position texts. S0 reuse consumes no new encoder
calls. The run stops new calls at 0.50x V5 and has an absolute 1.00x V5 ceiling.

A passing independent certificate supports existence of at least one registered
single-token frozen-cap attractor under this model, dataset, insertion protocol,
and thresholds. A non-pass supports only that the rapid shortlist and its one
frozen primary did not certify; it cannot exclude attractors outside the
shortlist or establish a vocabulary-wide negative result.
