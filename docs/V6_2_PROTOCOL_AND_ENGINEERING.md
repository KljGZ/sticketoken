# StickyToken Mode 3 V6.2 protocol and engineering contract

## Status and claim boundary

V6.2 is a new experiment namespace. V6 Compact code, results, and release artifacts remain read-only history. V6.2 can guarantee a valid protocol, implementation/data separation, reproducibility, and correctly bounded conclusions; it cannot guarantee that a positive token exists.

The primary claim, ST-FCA-Core, is that one real nonspecial tokenizer-length-one token, inserted exactly once, attracts heterogeneous normalized encoder representations into one immutable high-dimensional angular cap. Two-, three-, and four-cap results are secondary ST-mFCA evidence and are never relabeled as a one-cap result.

## Irreversible role graph

The experiment uses the following one-way graph:

`D_fit -> center; D_radius -> radius; D_select -> candidate selection; freeze -> D_confirm certificate`.

No confirmation module imports a fitting or calibration function. The role contract binds every ordered record list, source set, document count, code commit, model revision, tokenizer vocabulary, configuration, random-boundary manifest, and freeze artifact with SHA-256. Sealed roles cannot be encoded before `freezes/INDEX.json` exists.

The formal sizes are:

| Stage | Fit | Radius | Score/Select |
|---|---:|---:|---:|
| S0 | 1,024 | 512 | 512 |
| S1 | 1,536 | 768 | 1,024 |
| S2 | 3,072 | 1,536 | 2,048 |
| Full | 8,000 | 4,000 | 6,000 |

The funnel is `complete legal vocabulary -> 12,000 -> 8,000 -> 5,000 -> 2,000 stability replay -> 100 semantic candidates -> 1 primary + 4 secondary P3 freezes`. Every stage refits centers and radii from scratch. The 2,000-candidate replay repeats the full registered role evaluation and makes the replay metrics authoritative; earlier values remain diagnostic history.

Confirmation contains 50,000 triggered and 150,000 independent benign records. Three IID replications contain 20,000 records each. Four source-isolated OOD domains contain 15,000 triggered and 30,000 benign records per domain. A separate 20,000-record role is reserved for retrieval. Semantic discovery and semantic confirmation use disjoint 3,000-record roles.

## Token and truncation contract

Enumeration audits every candidate across exactly 2,048 stratified S0 contexts and prefix, suffix, and the primary random boundary. The vocabulary is partitioned into 32 disjoint CPU shards; context pretruncation is computed once, tokenizer calls are made in bounded batches, and the merge independently reconstructs the complete standalone-roundtrip candidate set before accepting the union. This changes no audited observation. Each formal encoding repeats the assertion for every record:

1. the trigger overlaps exactly one no-special token;
2. that token has the registered ID;
3. removing it leaves the exact pretruncated source token IDs;
4. the trigger has attention value one;
5. the final sequence does not exceed the model maximum.

Source text is pretruncated once with one trigger-token slot reserved. Clean and triggered paths share those retained source tokens. Failures reject only that candidate; role mismatch, cache corruption, nonfinite output, budget failure, or shape mismatch aborts the entire shard.

Random replicate zero is the primary actual encoder observation. Replicates one through four are robustness observations. Embedding vectors are never averaged.

## Geometry

All formal operations use angular distance in the original normalized embedding dimension. The single-cap center is a 10%-trimmed spherical estimator with equal total weight per source-by-position stratum, at least 20 deterministic/random restarts, and lexicographic optimization of worst-stratum q90 then mean-stratum q90. Restart traces and inlier indices are saved.

The radius uses an independent role and a 0.92 design quantile. The shared P3 rule takes the maximum of equal-source position quantiles. A radius above 35 degrees is rejected. P1 uses independent tokens and position-specific caps; P2 uses one token and three position-specific caps; P3 uses one token, one center set, and one radius set across positions. They are selected, frozen, and confirmed as separate objects.

Multicap counts 2, 3, and 4 form an independent archive beginning at S0. Each cluster needs at least 10% global mass and 5% mass in every source-by-position stratum. Minimal complexity is preferred, and drift, cluster mass, assignments, union occupancy, and radius traces are retained.

## Statistical certificates

Confirmation computes Clopper–Pearson bounds separately for every source-by-position stratum using Bonferroni familywise alpha `0.05/(3S)`. Text positions are never treated as independent repetitions of the same observation.

The P3 primary gates are:

- balanced coverage LCB `> 0.90`;
- worst-position LCB `> 0.85`;
- worst-source LCB `> 0.80`;
- worst benign-source occupancy UCB `< 0.01`;
- outside-to-inside LCB `>= 0.85`;
- conditional original-outside LCB `>= 0.95`.

Secondary uniform P3 evidence additionally requires the minimum source-by-position LCB `> 0.90`. P1 and P2 apply simultaneous correction across positions and within-position sources; their position gates include balanced coverage, worst-source coverage, independent benign occupancy, outside-to-inside migration, and conditional outside-origin migration.

Evidence is reported in levels: A radial shift; B ST-FCA-Core; C Moat with benign occupancy UCB at `1.1 rho < 0.05`; D Basin with `lambda* >= 1.5` and occupancy AUC on `[1,1.5] <= 0.03`; E Central Collapse with median triggered normalized depth `<= 0.80`.

## Semantic controls

The top 100 token candidates are matched to 50 controls on frequency, IDF, POS, semantic category, character length, casing, input-embedding norm, naturalness, leading-space pattern, and Unicode/language class. Encoded candidate, control, and wrapper vectors are reused without extra model calls to compute distinct evidence for every exact `(token_id, cap_count)` model; evidence from one cap complexity can never rank or freeze another. The discovery role may influence selection. After freezing, the same declared procedure is repeated on the independent `semantic_confirm` role and cannot modify a cap. Required anomaly evidence is candidate coverage minus the control q95 at least 0.10 and minimum wrapper coverage at least 0.80.

## Budget and stopping

V5 submitted-text baseline is 83,605,976. V6.2 registers 1,008,791,696 equivalents (about 12.07 V5), a 12.5 V5 planned threshold, 13.5 warning, 14.7 hard stop, and an unreachable 15.0 forbidden threshold. The shared ledger reserves a complete call before model execution and never refunds failed calls. If pressure arises, isolated black/white-box diagnostics and auxiliary encoders are reduced first. Role independence, the common 5,000 evaluation, confirmation, sealing, and semantic controls cannot be cut.

## Formal execution

`scripts/run_v6_2_mode3_remote.sh` is idempotent at completed-stage boundaries, holds a single orchestration lock, writes an atomic state heartbeat, and runs 32 token shards over eight GPUs. A formal run requires a clean Git commit and a corpus manifest hash equal to the configuration. `scripts/status_v6_2_mode3.py` is read-only. `scripts/stop_v6_2_mode3_remote.sh` targets only the recorded process group.

Before the real-model dry run, `scripts/audit_v6_2_corpus.py` executes the complete formal 546,032-record role allocation without encoding, including exact per-IID-source quotas, whole-document disjointness, online verified LSH near-duplicate rejection, all OOD domain capacities, and an independent post-allocation leakage audit. Its result is retained outside the formal evidence directory; the formal `prepare` step repeats the same allocation and binds it into the run contract.

The corpus contains exactly four genuine IID sources and four source-isolated OOD datasets. `scripts/build_v6_2_corpus.py` preserves the registered V6 DBpedia derivative byte-for-byte, adds 100,000 deterministically selected records from each of the machine's existing MS MARCO, HotpotQA, and Natural Questions raw snapshots, and rebuilds every OOD derivative from the complete V6-manifest-bound Parquet snapshot. This is necessary because the historical 30,000-row OOD derivatives cannot supply disjoint 15,000-trigger and 30,000-benign roles. Every IID role is allocated with exact per-source quotas (difference at most one); every OOD domain remains bound to one isolated source. The manifest binds the clean builder commit, builder-script SHA-256, raw SHA-256 values, and all derivative SHA-256 values. A separately executed full reread binds all raw/derivative hashes, row counts, global normalized uniqueness, document-identity uniqueness, and source/domain identities in `corpus_independent_audit.json`; the configuration pins both artifacts. Source diversity is never manufactured by relabeling one dataset.
