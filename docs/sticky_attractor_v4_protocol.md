# Mode 3 V4 registered protocol

V4 studies one object only: the shortest actual-tokenizer-length, once-inserted discrete trigger that maps heterogeneous texts into one compact, low-normal-occupancy representation region whose fixed center remains inside empirical benign support.

## Frozen scope

- V1–V3 and modes 1–2 are immutable and are neither imported nor executed.
- V4 lives only in `sticky_lab/mode3_v4`, `configs/v4_mode3.yaml`, V4-named tests, scripts, documents, and result directories.
- Search receives only normalized final text embeddings from `encode(texts)`. It cannot access parameters, gradients, input embeddings, hidden states, attention, soft prompts, HotFlip, or retrieval feedback.
- The tokenizer is an independent discrete candidate constructor and exact actual-length/round-trip auditor.

## Registered search

Every legal non-special exact-round-trip single token is exhaustively screened. Actual lengths 2 through 30 each receive independent categorical CEM searches, starting from fresh uniform distributions with two registered restarts. No length uses a shorter-length warm start or a V1–V3 candidate. Prefix, suffix, deterministic random-boundary, and one shared-center universal-position tasks are all run. A trigger literal is inserted exactly once.

All 30 lengths finish search and validation. Early stopping is forbidden. Each length is compared with 64 independently sampled same-length random triggers. Only the shortest universal validation-certified trigger may open the one-time test.

## V4 certificate

Validation uses disjoint trigger and benign roles. The adaptive validation center is the normalized spherical mean pooled across registered positions. The certificate is the conjunction of:

1. exact token round-trip, actual registered length, non-special IDs, and 100% audited context realizability;
2. displacement q05 lower confidence bound at least 0.02;
3. radius q95 upper confidence bound at most 0.40;
4. pairwise-distance contraction q95 upper confidence bound at most 0.60;
5. center 10-NN distance no larger than the search-benign leave-one-out 10-NN q99 threshold; the trigger radius is not subtracted from this support margin;
6. one-sided 95% normal-occupancy upper bounds at radius and twice radius of at most 0.001 and 0.01, plus radius occupancy at or below the fifth percentile of benign reference-center occupancies;
7. composite validation score above the q99 same-length random baseline.

K-means envelopes are diagnostics only. Support escape, sample blank, cluster blank, density blank, and global linear separation are not V4 success conditions. A small number of benign points inside the trigger region is allowed.

## Frozen test and retrieval gate

The validation center and q95 radius are serialized and SHA-256 hashed. Test and OOD evaluations use those exact values and never refit either quantity. Pooled and every-position fixed-region coverage require a one-sided 95% lower confidence bound of at least 0.90. IID test additionally repeats support/low-occupancy checks; OOD verifies fixed-region displacement, contraction, support, and coverage.

Only after validation, IID test, and OOD fixed-center certification may one actual triggered validation text be selected as the poison entry. It is the discrete validation medoid under the frozen center. PoisonRank, Hit@K, retrieval margin, clean false activation, and clean Top-K retention are downstream measurements only and never feed search or length selection.
