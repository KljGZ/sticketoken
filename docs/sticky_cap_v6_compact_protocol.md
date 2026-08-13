# StickyToken Mode 3 V6 Compact protocol

V6 Compact is a preregistered, budget-bounded replacement for the aborted
initial V6 run. It studies only Mode 3. Mode 1, Mode 2, and all V1--V6-heavy
formal code, configuration, and results are immutable baselines.

## Primary question

The primary question is whether an actual tokenizer-length-one token, inserted
once, causes more than 90% of heterogeneous texts to enter a validation-frozen
high-dimensional angular spherical cap while fewer than 1% of independent
benign texts enter the exact core. All centers, angular radii, membership,
coverage, occupancy, and migration statistics are computed in the normalized
source embedding dimension. PCA/UMAP are not certification inputs.

## Compact candidate funnel

- S0 exhaustively evaluates every legal single token on the same 1,024 fit and
  1,024 evaluation texts. It fits one P3 shared cap and retains 3,000 tokens.
- S1 evaluates 3,000 tokens on 2,048 additional texts and retains 600.
- S2 evaluates 600 tokens on 4,096 additional texts and retains 200.
- S3 evaluates 200 tokens on 8,000 additional texts and retains 20.
- Validation re-fits on 6,000 Cap-fit texts and freezes radius on an independent
  6,000 Cap-calibration texts. Only these 20 candidates may attempt a two-cap
  rescue, and only after their one-cap test fails.
- At most one primary and four secondary candidates are frozen. P1 and P2 are
  derived offline from the same P3 position arrays; they are not extra searches.

Selection is deterministic and interleaves coverage, worst-position coverage,
angular radius, benign occupancy, migration, and margin ranks. White-box,
black-box, exhaustive, and V5-history candidates enter only a later bounded
union; all candidates must pass the same exhaustive/progressive evaluations.

## Discovery tracks

The white-box track uses 16 seeds, four restarts, ten HotFlip iterations,
top-64 gradient proposals, top-32 exact forwards, and a beam of 16. Its
continuous-token optimization is a mechanistic upper bound only. The black-box
track is physically isolated, accepts only final embedding outputs, and runs
categorical CEM with population 256, 25 generations, and eight independently
uniform restarts. It cannot accept a white-box seed.

## Data and leakage controls

Whole documents are assigned to roles. Before a document enters any role, a
global LSH index rejects exact, normalized, and verified shingle-Jaccard near
duplicates of all earlier roles. The original independent V6 leakage auditor
must then return zero cross-role pairs. No resampling is allowed. IID test,
replications, and three OOD domains remain sealed until the validation gate
opens.

Token-independent clean and benign embeddings are encoded exactly once per role
and saved with role, text-manifest, model revision, shape, dtype, and SHA-256
bindings. All shards reuse these arrays.

## Query budget

The V5 baseline is 83,605,976 submitted texts. The registered limits are:

- planned: 300,981,514 (3.6 T_V5);
- warning: 351,145,099 (4.2 T_V5);
- hard: 392,948,087 (4.7 T_V5);
- forbidden: 418,029,880 (5.0 T_V5).

The static estimate is 260,573,664 equivalents, or 3.1167 T_V5. Every process
reserves its full forward/backward equivalent under a shared file lock before a
model call. A request crossing the hard limit is rejected and no later model
call is allowed. Backward accounting uses a preregistered conservative gamma
ceiling of 8 and also records a real-machine benchmark.

## Confirmation and downstream evidence

If the gate opens, frozen geometry is applied without refitting to 12,000 IID
test texts, two 5,000-text replications, and three 4,000-text OOD sets. IID
benign occupancy uses 100,000 independent texts; each OOD set uses 20,000.
Triggered, paired-clean, and independent-benign cumulative and shell occupancy
are recorded from 0.1 rho through 2 rho, together with four-way migration and
1,000 bootstrap replicates.

Matched semantic controls, wrapper counterfactuals, additive semantic models,
white-box mechanisms, and controlled one-poison retrieval run only for frozen
finalists. Retrieval never feeds candidate search. If one- and two-cap
validation both fail, sealed Test/OOD remain unencoded and the run ends as a
compliant negative result.

## Process and artifact guarantees

The formal launcher holds `flock`; systemd user service execution uses a single
control group with `KillMode=control-group`, with `setsid` as the documented
fallback. Every stage is idempotent and writes COMPLETE only after its required
artifacts. The full raw directory is content-addressed, published as Release
assets, and must be restored in a fresh clone with per-file SHA-256 equality.
