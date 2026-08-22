# V7 power and budget analysis

With four balanced IID sources, the formal 50,000-row calibration role has about 12,500 benign
observations per source. At zero observed events, a one-sided Clopper-Pearson bound with the
registered source correction is well below the smallest 0.1% beta, so that endpoint is
statistically resolvable. Independent confirmation increases this to about 37,500 benign
observations per source. The 25,000 rows assigned separately to prefix and suffix provide about
6,250 observations per source-position cell, making an 80% lower-bound gate estimable without
pooling the two positions.

For the expected aggregate fallback of 512 FULL candidates, the registered model-call budget
is:

| Component | Submitted-text equivalents |
|---|---:|
| 512 × 2 × (8,000 fit + 6,000 select) | 14,336,000 |
| shared discovery clean calls | 106,000 |
| primary confirmation and paired audit | 256,000 |
| total | 14,698,000 |

This is about 17.6% of the V5 baseline (83,605,976), below the warning limit 29,262,092 and
hard reservation limit 41,802,988. The complete-raw-cache route retains 256 candidates and is
smaller. Center bootstrap is report-only and runs only after top-20 selection, so it adds no
model calls and cannot affect multiplicity or ranking.

At 768 float32 coordinates per embedding, the aggregate plan stores about 45.2 GB of raw
vectors at peak. The registered estimate remains 50 GB and its 1.35 multiplier remains a
67.5 GB reference. V7 r3 applies an explicitly authorized 10 GB registration/model-start
gate instead. Ten GB is not a capacity guarantee, so free space remains a monitored runtime
risk. Non-selected V7 caches are compacted only after the all-candidate frontier and top-20
diagnostics are durable.

These calculations are design checks, not a guarantee of a positive result. Confirmation can
validly fail either coverage position or occupancy even when discovery passes.
