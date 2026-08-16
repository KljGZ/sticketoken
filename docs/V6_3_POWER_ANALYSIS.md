# V6.3 power and reachability analysis

V6 Compact calibrated a 90% radius and applied a strict 95% lower-bound-above-90% gate to
the same observations. That design was structurally self-blocking. V6.3 separates radius
design from confirmation and calibrates at 0.92.

For the 50,000-text formal confirm role, a one-sided 95% Clopper-Pearson lower bound is
strictly above 0.90 beginning at 45,111 observed successes. The registered synthetic tests
therefore check that a 0.90 truth does not systematically pass, a 0.92 truth has useful
power, and the fixed-cap gate is reachable. Coverage is not pooled across repeated
positions: each confirm text contributes one independent Bernoulli observation.

Inference is source-position stratified. Per-cell Clopper-Pearson bounds use Bonferroni
alpha `0.05 / (3S)`, where `S` is the number of observed sources. Balanced aggregates give
equal weight to sources and positions rather than letting the largest source dominate.
Worst-source and worst-position bounds are separate registered gates.

The benign role has 150,000 independent observations. With zero observed core hits its
one-sided upper bound is far below 1%; this size also preserves power under balanced source
aggregation. Migration and conditional-origin gates use the same independent confirm texts
and paired clean embeddings, never the radius-calibration observations.

This analysis guarantees reachability and registered operating characteristics, not a
positive scientific result. Passing depends on the actual frozen candidate.
