# V7 versus V6.3

| Dimension | V6.3 r5/r7 | V7 |
|---|---|---|
| Positions | prefix, suffix, random | prefix and suffix only |
| Center | includes registered random strata | one prefix/suffix shared center |
| Radius | triggered q92 design radius | largest benign-occupancy-UCB-feasible radius per beta |
| Operating points | one cap | immutable 19-beta frontier |
| Coverage | balanced/worst summaries | prefix and suffix LCBs must each reach 0.80 |
| Primary rank | V6.3 metric order | beta80, min LCB, mean coverage, radius, token ID |
| Migration | could enter prior claims | report-only |
| `e*` geometry | follow-up geometry | report-only axis threshold and exclusion margin |
| Confirm | old frozen cap | newly frozen token + beta + center + radius |

V7 reads V6.3 r5 only through a fail-closed audit. It may reuse the exact legal vocabulary,
tokenizer audit, document/source manifests, aggregate S0 metrics, and complete raw
prefix/suffix caches when present. It cannot reuse V6.3 q92 radii, random-influenced centers,
selection order, frozen primary/secondaries, or confirmation. The V6.3 r7 certificate remains
a separate result and is never overwritten or relabeled as V7 evidence.

The observed r5 storage-recovery history makes aggregate-only proposal reuse the expected
route. That route deliberately retains 512 candidates, records the old metrics as
proposal-only, and performs every formal V7 center, radius frontier, and coverage calculation
from scratch.
