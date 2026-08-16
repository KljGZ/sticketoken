# V6.3 budget plan

Budget unit: V5 submitted-text equivalent. Baseline: 83,605,976.

| Gate | Ratio | Units |
|---|---:|---:|
| Core target | 3.2x | 267,539,123 |
| Complete planned maximum | 4.5x | 376,226,892 |
| Warning | 5.5x | 459,832,868 |
| Hard stop for new calls | 6.5x | 543,438,844 |
| Forbidden absolute ceiling | 15x | 1,254,089,640 |

The registered complete estimate is 374,470,000 units. The ledger reserves the full batch
before model execution and keeps conservative reservations after worker failure. Reusing a
cache hit costs zero; attempting the same candidate-text-position-boundary call twice is a
protocol error. Budget limits cannot be bypassed by changing batch size, shard count or
restart policy.

The main savings relative to V6.2 come from balanced incomplete position blocks and exact
cross-stage cache reuse. They do not change the registered data chains, stage-local refits,
top-100 complete-position evaluation or independent confirm sample sizes.
