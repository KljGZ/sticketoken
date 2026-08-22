# V7 data contract

V7 uses document-disjoint, near-duplicate-screened, source-balanced roles. S0 is a nested view
of FULL and consumes no additional observation units.

| Role | Formal rows | Use | Trigger positions |
|---|---:|---|---|
| fit | 8,000 | shared-center fitting | prefix + suffix |
| calibration | 50,000 | benign occupancy radius | clean only |
| select | 6,000 | coverage/ranking and paired migration | clean + prefix + suffix |
| axis_fit_benign | 50,000 | independent `e*` diagnostic | clean only |
| confirm_prefix | 25,000 | independent prefix gate | clean + prefix |
| confirm_suffix | 25,000 | independent suffix gate | clean + suffix |
| confirm_benign | 150,000 | independent occupancy gate | clean only |
| confirm_paired | 2,000 | report-only position agreement | clean + prefix + suffix |

The formal registry therefore needs 316,000 unique IID text/document units before
near-duplicate rejection. Fit, calibration, select, axis, and every confirm role are mutually
document-disjoint. Prefix and suffix confirmation use different texts; only `confirm_paired`
uses both positions for one text.

Discovery roles are written into the run output. Confirmation roles are written to a separate
sealed directory, hashed into `SEALED_INVENTORY.json`, and made unreadable before freeze on the
Linux formal host. A grant bound to the primary freeze and role-manifest hashes is required to
reopen them. Any confirm cache before freeze, role-hash collision, or changed sealed file is a
protocol failure.

The registered call space contains only realizable clean/prefix/suffix calls. Token IDs are
specialized into immutable cache keys at runtime. Every new call is reserved in the durable
budget ledger before the model executes; duplicate or unregistered calls fail closed.
