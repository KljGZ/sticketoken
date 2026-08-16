# V6 Compact certification erratum

The released V6 Compact artifacts remain immutable historical evidence. Their recorded
`negative_endpoint=true` field is not deleted or rewritten, but it must not be interpreted
as evidence that single-token frozen-cap attraction is absent.

Compact calibrated a 90% radius and then required a 95% Clopper-Pearson lower bound to be
strictly above 90% on the same observations. At 16,201/18,000 induced successes the lower
bound is approximately 0.8963, so the registered primary gate was structurally unreachable
at that calibration rule. Radius design and certification also reused the same records.

Other material limitations included position pseudoreplication, averaged random-position
vectors the encoder never emitted, mismatch between the registered and implemented center,
no later-stage refit, a two-cap rescue restricted to the final 20 single-cap candidates,
insufficient token-realization/truncation auditing and incomplete simultaneous inference.

The correct classification is **protocol-inconclusive historical endpoint**. V6.3 changes
the prospective design without modifying old files: independent fit/radius/score/confirm
roles, q=0.92 radius design, one independent confirm position per text, actual non-averaged
random vectors, source-position-balanced inference, stage-local refits, exhaustive runtime
token assertions and a single-cap-only primary protocol.
