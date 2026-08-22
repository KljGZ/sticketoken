# V7 occupancy-constrained radius policy

For a fixed candidate center, calibration distances are split by source. At a proposed closed
radius, each source receives a one-sided Clopper-Pearson occupancy UCB with Bonferroni alpha
`0.05 / number_of_sources`; the source-balanced UCB is the arithmetic mean of those bounds.

For every registered beta independently, V7 chooses the largest radius among the exact benign
distance boundaries (plus the just-below-first-event boundary and the 35-degree cap) for which
the source-balanced UCB is no greater than beta. Binary search is valid because occupancy and
its UCB are monotone in radius. The resulting radii must also be monotone in beta.

The center is identical across all 19 betas. Calibration uses no triggered select or confirm
data. Selection then evaluates prefix and suffix coverage at each frozen candidate radius.
`beta80_ps` is the first beta where both separate LCBs reach 0.80. A later independent confirm
reuses the chosen center, beta, and radius byte-for-byte and recomputes only the three primary
statistics.

The 35-degree cap is a hard geometric maximum, not a target. V6.3 q92 is emitted as a legacy
comparison only; it is never considered by the V7 radius selector, ranking, or certificate.
