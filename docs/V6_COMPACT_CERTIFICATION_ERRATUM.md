# V6 Compact certification erratum

The released V6 Compact artifacts are retained as immutable historical evidence. Their recorded `negative_endpoint=true` field is not deleted or rewritten. However, that endpoint must not be interpreted as evidence that single-token frozen-cap attraction is absent.

The Compact protocol calibrated a 90% cap radius and then asked a 95% Clopper–Pearson lower bound to be strictly above 90% on the same observations. At its registered sample size, the mechanically induced success count (16,201/18,000) has a lower bound of approximately 0.8963, so the primary gate was structurally unreachable at the calibration rule. The same records were also used for radius calibration and certification.

Additional material defects affect the strength of the old endpoint: P3 treated three positions of the same source text as independent trials; random-position replicate vectors were averaged into a vector that the encoder never emitted; the implemented center differed from the registered robust center; later funnel stages scored inherited caps instead of refitting; multicap models were available only as a rescue for the last 20 single-cap finalists; token realization and truncation audits were too small; and P1/P2/P3 did not receive genuinely separate simultaneous certificates.

Therefore the Compact result is reclassified as:

> **Protocol-inconclusive historical endpoint.** The artifacts remain valid records of what the Compact implementation executed, but `0 certified` is not a scientifically valid negative result for ST-FCA.

V6.2 supersedes only the interpretation, not the files. V6.2 uses independent `fit -> radius -> select -> confirm` roles, a 0.92 design radius, source-by-position simultaneous inference, actual non-averaged random replicates, stage-local refitting, independent one/multicap archives, exhaustive runtime token assertions, and separate P1/P2/P3 freeze objects.
