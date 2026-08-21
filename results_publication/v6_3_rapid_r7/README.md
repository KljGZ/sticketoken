# StickyToken V6.3 rapid r7 — audited result package

This directory contains the compact, auditable result set for the completed
Sentence-T5-base rapid positive-track experiment. The full embedding cache and
all large runtime intermediates remain on the experiment host and are not
duplicated in GitHub.

## Scientific identity

- Run ID: `mode3_v6_3_rapid_r7`
- Protocol revision: `7`
- Source commit: `292321238fedff6ed418a55a239bfd86b90d245b`
- Source config SHA-256: `251e72e87147b6205ff5b0fc12315aea8fb07488a8bb41fe5702f6f114af932f`
- Amendment: `V6_3_RAPID_POSITIVE_TRACK_A2_8GPU_HIGH_PRIORITY`
- Model revision: `fc5d4628481afbbaaacd7af6bb07cf9d3865f781`
- Terminal status: `CERTIFIED_ST_RADIAL_SHIFT`

The registered S0 audit reused 32 complete r5 shards with 21,984 unique legal
tokens, no failed shards, no missing token IDs, and no cache corruption. It
selected 200 candidates. FULL refit reduced these to 20; the frozen discovery
set contains one primary and four secondary candidates. Only the primary was
opened for the sealed confirmation.

## Main result

The primary candidate is the single token `vegan` (`token_id=10278`). Its
frozen center is stored verbatim in `freeze/primary.json`; the center, radius,
role hashes, call-space hash, source weights, and position weights are all part
of that frozen record.

The rapid r7 experiment certifies a strong single-token radial shift, but does
not certify an independently selective frozen cap:

| Certification level | Result |
| --- | --- |
| A — ST radial shift | **PASS** |
| B — frozen-cap core | FAIL |
| C — frozen-cap moat | FAIL |
| D — frozen-cap basin | FAIL |
| E — central collapse | FAIL |

Accordingly, `FINAL_STATUS.json` records:

- `answer=false` to the primary frozen-cap question;
- `claim_boundary=radial_shift_only_not_frozen_cap_certified`;
- positive gates for radius, balanced coverage, worst position, and worst source;
- negative gates for independent benign core, outside-to-inside migration, and
  conditional outside-origin migration.

This positive result supports the existence of at least one strong radial-shift
token under this protocol. It does not support a whole-vocabulary negative
claim, a token-specific low-occupancy frozen cap, or central collapse.

## Primary frozen and confirmation statistics

Frozen discovery values (`freeze/primary.json`):

- radius: `0.5623568955206134` rad = `32.22067669341054` degrees;
- balanced triggered coverage: `0.964111111111111`;
- worst-position coverage: `0.9183333333333333`;
- worst-source coverage: `0.9404444444444445`;
- independent benign occupancy inside 1.0R: `0.15386`;
- outside-to-inside estimate: `0.8116111111111112`.

Independent sealed confirmation (`confirm/primary_certificate.json`, 50,000
independent text units):

- balanced triggered coverage CI: `[0.9561989830073826, 0.9685527697951818]`;
- worst-position lower bound: `0.901512882321024`;
- worst-source lower bound: `0.927341461153904`;
- benign 1.0R occupancy: `0.15268` (`22,902 / 150,000`);
- benign balanced occupancy UCB: `0.1568034983735736`;
- outside-to-inside balanced lower bound: `0.792547466142658`;
- conditional outside-origin balanced lower bound: `0.8248054817395715`;
- benign median radial depth: `1.059909730587604`;
- triggered median radial depth: `0.8686199984405769`;
- median shift: `-0.19128973214702705`;
- KS statistic: `0.8687266666666666` (`p=0.0` as numerically reported);
- Wasserstein distance: `0.18815005868163737`;
- Cliff's delta: `-0.9612225642666666`.

## Benign radial occupancy curve

The complete machine-readable curve is in `radial_occupancy_curve.csv` and the
confirmation certificate. For the 0–1.0R range:

| Radius multiplier | Inside count / 150,000 | Occupancy | 95% upper bound |
| ---: | ---: | ---: | ---: |
| 0.1R | 0 | 0 | 0.00001997134906031303 |
| 0.2R | 0 | 0 | 0.00001997134906031303 |
| 0.3R | 0 | 0 | 0.00001997134906031303 |
| 0.4R | 0 | 0 | 0.00001997134906031303 |
| 0.5R | 0 | 0 | 0.00001997134906031303 |
| 0.6R | 0 | 0 | 0.00001997134906031303 |
| 0.7R | 0 | 0 | 0.00001997134906031303 |
| 0.8R | 8 | 0.00005333333333333333 | 0.00009622893406360496 |
| 0.9R | 358 | 0.0023866666666666667 | 0.002604539739472496 |
| 1.0R | 22,902 | 0.15268 | 0.1542158441685143 |

The curve continues through 1.05R, 1.1R, 1.25R, 1.5R, and 2.0R in the CSV.

## Budget

- Planned core-search total: `11,124,000` text equivalents.
- Observed submitted/raw-forward text equivalents: `11,112,000`.
- Observed reservations: `11,106`.
- Warning reached: `false`; hard limit reached: `false`.

See `budget/`, `run_manifest.json`, `orchestration_logs/status.json`, and
`result_inventory.json` for the registered accounting and identity chain.

## Package map

- `FINAL_STATUS.json`: terminal claim boundary and certification levels.
- `freeze/`: frozen primary/secondary candidates and freeze hashes.
- `confirm/`: primary certificate, migration rows, and position audit.
- `radial_occupancy_curve.csv`: 0.1R–2.0R benign occupancy curve.
- `source_position_intervals.csv`: source/position confidence intervals.
- `stages/`: S0, FULL, and TOP100 selections and metrics.
- `S0_REUSE_AUDIT.json`: provenance and integrity of the reused r5 S0 funnel.
- `budget/`: planned and observed model-call accounting.
- `run_manifest.json`, `resolved_config.json`, `sealed/`: experiment identity.
- `result_inventory.json`: complete result inventory from the formal output.
- `PACKAGE_MANIFEST.sha256`: SHA-256 for every published artifact except itself.

Large embedding caches, per-call ledgers, and verbose runtime realization logs
are intentionally excluded from this GitHub package. Their exclusion changes
distribution size, not the registered terminal result or the frozen evidence
included here.
