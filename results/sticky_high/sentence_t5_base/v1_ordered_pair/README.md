# Sticky-high v1 result (Sentence-T5-base)

This directory contains the first registered sticky-high experiment for
`sentence-transformers/sentence-t5-base`. The target is deliberately different
from a mean-collapsing sticky token: low-similarity pairs should increase while
high-similarity pairs should not materially decrease.

## Registered acceptance criteria

- Low group: baseline cosine similarity <= 0.65.
- High group: baseline cosine similarity >= 0.82.
- The 10th percentile low-group gain after 30 insertions must be >= 0.02.
- The 5th percentile high-group gain must be >= -0.02.
- At most 10% of high-group pairs may fall below the -0.02 tolerance.
- Material per-step drops (> 0.002) may occur in at most 10% of low- and
  high-group curve transitions.

Search, validation, and plotting rows are disjoint. The vocabulary screen uses
8 low and 8 high pairs; validation uses 48 low and 48 high pairs; the final
Figure 2(b)-style plot uses 25 held-out pairs spanning the similarity range.

## Candidate space and result

The experiment screened 31,994 reachable ordinary token strings. It then formed
16,384 ordered two-token literal strings from the top 128 single-token
components selected on the search split, for 48,378 screened candidates in
total. A candidate is evaluated as literal text after concatenation, so its
actual contextual tokenization is part of the measurement.

No candidate met every registered acceptance criterion. The closest candidate
by normalized constraint violation was the literal string ` Ki` (token ID
4320):

- low-group mean gain: +0.033295;
- low-group 10th-percentile gain: +0.011914;
- high-group mean gain: -0.009247;
- high-group 5th-percentile gain: -0.023584;
- high-group tolerance failure rate: 0.083333;
- low/high step failure rates: 0.011111 / 0.060417;
- certified: **false**.

The held-out plot moves from an initial range of 0.507812--0.965820 to a final
range of 0.549316--0.940918. Its median change is +0.006348. This is the desired
directional shape, but it is a near-feasible baseline rather than a certified
sticky-high string.

## Reproduction

The full screen was produced with:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
python scripts/detect_sticky_high.py \
    --model-id sentence-transformers/sentence-t5-base \
    --device cuda:7 \
    --max-components 2 \
    --component-pool-size 128 \
    --batch-size 256 \
    --candidate-chunk-size 64 \
    --validation-chunk-size 8 \
    --no-show-progress
```

The final balanced revalidation used `--shortlist-size 256 --finalist-size 32`
and the audited screen-reuse mechanism recorded in `metadata.json`. The source
screen configuration and split manifest passed all compatibility checks before
reuse.
