# Mode 3 V6 result status

## Current status

Implementation preflight is complete, but the repository's current corpus correctly fails the formal V6 data contract. No V6 model query, token search, validation, Test, OOD, or retrieval result is claimed.

Observed local corpus audit:

- 40 CSV files and 40,000 rows;
- 18,753 exact unique text strings and 18,616 normalized unique strings;
- columns only `sentence1`, `sentence2`, and `similarity`;
- zero registered real document IDs, source IDs, or domains;
- V6 preregistered disjoint capacity requirement: 640,000 unique texts;
- at least four source-isolated OOD domains required.

The formal entry point exits before model loading with four explicit gaps: missing canonical columns, insufficient unique texts, insufficient sources, and missing document identity. It does not use repeated sampling or sentence-as-document fallback.

After a compliant corpus is registered, this file must be generated from the finalized result inventory and report P1, P2, and P3 separately; one-cap ST-FCA and multi-cap ST-mFCA must never be combined.
