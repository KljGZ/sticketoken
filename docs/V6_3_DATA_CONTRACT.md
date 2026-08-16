# V6.3 data contract

The formal corpus must contain `text`, `document_id`, `source_id`, `domain`, `language`,
`text_type` and `license`. Sentence-as-document fallback and resampling are prohibited.
At least four IID sources and four allowlisted OOD domains are required.

Before any model load, preflight must verify:

- the registered corpus manifest and independent audit SHA-256 values;
- exact, normalized and near-duplicate exclusions across roles;
- document-disjoint fit, radius, score, benign and sealed roles;
- minimum role capacities and source/domain balance;
- deterministic role allocation and nested search views;
- model, tokenizer and resource checksums;
- sufficient free storage for 1.5 times the estimated peak cache;
- physical sealing of all confirm and follow-up role files.

Formal role sizes are:

| Role | Records |
|---|---:|
| Discovery benign | 50,000 |
| Confirm trigger | 50,000 |
| Confirm benign | 150,000 |
| Paired three-position audit | 2,000 |
| Semantic controls | 10,000 |
| IID replication 0/1/2 | 20,000 each |
| Retrieval probes | 20,000 |
| OOD trigger | 15,000 per domain |
| OOD benign | 30,000 per domain |

The four OOD domains are `ood_news`, `ood_movie_reviews`, `ood_business_reviews` and
`ood_product_reviews`. Role membership is bound by normalized-record hashes. A sealed role
must not appear in the discovery call space or any pre-freeze embedding cache. After freeze,
the access grant must match both the role manifest and sealed inventory hashes.

Any data-contract, role-leakage or checksum failure blocks the run. Sample sizes, source
definitions and role assignments may not be relaxed in-place; changing them requires a new
run ID and fresh output leaf.
