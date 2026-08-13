# Mode 3 V5 formal results

## Outcome

Mode 3 V5 found and independently confirmed discrete, single-insertion representation attractors for P1, P2, and P3. The strongest universal result is a P3 shared low-benign-occupancy attractor of actual tokenizer length 14. V5 also found a shared single-token Level-1 attractor (`Diabetes`), but that token does not satisfy the registered low-occupancy certificate.

The validation gate opened with ten frozen selections: the shortest Level-1 and Level-2 candidate for each of the five registered tasks. All ten passed the frozen-structure IID Test Level-1 certificate and all ten generalized at Level 1 to OOD. All five Level-2 candidates passed IID Test. Four retained Level 2 on OOD; the suffix candidate remained a valid Level-1 attractor but its OOD occupancy AUC, `0.0303446`, narrowly exceeded the preregistered `0.030` Level-2 bound.

Neither Test nor OOD refitted cluster count, centers, radii, outlier budgets, or assignment rules.

## Registered lineage

| Item | Registered value |
|---|---|
| Run ID | `sentence-t5-base-mode3-v5-seed-55051` |
| Formal search/validation commit | `2c9d478ed6642b809ff3dfe91c252632724a745e` |
| Resource-exhaustion hardening commit | `b892e12cbdbea62d1f48428c0053f3d2235570c9` |
| Sealed-phase recovery commit | `8648f676265d040fde40e3598ee1bbe846ad4400` |
| Final audit implementation | `9a2be26164724c1be3320846d84b0d43e76dbedf` |
| Config SHA-256 | `7865bd10a85525094b6a8813cfc62029bc219fe819d0e5d59f29582e7f14ab9d` |
| Data split SHA-256 | `c6fd319de19bfb14c19050c8952514b51292d41cf70df3624d9aabd02cc2eea9` |
| Tokenizer SHA-256 | `58b50bf2bbad306b7f17dbfee0d41b164d6186ab8e27d589a15bf172a95f4cb4` |
| Dependency lock SHA-256 | `39d37a19d70159056d056f3fb3ed830318e0082e86b39d0ee1b0c3838d5294d2` |
| Model revision | `fc5d4628481afbbaaacd7af6bb07cf9d3865f781` |

The original one-time Test invocation atomically wrote only the two base embedding roles and then failed before evaluating any frozen candidate because `Candidate.task` was referenced even though task belongs to the frozen validation record. The failed log is retained with SHA-256 `10b2ddc524326598eaa8db883f834eb577812b4cc9601a95e920fb792b4b2f91`. Recovery commit `8648f67` passed all 17 V5-specific tests, validated and reused those atomic base embeddings, reconstructed their query-ledger contribution, and allowed only `test`, `ood`, `retrieval`, and `finalize`. Search, full-search merge, validation, freeze, configuration, data, seeds, thresholds, and budgets remained immutable. The complete recovery contract is in `results_publication/v5/recovery_lineage.json`.

## Search and audit closure

| Artifact family | Completed | Expected |
|---|---:|---:|
| Calibration shards | 8 | 8 |
| Single-token screening shards | 40 | 40 |
| Pareto-CEM search restarts | 435 | 435 |
| Generations | 6,960 | 6,960 |
| Full-search merges | 145 | 145 |
| Formal task/length archives | 150 | 150 |
| Validations | 150 | 150 |
| Optimization GIFs | 435 | 435 |
| Optimization MP4s | 435 | 435 |

All 6,960 generations contain `population.csv`, `batch_manifest.json`, `rng_state.json`, and `query_ledger.json`. Required artifact zero-byte count is zero. The formal result audit found no resource-failure marker in formal artifacts and passed with no errors. There are 7,469 retained high-dimensional/cluster snapshot groups.

## Length frontier

The shortest validation-certified lengths are:

| Protocol level | Task | Level 1 length | Level 2 length |
|---|---|---:|---:|
| P1 | Prefix | 1 | 5 |
| P1 | Suffix | 1 | 6 |
| P1 | Fixed random boundary | 1 | 13 |
| P2 | Position-conditional centers | 1 | 5 |
| P3 | One shared center set | 1 | 14 |

Level 1 establishes a compact, stable, high-coverage attractor. Level 2 additionally enforces the preregistered multi-scale low-benign-occupancy thresholds. Thus the single-token result is evidence of representation attraction, not evidence that one token alone creates a low-benign-occupancy retrieval region.

## Frozen candidates and independent confirmation

| Task | Level | Length | Trigger | IID | OOD |
|---|---|---:|---|---|---|
| Prefix | A | 1 | `diabetes` | A pass | A pass |
| Prefix | LO | 5 | `Songs glitch Results Wish tapping` | LO pass | LO pass |
| Suffix | A | 1 | `racist` | A pass | A pass |
| Suffix | LO | 6 | `forgot brought Edward presidential einer Minecraft` | LO pass | A pass; LO fail |
| Random | A | 1 | `Diabetes` | A pass | A pass |
| Random | LO | 13 | `peripheral crying după bakery Cannabis pentru races Chelsea doesn Nobody inclus nervous fabricat` | LO pass | LO pass |
| P2 conditional | A | 1 | `chemotherapy` | A pass | A pass |
| P2 conditional | LO | 5 | `Minecraft prisoners seamlessly reject Dragnea` | LO pass | LO pass |
| P3 shared | A | 1 | `Diabetes` | A pass | A pass |
| P3 shared | LO | 14 | `Menschen Slovakia literally copiilor conscious inferior sta sta connaissance secure Christians atunci incercat Simply` | LO pass | LO pass |

For the P3 Level-2 trigger, IID worst-position coverage LCB is `0.9360`, outlier-rate UCB is `0.0640`, occupancy AUC is `0.00722`, and `lambda_star` is `2.25`. OOD worst-position coverage LCB is `0.9400`, outlier-rate UCB is `0.0600`, occupancy AUC is `0.01110`, and `lambda_star` remains `2.25`.

For the P2 Level-2 trigger, IID worst-position coverage LCB is `0.9338` with occupancy AUC `0.02577`; OOD worst-position coverage LCB is `0.9573` with occupancy AUC `0.02732`.

## Controlled single-poison retrieval

Only after encoder-level certification, V5 selected one real poison text for the frozen prefix Level-2 length-5 trigger. No retrieval result fed trigger search or length selection.

| Metric | Triggered query | Clean query |
|---|---:|---:|
| Poison Hit@1 | 0.995 | 0.000 |
| Poison Hit@5 | 1.000 | 0.001 |
| Poison Hit@10 | 1.000 | 0.001 |
| Median poison rank | 1 | 509 |
| Mean poison rank | 1.005 | 561.012 |
| Mean poison similarity | 0.91899 | 0.69289 |
| q05 margin at 1 | 0.04348 | -0.28450 |
| q05 margin at 5 | 0.07961 | -0.20219 |
| q05 margin at 10 | 0.09165 | -0.17798 |

This downstream result is consistent with the proposed mechanism: the trigger first migrates heterogeneous queries into a frozen low-occupancy representation region, after which a single aligned entry obtains a strong rank advantage.

## Query budget

Across 787 ledgers, V5 recorded 1,355,536 embedding API calls, 92,721,912 requested texts, 9,115,936 cache hits, and 83,605,976 texts actually submitted to the embedding model. These counts include the reconstructed ledger contribution of the two atomic Test base-role calls made immediately before the implementation fault.

## Complete raw-result identity

The deterministic remote inventory contains:

| Field | Value |
|---|---:|
| Files | 117,433 |
| Uncompressed bytes | 19,459,976,149 |
| Content-root SHA-256 | `906c5e1523fbe0897501f31c7258cc3665acf9a3bf19ae0046fb882cabca314f` |

`results_publication/v5/remote_inventory.json` is the complete readable JSON index and `results_publication/v5/complete_file_manifest.csv` is the per-file reconstruction contract. The raw tree is published as 14 deterministic size-bounded shards in the [GitHub Release `mode3-v5-full-results`](https://github.com/KljGZ/sticketoken/releases/tag/mode3-v5-full-results). The release contains 18 uploaded assets: the 14 raw-result shards plus the release index, manifest, remote inventory, and formal audit.

A fresh clone at commit `cf2d1c8d418d540c6fbd3915779217778e280390` downloaded the registered shards and restored the complete tree independently. The restored tree contains exactly 117,433 files and 19,459,976,149 bytes; every path, byte count, and per-file SHA-256 matches the registered manifest. A second independent inventory reproduced content-root SHA-256 `906c5e1523fbe0897501f31c7258cc3665acf9a3bf19ae0046fb882cabca314f`. Therefore the formal remote tree, GitHub reconstruction, and local fresh-clone reconstruction are byte-identical. One truncated download of shard 12 was retained as invalid-attempt evidence and safely retried without deleting or changing any valid result.

## Files

- Registered protocol: `docs/sticky_attractor_v5_protocol.md`
- Complete length frontier: `results_publication/v5/length_frontier.csv`
- Frontier figure: `results_publication/v5/length_frontier.png`
- Frozen selections: `results_publication/v5/frozen_selection.json`
- IID and OOD summaries: `results_publication/v5/test_summary.json`, `results_publication/v5/ood_summary.json`
- Controlled retrieval: `results_publication/v5/single_poison_retrieval.json`
- Query budget: `results_publication/v5/query_budget.json`
- Formal audit: `results_publication/v5/v5_formal_audit.json`
- Full inventory and manifest: `results_publication/v5/remote_inventory.json`, `results_publication/v5/complete_file_manifest.csv`
- GitHub asset registry: `results_publication/v5/release_asset_index.json`, `results_publication/v5/release_publish.json`
- Fresh-clone and publication audits: `results_publication/v5/fresh_clone_release_audit.json`, `results_publication/v5/local_inventory.json`, `results_publication/v5/publication_audit.json`
- Publication completion marker: `results_publication/v5/V5_RESULT_PUBLICATION_COMPLETE.json`
