# Mode 3 V5 registered protocol

## Scope and scientific question

V5 studies Mode 3 only. Modes 1 and 2, and every V1--V4 implementation and result, are frozen and must not be executed or modified. V5 asks whether an actual-tokenizer-length-minimal discrete trigger, inserted exactly once, can map heterogeneous texts into one or a few stable, compact representation basins with low benign occupancy.

V5 is an embedding-output query-only black box. Search code can submit text and receive final normalized embeddings. It cannot access parameters, gradients, input embeddings, hidden states, attention, HotFlip, soft prompts, continuous prompts, a target index, Top-K results, or ranking margins. The tokenizer is a separate candidate-construction and exact-length audit surface.

## V4 defects addressed before V5

The complete V4 remote result tree was recovered and published before V5 coding. Its 2,856 files, 157,863,998 bytes, 32 completed single-token shards, 232 CEM archives, 120 validations, 385 query ledgers, and 2,854 original manifest rows were verified remotely, locally, from a GitHub Release download, and from a fresh clone. V4 was not rerun.

V5 corrects the following V4 design or implementation risks:

- random insertion boundaries no longer depend on the trigger;
- support-interior margin, displacement, global contraction, and random-baseline rank leave the search objective and hard certificate;
- a single pooled q95 ball is replaced by one to four stable spherical clusters with an explicit global outlier budget;
- multi-scale occupancy is optimized directly instead of applied only after search;
- mini-batches rotate over the search corpus and every formal archive member is re-evaluated on the complete search split;
- search and validation use directionally consistent occupancy upper bounds;
- test and OOD text is not encoded until validation opens the sealed gate;
- a formal run is tied to one code commit, configuration hash, split hash, tokenizer hash, model revision, and dependency hash;
- partial outputs never count as complete without an atomic completion marker and artifact hash validation;
- all generations, distributions, archives, snapshots, query ledgers, and resource records are retained.

## Robust attractor set

For a trigger `x`, V5 fits between one and four stable spherical clusters. Cluster `j` has center `c_j`, robust radius `rho_j`, and inlier mass `pi_j`. The attractor set is the union of the cluster balls. A global outlier budget, not an independent per-cluster allowance, limits all trimmed and noise points to at most 10 percent. Every retained cluster must contain at least 10 percent of the complete sample as inliers.

For every cluster, V5 reports q80, q90, q95, and CVaR90 radius. Search uses q90; validation freezes a conservative q95/bootstrap upper radius. Cluster counts are chosen by an Occam split rule: an added cluster must have sufficient mass and must materially improve compactness or occupancy without unacceptable occupancy degradation. Validation measures center persistence, matched center drift, assignment ARI/Jaccard, mass intervals, and outlier/coverage intervals with 500 group bootstraps.

## Core objectives and structural constraints

The only search objectives are:

1. minimize maximum and inlier-mass-weighted robust cluster radius;
2. minimize the benign occupancy UCB curve over radius multipliers 1.0 through 2.5, including its weighted AUC.

Exact realizability is a hard filter. Minimum total and per-position coverage, maximum outlier rate, maximum cluster count, and minimum cluster mass are structural constraints. During CEM these constraints progress from a loose early search envelope to their registered final values. The historical Pareto archive is the non-dominated union of all previous full-search archive members and can never regress.

Displacement, global pairwise contraction, empirical support type, separability, and matched random triggers are diagnostics only. The compactness certificate uses the independently preregistered absolute cosine-distance threshold, never a random-trigger quantile. Top-K, PoisonRank, and retrieval margins are downstream outcomes and never feed search, length selection, or cluster selection.

## P1, P2, and P3

- P1 fits a position-specific attractor for prefix, suffix, or deterministic random insertion.
- P2 uses one string at every position but allows a separate frozen attractor set per position.
- P3 requires all positions to enter one shared frozen attractor set and applies coverage/occupancy gates to the worst position, not only pooled values.

P3 implies P2 and P2 implies P1, but failure of P3 cannot invalidate a P1 or P2 certificate. Every length 1--30 is searched and validated. All legal single tokens are exhaustively screened; each multi-token length receives an independent Pareto-CEM search without seeds from Modes 1/2 or frozen V4 candidates.

## Evidence levels and minimal lengths

- Level 0 (`V5-R`): exact tokenizer and contextual realizability.
- Level 1 (`V5-A`): a stable, high-coverage compact attractor set.
- Level 2 (`V5-LO`): Level 1 plus registered multi-scale low benign occupancy.
- Level 3: position status P1, P2, or P3.
- Level 4 (`V5-SA`): one realizable poison text can cover the frozen attractor set.
- Level 5: controlled Top-K retrieval effects after all encoder-level choices are frozen.

The report separately identifies the shortest Level-1, Level-2, position-robust, and single-anchor token lengths. A stronger downstream failure cannot erase a weaker encoder phenomenon.

## Random-position causal control

Before candidate evaluation, V5 creates a boundary manifest from only `text_id`, the registered seed, and replicate. Candidate text is absent from the boundary hash. Every candidate and matched random control uses the same boundary for the same text and replicate.

## Data sealing and freezing

V5 creates fresh, disjoint calibration, search, validation, IID test, OOD, and benign-probe roles under seed 55051. No V4 held-out embedding is reused. Test and OOD CSV identities and insertion manifests may be prepared, but their embeddings cannot be queried before the validation gate opens.

Validation freezes the trigger literal and IDs, actual token length, protocol, cluster count, centers, conservative radii, outlier budget, assignment rule, and position model. Test and OOD only assign new embeddings to this structure; they cannot select a cluster count, refit a center, expand a radius, or change an outlier allowance.

## Pareto-CEM and full-search archive

Each generation uses a deterministic rotating batch manifest. Candidate feasibility follows the progressively tightened structural envelope, while non-dominated sorting and crowding distance operate only on the registered compactness and occupancy objectives. Every generation persists the complete population, Pareto front, elites, token counts, categorical distribution, RNG state, query delta, batch manifest, resource record, checkpoint, and selected high-dimensional cluster snapshots.

At registered intervals and at each restart's end, selected historical candidates are re-evaluated on the entire search-trigger and search-benign-probe roles. Only these full-search records may enter a task/length formal archive, and only formal archive candidates may enter validation.

## Fixed projection visualization

PCA and UMAP bases are fitted once from calibration benign and clean embeddings and then frozen. Every search iteration uses the same basis and axis limits. Frames show benign points, clean source points, triggered cluster inliers, explicit outliers, cluster centers, and clean-to-triggered arrows. V5 publishes per-frame PNGs, raw 2-D coordinates, GIF, MP4, and the high-dimensional source arrays needed to regenerate them.

## Publication and acceptance

Large raw V5 artifacts are released through GitHub Release assets or Git LFS, never as oversized ordinary Git objects. The repository retains readable summaries, complete file indexes, SHA-256 manifests, recovery tooling, configuration, audit records, and the final report. Acceptance requires identical file count, uncompressed bytes, per-file SHA-256, and content-root SHA-256 on the remote compute host, the local recovery directory, and a fresh-clone GitHub reconstruction.
