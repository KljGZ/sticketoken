# Mode 3 V6 protocol: ST-FCA

V6 only studies Mode 3. Modes 1 and 2 and all V1--V5 formal code, configuration, tests, and results are protected read-only history. V6 is implemented only under `sticky_lab/mode3_v6`, `configs/v6_mode3.yaml`, V6-specific scripts, tests, documentation, and result paths.

## Primary claim

The only primary question is whether an actual tokenizer-length-one token, inserted once, sends heterogeneous texts into one validation-frozen high-dimensional spherical cap with:

- one-sided 95% Clopper--Pearson triggered-coverage lower bound strictly above 0.90;
- one-sided 95% independent-benign exact-core occupancy upper bound strictly below 0.01;
- outside-to-inside migration lower bound at least 0.85;
- among triggered-inside samples, an original-outside lower bound at least 0.95.

This claim is named **Single-Token Frozen-Cap Attraction (ST-FCA)**. A rescue model with 2--4 caps is reported separately as **ST-mFCA** and can never be described as a one-center universal phenomenon.

## Formal geometry

Every encoder output is normalized in its original dimension. The formal distance is

`d_angle(z,c) = acos(clip(z dot c,-1,1))`.

Radians are stored internally; degrees and chord length are reporting aids. Reduced two-dimensional coordinates are visualization only and cannot define centers, radii, membership, coverage, occupancy, migration, or certification.

The center is fit only on triggered Cap-fit vectors using a 90%-trimmed spherical mean with mean, medoid, and random restarts. Cap-calibration is independent and freezes the radius using the finite-sample split-conformal order statistic `ceil((n+1)*alpha)`. A radius above 35 degrees fails the preregistered anti-triviality check.

P1 is position-specific. P2 uses the same token with position-conditional centers. P3 uses the same token, center, and radius across prefix, suffix, and random. P3 assigns exactly one third weight to each logical position; random replicates are averaged per text before random receives its one-third weight.

## Candidate discovery

All legal actual-length-one nonspecial tokens are deterministically enumerated with standalone and prefix/suffix/random contextual round trips. Visible and unrestricted sets are both published. Every shard uses the same registered text manifest; sharding partitions tokens only.

The full-search union contains exhaustive-screen candidates, isolated white-box candidates, isolated pure-black-box candidates, and historical V5 single-token candidates. At least 2,000 and by default 5,000 candidates must be re-evaluated on identical full-search roles. Only that re-evaluation can enter validation. An exhaustive negative claim requires either full evaluation of every legal token or a registered safe-elimination proof.

The white-box track may inspect the input embedding and gradients for HotFlip, continuous-token upper bounds, and mechanism analysis. The black-box track can only submit text and receive final normalized vectors. White-box outputs never seed the formal black-box population. Both tracks are candidate sources only, and their candidates receive the same full-search re-evaluation.

## Data and sealing

V6 uses fresh document-disjoint search, Cap-fit, Cap-calibration, IID Test, three IID replications, and at least four source-isolated OOD domains. Exact normalization, SimHash-bucketed candidate generation, and exact shingle-Jaccard verification audit leakage. Repeated sampling is forbidden. Missing document/source/domain/license metadata or insufficient unique records causes formal execution to fail before model loading.

Test, IID replications, and OOD remain unencoded until validation freezes token, cap count, centers, radii, assignment rule, and outlier budget. No sealed phase may refit any of them.

## Required evidence

For triggered, paired-clean, and independent-benign groups, V6 publishes raw normalized radii, cumulative and shell occupancy at 0.1 rho through 2 rho, quantiles, Wasserstein distance, Cliff's delta, KS statistics, group-bootstrap intervals, and raw normal-core counts. Per-sample migration is reported as outside-to-inside, inside-to-inside, outside-to-outside, and inside-to-outside.

The frozen token is matched to at least 50 controls on frequency, IDF, POS, semantic category, character length, casing, input-embedding norm, and naturalness. Additive semantic, wrapper counterfactual, and white-box mechanism analyses distinguish abnormal attraction from an ordinary semantic direction.

A single real poisoned document and retrieval evaluation may run only after cap, migration, and anomaly gates pass. Retrieval never feeds candidate discovery or selection.

## Artifact and publication contract

The run preserves complete exhaustive metrics, radial arrays, white-box trajectories, black-box populations, shared batch manifests, query ledgers, model/config/tokenizer/data fingerprints, validation freezes, sealed evidence, figures with raw projected coordinates, and resource logs. A deterministic content inventory and SHA-256 root are published with split GitHub Release assets. A fresh clone must download, reassemble, restore, and match file count, byte count, every file hash, and the content root.
