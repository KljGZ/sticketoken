# V7 deployment and recovery

The formal deployment uses a separate tree and output:

```text
tree:   /mnt/data/jkl/StickyToken-v7-occupancy-frontier
output: /mnt/data/jkl/StickyToken-v7-results/sticky_lab/sentence_t5_base/mode3_v7_occupancy_frontier_r2_10g
unit:   sticky-v7-occupancy-frontier.service
GPUs:   physical 4,5,6,7 only
```

The preflight requires a clean worktree, validates immutable config/device policy, compiles the
V7 package, runs lint/type checks and all V7 tests, then performs registration. The service
audits/reuses r5 on CPU and waits without signaling r5. It starts model work only after r5 has
a terminal artifact and each selected physical GPU has the registered free-memory margin.

Every expensive boundary is resumable: clean precompute, each of 32 FULL shards, merge,
per-token post-selection diagnostics, freeze, and confirmation have durable complete/failure
markers. A completed marker is replayed without duplicate model calls. A `FAILED.json`, hash
drift, dirty formal tree, forbidden GPU, budget limit, or incomplete shard requires diagnosis;
scientific artifacts must not be deleted merely to force progress.

The aggregate route's raw float32 vectors peak at about 45.2 GB. V7 r2 records the original
67.5 GB peak reference but, by explicit operator authorization, uses a 10 GB registration and
model-work start gate. This override is an execution gate rather than a capacity guarantee;
the monitor must continue checking free space throughout the run. Once top-20 report-only
diagnostics are durable, a
resumable compaction plan removes only non-selected directories under V7's own
`embedding_cache`; all FULL frontiers and all V6/r5/r7 paths remain untouched.

Monitoring reads `orchestration_logs/status.json`, the user service state/journal, shard
markers, budget, disk, GPU processes, identity hashes, and `V7_FINAL_STATUS.json`. At a terminal
state the service should be inactive and the 15-minute monitor can be paused. While r5 is still
running, `waiting_priority_peer` is healthy and must not be “fixed” by stopping or preempting
r5.
