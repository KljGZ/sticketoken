# V7 deployment and recovery

The formal deployment uses a separate tree and output:

```text
tree:   /mnt/data/jkl/StickyToken-v7-occupancy-frontier
output: /mnt/data/jkl/StickyToken-v7-results/sticky_lab/sentence_t5_base/mode3_v7_occupancy_frontier_r3_priority
unit:   sticky-v7-occupancy-frontier.service
GPUs:   physical 4,5,6,7 only
```

The preflight requires a clean worktree, validates immutable config/device policy, compiles the
V7 package, runs lint/type checks and all V7 tests, then performs registration. The service
audits/reuses r5 on CPU while r5 continues working. Immediately before model work, r3 verifies
the r5 unit, live process, clean worktree, commit, config, run manifest, shard failures, GPU
bindings, and cooperative-yield paths. It then stops only the r5 orchestrator with `SIGSTOP`,
writes the current workers' registered yield-request files, and waits for all workers to leave
at durable cache boundaries. It never sends termination signals to r5 workers. An identity
drift, FAILED artifact, invalid path, or 300-second timeout resumes r5 and fails V7 closed.

While V7 owns priority, the stopped r5 orchestrator is rechecked before every V7 GPU launch.
V7 starts model work only after r5 has no live worker on GPUs 4-7 and each selected physical
GPU has the registered free-memory margin. After an explicit V7 terminal artifact is written,
the exact stored r5 orchestrator PID and command identity are revalidated before `SIGCONT`.

Every expensive boundary is resumable: clean precompute, each of 32 FULL shards, merge,
per-token post-selection diagnostics, freeze, and confirmation have durable complete/failure
markers. A completed marker is replayed without duplicate model calls. A `FAILED.json`, hash
drift, dirty formal tree, forbidden GPU, budget limit, or incomplete shard requires diagnosis;
scientific artifacts must not be deleted merely to force progress.

The aggregate route's raw float32 vectors peak at about 45.2 GB. V7 r3 records the original
67.5 GB peak reference but, by explicit operator authorization, uses a 10 GB registration and
model-work start gate. The gate is also checked before every new GPU shard. This override is
an execution gate rather than a capacity guarantee; the monitor must continue checking free
space throughout the run. Once top-20 report-only
diagnostics are durable, a
resumable compaction plan removes only non-selected directories under V7's own
`embedding_cache`; all FULL frontiers and all V6/r5/r7 paths remain untouched.

Monitoring reads `orchestration_logs/status.json`, the user service state/journal, shard
markers, budget, disk, GPU processes, r5 priority ownership evidence, identity hashes, and
`V7_FINAL_STATUS.json`. At a terminal state, the monitor first verifies
`R5_PRIORITY_RESUMED.json`, then stops the V7 service and can be paused. During
`acquire_v7_gpu_priority`, r5 is expected to move from live workers to a stopped scheduler with
zero live GPU workers; any hard kill, PID mismatch, new r5 worker, or missing resume record is
an operational failure.
