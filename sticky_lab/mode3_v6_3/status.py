"""Read-only V6.3 run status summarizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def summarize(output: Path) -> dict[str, Any]:
    root = Path(output)
    stages = {}
    for stage in ("s0", "s1", "s2", "full", "top100"):
        stage_root = root / "stages" / stage
        complete = _read(stage_root / "COMPLETE.json")
        shards = list(stage_root.glob("shard_*/COMPLETE.json")) if stage_root.exists() else []
        failed = list(stage_root.glob("shard_*/FAILED.json")) if stage_root.exists() else []
        stages[stage] = {
            "complete": complete is not None,
            "completed_shards": len(shards), "failed_shards": len(failed),
            "summary": complete,
        }
    budget = _read(root / "budget" / "observed.json")
    final = _read(root / "FINAL_STATUS.json")
    return {
        "schema_version": "mode3-v6-3-status-v1",
        "output": str(root.resolve()),
        "run_manifest": _read(root / "run_manifest.json"),
        "stages": stages,
        "freeze": _read(root / "freeze" / "COMPLETE.json"),
        "confirmation": _read(root / "confirm" / "COMPLETE.json"),
        "budget": budget,
        "final": final,
        "zero_byte_critical": [
            path.relative_to(root).as_posix() for path in root.rglob("*")
            if path.is_file() and path.stat().st_size == 0 and path.name not in {".ledger.lock"}
        ] if root.exists() else [],
    }
