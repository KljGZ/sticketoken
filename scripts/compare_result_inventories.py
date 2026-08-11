#!/usr/bin/env python3
"""Compare two content inventories and optionally enforce the V4 result contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def index(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["relative_path"]): row for row in payload["files"]}  # type: ignore[index]


def count_paths(paths: set[str], predicate) -> int:
    return sum(1 for path in paths if predicate(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", required=True, type=Path)
    parser.add_argument("--local", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--enforce-v4-contract", action="store_true")
    args = parser.parse_args()

    remote_payload = json.loads(args.remote.read_text(encoding="utf-8"))
    local_payload = json.loads(args.local.read_text(encoding="utf-8"))
    remote = index(remote_payload)
    local = index(local_payload)
    remote_paths = set(remote)
    local_paths = set(local)
    missing = sorted(remote_paths - local_paths)
    extra = sorted(local_paths - remote_paths)
    byte_mismatches = sorted(
        path for path in remote_paths & local_paths if int(remote[path]["bytes"]) != int(local[path]["bytes"])
    )
    hash_mismatches = sorted(
        path for path in remote_paths & local_paths if remote[path]["sha256"] != local[path]["sha256"]
    )
    structural_counts = {
        "single_token_completed_shards": count_paths(
            local_paths, lambda path: path.startswith("screen/") and Path(path).name.startswith("shard_")
        ),
        "cem_search_archives": count_paths(local_paths, lambda path: path.endswith("_archive.csv")),
        "task_by_length_validations": count_paths(
            local_paths, lambda path: path.startswith("validation/") and path.endswith("/summary.json")
        ),
        "query_ledgers": count_paths(local_paths, lambda path: path.startswith("query_ledgers/")),
        "reported_manifest_artifacts": 0,
        "length_frontier_rows": 0,
    }
    result_root = Path(str(local_payload["root"]))
    manifest_path = result_root / "sha256_manifest.csv"
    frontier_path = result_root / "length_frontier.csv"
    if manifest_path.is_file():
        structural_counts["reported_manifest_artifacts"] = max(
            0, sum(1 for _ in manifest_path.open("r", encoding="utf-8")) - 1
        )
    if frontier_path.is_file():
        structural_counts["length_frontier_rows"] = max(
            0, sum(1 for _ in frontier_path.open("r", encoding="utf-8")) - 1
        )
    expected = {
        "single_token_completed_shards": 32,
        "cem_search_archives": 232,
        "task_by_length_validations": 120,
        "query_ledgers": 385,
        "reported_manifest_artifacts": 2854,
        "length_frontier_rows": 120,
    }
    structural_mismatches = {
        key: {"expected": value, "observed": structural_counts[key]}
        for key, value in expected.items()
        if structural_counts[key] != value
    }
    complete = not (missing or extra or byte_mismatches or hash_mismatches)
    if args.enforce_v4_contract:
        complete = complete and not structural_mismatches
    result = {
        "schema_version": "mode3-inventory-comparison-v1",
        "remote_file_count": remote_payload["file_count"],
        "local_file_count": local_payload["file_count"],
        "remote_total_bytes": remote_payload["total_bytes"],
        "local_total_bytes": local_payload["total_bytes"],
        "remote_root_sha256": remote_payload["root_sha256"],
        "local_root_sha256": local_payload["root_sha256"],
        "missing_files": missing,
        "extra_files": extra,
        "byte_mismatches": byte_mismatches,
        "hash_mismatches": hash_mismatches,
        "structural_counts": structural_counts,
        "structural_mismatches": structural_mismatches,
        "recovery_complete": complete,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
