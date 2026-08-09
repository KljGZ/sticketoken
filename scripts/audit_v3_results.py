#!/usr/bin/env python3
"""Audit completeness and preregistered invariants of a finished V3 run."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


POSITIONS = ("prefix", "suffix", "random")
PROTOCOLS = ("separator", "blank")
RESTARTS = range(4)
SEARCH_LENGTHS = tuple(range(2, 31, 2))
REGISTERED_LENGTHS = (1, *SEARCH_LENGTHS)
CSV_FIELD_LIMIT = 16 * 1024 * 1024


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frontier_lengths(path: Path) -> list[int]:
    # Universal frontiers retain serialized per-position CI evidence.  Those
    # audit fields legitimately exceed the csv module's 128 KiB default.
    csv.field_size_limit(CSV_FIELD_LIMIT)
    with path.open(newline="", encoding="utf-8") as handle:
        return [int(row["component_length"]) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    data = _load(root / "data_audit.json")
    expected_sizes = {"search": 3000, "validation": 1000, "test": 1000}
    if data.get("split_sizes") != expected_sizes or int(data.get("ood_size", -1)) != 1000:
        errors.append("registered split sizes are not search/validation/test/OOD = 3000/1000/1000/1000")
    if any(int(value) != 0 for value in data.get("overlap", {}).values()):
        errors.append("sentence/group leakage audit is non-zero")
    if not data.get("document_provenance_available", False):
        warnings.append("source files have no verifiable document IDs; grouping falls back to one group per unique sentence")

    search_files: list[Path] = []
    missing_search: list[str] = []
    for position in (*POSITIONS, "universal"):
        for protocol in PROTOCOLS:
            for restart in RESTARTS:
                for length in SEARCH_LENGTHS:
                    path = (
                        root
                        / "search"
                        / position
                        / protocol
                        / f"restart_{restart:02d}"
                        / f"length_{length:02d}_candidates.csv"
                    )
                    if path.exists():
                        search_files.append(path)
                    else:
                        missing_search.append(str(path.relative_to(root)))
    if missing_search:
        errors.append(f"missing {len(missing_search)} of 480 registered search candidate files")

    source_commits: Counter[str] = Counter()
    completion_commits: Counter[str] = Counter()
    search_summaries = sorted(root.glob("search_summary_*.json"))
    for path in search_summaries:
        record = _load(path)
        source_commits[str(record.get("git_commit"))] += 1
        completion_commits[str(record.get("git_commit_at_completion"))] += 1
    if len(search_summaries) != 32:
        errors.append(f"expected 32 search summaries, found {len(search_summaries)}")
    if source_commits.get("None", 0):
        errors.append(f"{source_commits['None']} search summaries lack a source commit")
    if completion_commits.get("None", 0):
        warnings.append(
            f"{completion_commits['None']} search summaries predate completion-commit instrumentation; "
            "their source commits and candidate-file manifest hashes remain registered"
        )

    task_records: dict[str, Any] = {}
    for position in POSITIONS:
        for protocol in PROTOCOLS:
            label = f"{protocol}/{position}"
            letter = "A" if protocol == "separator" else "B"
            frozen_path = root / f"mode3{letter}_{position}_frozen.json"
            task_dir = root / "validation" / position / protocol
            required = [
                frozen_path,
                task_dir / "length_frontier.csv",
                task_dir / "test_result.json",
                task_dir / "ood_result.json",
            ]
            missing = [str(path.relative_to(root)) for path in required if not path.exists()]
            if missing:
                errors.append(f"{label}: missing {', '.join(missing)}")
                continue
            frozen = _load(frozen_path)
            test = _load(task_dir / "test_result.json")
            ood = _load(task_dir / "ood_result.json")
            if frozen.get("registered_length_schedule") != list(REGISTERED_LENGTHS):
                errors.append(f"{label}: frozen length schedule differs from registration")
            if _frontier_lengths(task_dir / "length_frontier.csv") != list(REGISTERED_LENGTHS):
                errors.append(f"{label}: validation frontier is incomplete or out of order")
            if frozen.get("selection_split") != "validation" or frozen.get("test_used_for_selection") is not False:
                errors.append(f"{label}: test-selection separation invariant failed")
            if int(frozen.get("actual_token_length", -1)) != int(frozen.get("component_length", -2)):
                errors.append(f"{label}: tokenizer length differs from registered component length")
            if frozen.get("exact_token_roundtrip") is not True or float(frozen.get("realizability_rate", 0.0)) < 0.95:
                errors.append(f"{label}: token round-trip/realizability invariant failed")
            for split, record in (("test", test), ("ood", ood)):
                if record.get("blank_region_certified") and not record.get("separator_certified"):
                    errors.append(f"{label}/{split}: 3B => 3A invariant failed")
            task_records[label] = {
                "component_length": int(frozen["component_length"]),
                "trigger": frozen["trigger"],
                "selection_status": frozen["selection_status"],
                "validation_certified": bool(frozen["validation_certified"]),
                "test_certified": bool(test["test_certified"]),
                "ood_certified": bool(test["ood_certified"]),
                "full_generalized": bool(test["full_generalized"]),
            }

    for protocol in PROTOCOLS:
        label = f"{protocol}/universal"
        letter = "A" if protocol == "separator" else "B"
        frozen_path = root / f"mode3{letter}_universal_frozen.json"
        task_dir = root / "validation" / "universal" / protocol
        required = [frozen_path, task_dir / "length_frontier.csv", task_dir / "test_result.json"]
        missing = [str(path.relative_to(root)) for path in required if not path.exists()]
        if missing:
            errors.append(f"{label}: missing {', '.join(missing)}")
            continue
        frozen = _load(frozen_path)
        test = _load(task_dir / "test_result.json")
        if frozen.get("registered_length_schedule") != list(REGISTERED_LENGTHS):
            errors.append(f"{label}: frozen length schedule differs from registration")
        if _frontier_lengths(task_dir / "length_frontier.csv") != list(REGISTERED_LENGTHS):
            errors.append(f"{label}: validation frontier is incomplete or out of order")
        if frozen.get("selection_split") != "validation" or frozen.get("test_used_for_selection") is not False:
            errors.append(f"{label}: test-selection separation invariant failed")
        if int(frozen.get("actual_token_length", -1)) != int(frozen.get("component_length", -2)):
            errors.append(f"{label}: tokenizer length differs from registered component length")
        if frozen.get("exact_token_roundtrip") is not True or float(frozen.get("realizability_rate", 0.0)) < 0.95:
            errors.append(f"{label}: token round-trip/realizability invariant failed")
        for split_key in ("test_per_position_metrics", "ood_per_position_metrics"):
            for position, record in test.get(split_key, {}).items():
                if record.get("blank_region_certified") and not record.get("separator_certified"):
                    errors.append(f"{label}/{split_key}/{position}: 3B => 3A invariant failed")
        task_records[label] = {
            "component_length": int(frozen["component_length"]),
            "trigger": frozen["trigger"],
            "selection_status": frozen["selection_status"],
            "validation_certified": bool(test["validation_position_universal_certified"]),
            "test_certified": bool(test["test_position_universal_certified"]),
            "ood_certified": bool(test["ood_position_universal_certified"]),
            "full_generalized": bool(test["full_generalized"]),
        }

    registered_inputs = [
        root / "resolved_config.json",
        root / "data_audit.json",
        root / "support_audit.json",
        root / "spherical_kmeans_audit.json",
        root / "benign_support.npz",
        root / "candidate_pool.csv",
        root / "soft_prompt_summary.csv",
        root / "merge-prepare_summary.json",
        root / "unique_search.csv",
        root / "unique_validation.csv",
        root / "unique_test.csv",
        root / "unique_ood.csv",
    ]
    missing_inputs = [str(path.relative_to(root)) for path in registered_inputs if not path.exists()]
    if missing_inputs:
        errors.append(f"missing registered inputs: {', '.join(missing_inputs)}")

    return {
        "protocol_version": 3,
        "complete": not errors,
        "errors": errors,
        "warnings": warnings,
        "registered_lengths": list(REGISTERED_LENGTHS),
        "search_candidate_file_count": len(search_files),
        "search_candidate_manifest_sha256": _manifest_digest(root, search_files),
        "search_summary_count": len(search_summaries),
        "search_source_commits": dict(sorted(source_commits.items())),
        "search_completion_commits": dict(sorted(completion_commits.items())),
        "registered_input_sha256": {
            path.relative_to(root).as_posix(): _sha256(path)
            for path in registered_inputs
            if path.exists()
        },
        "tasks": task_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    output = args.output or args.root / "registered_result_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
