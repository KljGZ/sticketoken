#!/usr/bin/env python3
"""Create a deterministic, content-addressed inventory for a result tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_metadata(relative_path: str) -> dict[str, Any]:
    parts = Path(relative_path).parts
    metadata: dict[str, Any] = {
        "schema_version": None,
        "run_phase": parts[0] if len(parts) > 1 else "root",
        "task": None,
        "length": None,
        "restart": None,
        "iteration": None,
        "implementation_commit": None,
        "config_hash": None,
    }
    for part in parts:
        if part.startswith("length_"):
            try:
                metadata["length"] = int(part.split("_", 1)[1])
            except ValueError:
                pass
        elif part.startswith("restart_"):
            try:
                metadata["restart"] = int(part.split("_", 1)[1])
            except ValueError:
                pass
        elif part.startswith("iteration_"):
            try:
                metadata["iteration"] = int(part.split("_", 1)[1])
            except ValueError:
                pass
        elif part.startswith("generation_"):
            try:
                metadata["iteration"] = int(part.split("_", 1)[1])
            except ValueError:
                pass
        elif part in {"prefix", "suffix", "random", "conditional", "shared", "universal"}:
            metadata["task"] = part
    return metadata


def enrich_json_metadata(path: Path, metadata: dict[str, Any]) -> None:
    if path.suffix.lower() != ".json" or path.stat().st_size > 8 * 1024 * 1024:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    aliases = {
        "schema_version": ("schema_version", "schemaVersion"),
        "implementation_commit": (
            "implementation_commit",
            "run_code_commit",
            "git_commit",
            "commit",
        ),
        "config_hash": ("config_hash", "config_sha256"),
        "iteration": ("iteration", "generation"),
        "restart": ("restart", "restart_id"),
        "length": ("length", "token_length", "actual_token_length"),
        "task": ("task", "position", "protocol"),
    }
    for destination, keys in aliases.items():
        if metadata.get(destination) is not None:
            continue
        for key in keys:
            if key in payload and not isinstance(payload[key], (dict, list)):
                metadata[destination] = payload[key]
                break


def root_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def inventory(root: Path, excluded: set[Path] | None = None) -> dict[str, Any]:
    root = root.resolve()
    excluded = {path.resolve() for path in (excluded or set())}
    rows: list[dict[str, Any]] = []
    paths = sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix())
    for path in paths:
        if path.resolve() in excluded:
            continue
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        metadata = infer_metadata(relative)
        enrich_json_metadata(path, metadata)
        rows.append(
            {
                "relative_path": relative,
                "file_type": path.suffix.lower().lstrip(".") or "none",
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
                "mtime_utc_ns": stat.st_mtime_ns,
                **metadata,
            }
        )
    return {
        "schema_version": "mode3-result-inventory-v1",
        "root": str(root),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "root_sha256": root_hash(rows),
        "files": rows,
    }


def write_inventory(payload: dict[str, Any], output: Path, csv_output: Path | None) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    if csv_output is None:
        return
    rows = payload["files"]
    csv_output = csv_output.resolve()
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = csv_output.with_suffix(csv_output.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary_csv, csv_output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()
    root = args.results.resolve()
    if not root.is_dir():
        raise SystemExit(f"result directory does not exist: {root}")
    excluded = {args.output}
    if args.csv_output is not None:
        excluded.add(args.csv_output)
    payload = inventory(root, excluded)
    write_inventory(payload, args.output, args.csv_output)
    print(json.dumps({key: payload[key] for key in ("file_count", "total_bytes", "root_sha256")}))


if __name__ == "__main__":
    main()
