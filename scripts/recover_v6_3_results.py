#!/usr/bin/env python3
"""Restore V6.3 release shards into an empty directory and verify triple identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe_extract(archive: Path, destination: Path, prefix: str, seen: set[str]) -> None:
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if (
                not member.isfile() or path.is_absolute() or ".." in path.parts
                or len(path.parts) < 2 or path.parts[0] != prefix
            ):
                raise RuntimeError(f"unsafe V6.3 archive member: {member.name}")
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative in seen:
                raise RuntimeError(f"duplicate V6.3 archive member: {relative}")
            seen.add(relative)
            output = destination.joinpath(*path.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            if stream is None:
                raise RuntimeError(f"unreadable V6.3 archive member: {member.name}")
            temporary = output.with_suffix(output.suffix + ".tmp")
            with temporary.open("wb") as handle:
                shutil.copyfileobj(stream, handle, 8 * 1024 * 1024)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-index", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()
    index = json.loads(args.asset_index.read_text(encoding="utf-8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    destination = args.destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"fresh-clone restore destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    archives = []
    for record in index["assets"]:
        path = args.archive_dir.resolve() / str(record["name"])
        if not path.is_file():
            raise RuntimeError(f"release shard is missing: {path}")
        observed_hash = sha256_file(path)
        observed_bytes = path.stat().st_size
        if observed_hash != record["sha256"] or observed_bytes != int(record["bytes"]):
            raise RuntimeError(f"release shard identity mismatch: {path}")
        safe_extract(path, destination, str(index["archive_prefix"]), seen)
        archives.append({
            "name": path.name, "bytes": observed_bytes, "sha256": observed_hash
        })
    restored = destination / str(index["archive_prefix"])
    expected = {str(row["relative_path"]): row for row in inventory["files"]}
    actual = {
        path.relative_to(restored).as_posix(): path
        for path in restored.rglob("*") if path.is_file()
    }
    mismatches = []
    digest = hashlib.sha256()
    total = 0
    for relative, row in sorted(expected.items()):
        path = actual.get(relative)
        if path is None:
            mismatches.append({"path": relative, "reason": "missing"})
            continue
        size = path.stat().st_size
        observed = sha256_file(path)
        total += size
        digest.update(f"{relative}\0{size}\0{observed}\n".encode("utf-8"))
        if size != int(row["bytes"]) or observed != str(row["sha256"]):
            mismatches.append({"path": relative, "reason": "identity"})
    extras = sorted(set(actual) - set(expected))
    audit = {
        "schema_version": "mode3-v6-3-fresh-clone-audit-v1",
        "archives": archives,
        "file_count": len(actual), "expected_file_count": int(inventory["file_count"]),
        "total_bytes": total, "expected_total_bytes": int(inventory["total_bytes"]),
        "root_sha256": digest.hexdigest(),
        "expected_root_sha256": str(inventory["root_sha256"]),
        "archive_members": len(seen), "mismatches": mismatches, "extra_files": extras,
    }
    audit["triple_identity_ready"] = bool(
        not mismatches and not extras
        and audit["file_count"] == audit["expected_file_count"]
        and audit["total_bytes"] == audit["expected_total_bytes"]
        and audit["root_sha256"] == audit["expected_root_sha256"]
        and audit["archive_members"] == audit["expected_file_count"]
        and index["content_root_sha256"] == audit["expected_root_sha256"]
    )
    atomic_json(args.audit_output.resolve(), audit)
    print(json.dumps(audit, sort_keys=True))
    return 0 if audit["triple_identity_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
