"""Immutable run and content fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def git_status(root: Path) -> str:
    return subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)


def inventory(root: Path, *, exclude: Iterable[str] = ()) -> dict[str, object]:
    excluded = set(exclude)
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    root_digest = hashlib.sha256()
    for row in rows:
        root_digest.update(f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    return {
        "algorithm": "sha256-merkle-v1",
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "content_root": root_digest.hexdigest(),
        "files": rows,
    }
