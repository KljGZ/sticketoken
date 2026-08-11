"""Atomic result writes and hash-checked completion markers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA = "mode3-v5-artifact-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    writer(temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    def writer(temporary: Path) -> None:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")

    _replace(path, writer)


def write_text(path: Path, value: str) -> None:
    _replace(path, lambda temporary: temporary.write_text(value, encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    values = list(rows)
    names = fieldnames or (list(values[0]) if values else [])

    def writer(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
            if names:
                csv_writer.writeheader()
                csv_writer.writerows(values)
            handle.flush()
            os.fsync(handle.fileno())

    _replace(path, writer)


def write_npz(path: Path, **arrays: Any) -> None:
    def writer(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())

    _replace(path, writer)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_completion(directory: Path, artifacts: Iterable[Path], metadata: Mapping[str, Any]) -> Path:
    directory = directory.resolve()
    rows = []
    for artifact in sorted({path.resolve() for path in artifacts}, key=lambda value: value.as_posix()):
        relative = artifact.relative_to(directory).as_posix()
        rows.append({"path": relative, "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)})
    marker = directory / "COMPLETE.json"
    write_json(marker, {"schema_version": SCHEMA, "artifacts": rows, "metadata": dict(metadata)})
    return marker


def validate_completion(directory: Path, expected_metadata: Mapping[str, Any] | None = None) -> bool:
    marker = directory / "COMPLETE.json"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if payload.get("schema_version") != SCHEMA:
        return False
    if expected_metadata:
        observed = payload.get("metadata", {})
        if any(observed.get(key) != value for key, value in expected_metadata.items()):
            return False
    for row in payload.get("artifacts", []):
        path = directory / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            return False
    return True
