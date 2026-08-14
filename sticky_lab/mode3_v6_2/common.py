"""Shared V6.2 configuration and artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from sticky_lab.mode3_v6.atomic_io import write_json, write_jsonl
from sticky_lab.mode3_v6.insertion import BoundaryManifest, BoundaryRecord
from sticky_lab.mode3_v6.tokenizer_audit import LegalToken

from .geometry import FrozenCapModel


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if str(config.get("protocol_version")) != "6.2" or int(config.get("protocol_revision", 0)) != 1:
        raise RuntimeError("not a V6.2 configuration")
    if config.get("scope", {}).get("only_mode") != 3:
        raise RuntimeError("V6.2 scope must be Mode 3 only")
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_role(output: Path, role: str) -> list[dict[str, str]]:
    return [dict(row) for row in read_jsonl(output / "registration" / "roles" / f"{role}.jsonl")]


def load_manifest(output: Path, roles: Sequence[str] | None = None) -> BoundaryManifest:
    if roles is not None:
        rows = [
            row
            for role in roles
            for row in read_jsonl(output / "registration" / "random_boundaries" / f"{role}.jsonl")
        ]
    else:
        rows = [
            row
            for path in sorted((output / "registration" / "random_boundaries").glob("*.jsonl"))
            for row in read_jsonl(path)
        ]
    return BoundaryManifest(
        [
            BoundaryRecord(**row)
            for row in rows
        ]
    )


def load_legal(output: Path) -> list[LegalToken]:
    return [
        LegalToken(**{key: row[key] for key in LegalToken.__dataclass_fields__})
        for row in read_jsonl(output / "enumeration" / "legal_unrestricted.jsonl")
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_is_relative_to(path: Path, root: Path) -> bool:
    """Python-3.8-compatible containment check on resolved paths."""
    try:
        Path(path).relative_to(Path(root))
        return True
    except ValueError:
        return False


def verified_checksum_tree(root: Path, checksum_manifest: Path) -> dict[str, Any]:
    """Verify an exact local file tree against a SHA-256 checksum manifest."""
    root = Path(root).resolve()
    manifest = Path(checksum_manifest)
    expected: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise RuntimeError(f"invalid SHA-256 manifest line {line_number}")
        target = Path(fields[1].lstrip("* ")).resolve()
        if not path_is_relative_to(target, root):
            raise RuntimeError(f"model checksum path escapes model root: {target}")
        relative = target.relative_to(root).as_posix()
        if relative in expected:
            raise RuntimeError(f"duplicate model checksum entry: {relative}")
        expected[relative] = fields[0].lower()
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(expected):
        raise RuntimeError(
            f"model checksum inventory mismatch: missing={sorted(set(expected) - actual)}, "
            f"unregistered={sorted(actual - set(expected))}"
        )
    observed = {relative: sha256_file(root / relative) for relative in sorted(expected)}
    mismatched = [relative for relative in observed if observed[relative] != expected[relative]]
    if mismatched:
        raise RuntimeError(f"model checksum mismatch: {mismatched}")
    payload = json.dumps(observed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "root": root.as_posix(),
        "checksum_manifest": manifest.resolve().as_posix(),
        "files": observed,
        "file_count": len(observed),
        "tree_sha256": hashlib.sha256(payload).hexdigest(),
    }


def cap_from_arrays(
    *,
    token_id: int,
    token_text: str,
    center: np.ndarray,
    radius: float,
    fit_role: str = "s0_fit",
    calibration_role: str = "s0_radius",
) -> FrozenCapModel:
    return FrozenCapModel(
        token_id=int(token_id),
        token_text=str(token_text),
        protocol="P3_ST_FCA_Core",
        centers=np.asarray(center, dtype=np.float64)[None, :],
        radii=np.asarray([radius], dtype=np.float64),
        design_coverage=0.92,
        fit_role=fit_role,
        radius_role=calibration_role,
        cap_count=1,
    )


def model_from_dict(value: Mapping[str, Any]) -> FrozenCapModel:
    return FrozenCapModel(
        token_id=int(value["token_id"]), token_text=str(value["token_text"]),
        protocol=str(value["protocol"]),
        centers=np.asarray(value["centers"], dtype=np.float64),
        radii=np.asarray(value["radii"], dtype=np.float64),
        design_coverage=float(value["design_coverage"]),
        fit_role=str(value["fit_role"]), radius_role=str(value["radius_role"]),
        cap_count=int(value["cap_count"]),
        assignment_rule=str(value.get("assignment_rule", "minimum_normalized_angular_distance")),
        outlier_rule=str(value.get("outlier_rule", "global_weighted_trim")),
    )


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, value = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(value)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "load_config",
    "read_jsonl",
    "load_role",
    "load_manifest",
    "load_legal",
    "sha256_file",
    "path_is_relative_to",
    "verified_checksum_tree",
    "cap_from_arrays",
    "model_from_dict",
    "atomic_savez",
    "write_json",
    "write_jsonl",
]
