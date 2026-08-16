"""Auditable table writers and layered V6.3 final reporting."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([dict(row) for row in rows])
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_inventory(root: Path) -> dict[str, Any]:
    base = Path(root)
    rows = []
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base).as_posix() if path.is_file() else ""
        if (
            path.is_file()
            and ".lock" not in path.name
            and relative not in {"result_inventory.json", ".orchestrator.pid"}
        ):
            rows.append({
                "relative_path": relative,
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n".encode("utf-8")
        )
    return {
        "schema_version": "mode3-v6-3-result-inventory-v1",
        "files": rows, "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "root_sha256": digest.hexdigest(),
    }


def final_status(
    confirmation: Mapping[str, Any], *, profile: str, followups: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    levels = confirmation["levels"]
    if profile != "formal":
        status = "SEARCH_COMPLETE"
        claim = "engineering_only_no_scientific_claim"
    elif levels["D_ST_FCA_BASIN"]:
        status = "CERTIFIED_ST_FCA_BASIN"
        claim = "independently_certified_p3_single_token_frozen_cap"
    elif levels["C_ST_FCA_MOAT"]:
        status = "CERTIFIED_ST_FCA_MOAT"
        claim = "independently_certified_p3_single_token_frozen_cap"
    elif levels["B_ST_FCA_CORE"]:
        status = "CERTIFIED_ST_FCA_CORE"
        claim = "independently_certified_p3_single_token_frozen_cap"
    elif levels["A_ST_RADIAL_SHIFT"]:
        status = "CERTIFIED_ST_RADIAL_SHIFT"
        claim = "radial_shift_only_not_frozen_cap_certified"
    else:
        status = "VALID_PRIMARY_NOT_CERTIFIED"
        claim = "valid_independent_confirm_did_not_certify_primary"
    return {
        "schema_version": "mode3-v6-3-final-status-v1",
        "status": status, "profile": profile,
        "primary_question": "是否获得了独立认证的 P3 单 token 冻结球冠？",
        "answer": bool(levels["B_ST_FCA_CORE"] and profile == "formal"),
        "claim_boundary": claim,
        "levels": dict(levels),
        "core_gates": dict(confirmation["core_gates"]),
        "followups": dict(followups or {}),
    }
