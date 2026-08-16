"""Physical and logical sealing for all V6.3 confirmation roles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from .config import canonical_sha256
from .errors import ManifestMismatch, RoleLeakage
from .roles import RoleAccessGuard


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_sealed_inventory(
    output: Path,
    role_paths: Mapping[str, Path],
    *,
    role_manifest_sha256: str,
) -> dict[str, Any]:
    files = {
        str(role): {
            "path": str(Path(path).resolve()),
            "sha256": _sha256_file(Path(path)),
            "bytes": Path(path).stat().st_size,
        }
        for role, path in sorted(role_paths.items())
    }
    payload = {
        "schema_version": "mode3-v6-3-sealed-inventory-v1",
        "role_manifest_sha256": str(role_manifest_sha256),
        "files": files,
        "confirm_embeddings_present": False,
        "freeze_present_at_seal_time": False,
    }
    payload["inventory_sha256"] = canonical_sha256(payload)
    target = Path(output) / "sealed" / "SEALED_INVENTORY.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def physically_seal(role_paths: Sequence[Path]) -> None:
    if os.name == "nt":
        # Windows permission semantics are covered by RoleAccessGuard in unit
        # tests; formal execution is Linux and must use mode 000 below.
        return
    for path in role_paths:
        os.chmod(path, 0)
    for directory in sorted({Path(path).parent for path in role_paths}):
        os.chmod(directory, 0)


def assert_still_sealed(output: Path) -> None:
    root = Path(output)
    if (root / "sealed" / "SEALED_ACCESS_GRANT.json").exists():
        raise RoleLeakage("sealed grant exists before freeze")
    forbidden = [
        path for path in (root / "embedding_cache").rglob("*")
        if path.is_file() and any(name in path.as_posix() for name in ("confirm", "iid_", "ood_", "semantic", "retrieval"))
    ] if (root / "embedding_cache").exists() else []
    if forbidden:
        raise RoleLeakage(f"sealed embedding cache exists before freeze: {forbidden[:5]}")


def grant_access(
    output: Path,
    *,
    role_manifest_sha256: str,
    sealed_inventory_path: Path,
) -> dict[str, Any]:
    root = Path(output)
    primary = root / "freeze" / "primary.json"
    freeze_marker = root / "freeze" / "FREEZE.sha256"
    if not primary.is_file() or not freeze_marker.is_file():
        raise RoleLeakage("cannot open sealed roles before an immutable primary freeze")
    freeze_sha256 = _sha256_file(primary)
    if freeze_marker.read_text(encoding="utf-8").strip().split()[0] != freeze_sha256:
        raise ManifestMismatch("FREEZE.sha256 does not match primary.json")
    inventory = json.loads(Path(sealed_inventory_path).read_text(encoding="utf-8"))
    if inventory.get("role_manifest_sha256") != str(role_manifest_sha256):
        raise ManifestMismatch("sealed inventory role-manifest mismatch")
    if inventory.get("inventory_sha256") != canonical_sha256({key: value for key, value in inventory.items() if key != "inventory_sha256"}):
        raise ManifestMismatch("sealed inventory hash mismatch")
    if os.name != "nt":
        directories = {Path(row["path"]).parent for row in inventory["files"].values()}
        for directory in directories:
            os.chmod(directory, stat.S_IRWXU)
        for row in inventory["files"].values():
            path = Path(row["path"])
            os.chmod(path, stat.S_IRUSR)
            if _sha256_file(path) != row["sha256"]:
                raise ManifestMismatch(f"sealed role changed while inaccessible: {path}")
    grant = {
        "schema_version": "mode3-v6-3-sealed-grant-v1",
        "freeze_sha256": freeze_sha256,
        "role_manifest_sha256": str(role_manifest_sha256),
        "sealed_inventory_sha256": str(inventory["inventory_sha256"]),
        "allowed_phases": ["confirm", "semantic", "iid", "ood", "retrieval"],
        "refit_allowed": False,
    }
    target = root / "sealed" / "SEALED_ACCESS_GRANT.json"
    target.write_text(json.dumps(grant, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    RoleAccessGuard(root, role_manifest_sha256).assert_access("confirm", ["confirm_trigger"])
    return grant


def read_sealed_jsonl(
    output: Path,
    role: str,
    path: Path,
    *,
    role_manifest_sha256: str,
) -> list[dict[str, Any]]:
    RoleAccessGuard(Path(output), role_manifest_sha256).assert_access("confirm", [role])
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
