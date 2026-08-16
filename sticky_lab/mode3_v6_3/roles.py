"""Nested search chains and irreversible sealed-role access for V6.3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import canonical_sha256
from .errors import ManifestMismatch, RoleLeakage


STAGES = ("s0", "s1", "s2", "full")
CHAINS = ("fit", "radius", "score")
SEALED_ROLES = (
    "confirm_trigger", "confirm_benign", "paired_position_audit",
    "semantic_control", "iid_replication_0", "iid_replication_1",
    "iid_replication_2", "retrieval_probe",
)


def records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    rows = [{
        "text_id": str(row["text_id"]),
        "document_id": str(row["document_id"]),
        "source_id": str(row["source_id"]),
        "domain": str(row["domain"]),
        "text_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
    } for row in records]
    return canonical_sha256(rows)


@dataclass(frozen=True)
class RoleBinding:
    role: str
    role_class: str
    count: int
    records_sha256: str
    document_count: int
    source_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_ids"] = list(self.source_ids)
        return value


def bind_role(role: str, role_class: str, records: Sequence[Mapping[str, Any]]) -> RoleBinding:
    return RoleBinding(
        str(role), str(role_class), len(records), records_sha256(records),
        len({(str(r["source_id"]), str(r["document_id"])) for r in records}),
        tuple(sorted({str(r["source_id"]) for r in records})),
    )


def validate_nested_search_chains(
    views: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    expected_sizes: Mapping[str, Mapping[str, int]],
) -> None:
    for chain in CHAINS:
        previous: set[str] = set()
        for stage in STAGES:
            records = views[stage][chain]
            ids = [str(row["text_id"]) for row in records]
            if len(ids) != len(set(ids)):
                raise ManifestMismatch(f"duplicate text in {stage}/{chain}")
            if len(ids) != int(expected_sizes[stage][chain]):
                raise ManifestMismatch(f"size mismatch in {stage}/{chain}")
            current = set(ids)
            if not previous.issubset(current):
                raise ManifestMismatch(f"{stage}/{chain} is not nested")
            previous = current
    full_sets = {
        chain: {str(row["document_id"]) for row in views["full"][chain]}
        for chain in CHAINS
    }
    for index, left in enumerate(CHAINS):
        for right in CHAINS[index + 1:]:
            overlap = full_sets[left].intersection(full_sets[right])
            if overlap:
                raise RoleLeakage(f"search chains {left}/{right} share documents")


def build_role_manifest(bindings: Iterable[RoleBinding], nested: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "mode3-v6-3-role-manifest-v1",
        "graph": "fit|radius|score -> freeze -> confirm|iid|ood|semantic|retrieval",
        "nested_search_views": nested,
        "bindings": {binding.role: binding.to_dict() for binding in sorted(bindings, key=lambda x: x.role)},
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


class RoleAccessGuard:
    """Denies every sealed role until a hash-bound access grant exists."""

    def __init__(self, output: Path, role_manifest_sha256: str) -> None:
        self.output = Path(output)
        self.role_manifest_sha256 = str(role_manifest_sha256)

    @property
    def freeze_path(self) -> Path:
        return self.output / "freeze" / "primary.json"

    @property
    def grant_path(self) -> Path:
        return self.output / "sealed" / "SEALED_ACCESS_GRANT.json"

    def assert_access(self, phase: str, roles: Iterable[str]) -> None:
        requested = tuple(map(str, roles))
        sealed = [r for r in requested if r in SEALED_ROLES or r.startswith("ood_")]
        if not sealed:
            return
        if str(phase) in {"preflight", "enumerate", "s0", "s1", "s2", "full", "ranking", "freeze"}:
            raise RoleLeakage(f"{phase} attempted sealed roles {sealed}")
        if not self.freeze_path.is_file() or not self.grant_path.is_file():
            raise RoleLeakage("sealed roles remain unreadable before freeze and grant")
        freeze_sha256 = hashlib.sha256(self.freeze_path.read_bytes()).hexdigest()
        grant = json.loads(self.grant_path.read_text(encoding="utf-8"))
        if grant.get("freeze_sha256") != freeze_sha256:
            raise RoleLeakage("sealed grant is not bound to current primary freeze")
        if grant.get("role_manifest_sha256") != self.role_manifest_sha256:
            raise RoleLeakage("sealed grant role-manifest hash mismatch")
