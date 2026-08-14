"""Irreversible role graph and manifest bindings for V6.2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import ManifestMismatch, ProtocolViolation


DISCOVERY_ORDER = (
    "s0_fit", "s0_radius", "s0_score",
    "s1_fit", "s1_radius", "s1_score",
    "s2_fit", "s2_radius", "s2_score",
    "full_fit", "full_radius", "full_select", "discovery_benign",
    "semantic_control",
)
SEALED_ROLES = (
    "confirm_trigger", "confirm_benign",
    "iid_replication_0", "iid_replication_1", "iid_replication_2",
    "semantic_confirm", "retrieval_probe",
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def records_sha256(records: Sequence[Mapping[str, str]]) -> str:
    rows = [
        {
            "text_id": str(row["text_id"]),
            "document_id": str(row["document_id"]),
            "source_id": str(row["source_id"]),
            "domain": str(row["domain"]),
            "text_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
            "encoding_text_sha256": hashlib.sha256(
                str(row.get("encoding_text", row["text"])).encode("utf-8")
            ).hexdigest(),
        }
        for row in records
    ]
    return canonical_sha256(rows)


@dataclass(frozen=True)
class RoleBinding:
    role: str
    count: int
    records_sha256: str
    source_ids: tuple[str, ...]
    document_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "count": self.count,
            "records_sha256": self.records_sha256,
            "source_ids": list(self.source_ids),
            "document_count": self.document_count,
        }


def bind_role(role: str, records: Sequence[Mapping[str, str]]) -> RoleBinding:
    return RoleBinding(
        role=str(role),
        count=len(records),
        records_sha256=records_sha256(records),
        source_ids=tuple(sorted({str(row["source_id"]) for row in records})),
        document_count=len({(str(row["source_id"]), str(row["document_id"])) for row in records}),
    )


class RoleGraph:
    """Checks that sealed roles cannot be accessed before a verified freeze."""

    _DISCOVERY_PHASES = {"prepare", "enumerate", "s0", "s1", "s2", "full", "selection", "semantic_discovery"}
    _SEALED_PHASES = {"confirm", "iid", "ood", "semantic_confirm", "retrieval", "finalize"}

    def __init__(self, bindings: Mapping[str, RoleBinding], freeze_path: Path) -> None:
        self.bindings = dict(bindings)
        self.freeze_path = Path(freeze_path)

    def assert_access(self, phase: str, roles: Iterable[str]) -> None:
        phase = str(phase)
        requested = tuple(map(str, roles))
        unknown = [role for role in requested if role not in self.bindings and not role.startswith("ood_")]
        if unknown:
            raise ProtocolViolation(f"unregistered role access: {unknown}")
        sealed = [role for role in requested if role in SEALED_ROLES or role.startswith("ood_")]
        if phase in self._DISCOVERY_PHASES and sealed:
            raise ProtocolViolation(f"discovery phase {phase} attempted sealed role access: {sealed}")
        if phase in self._SEALED_PHASES and not self.freeze_path.is_file():
            raise ProtocolViolation(f"sealed phase {phase} requires a verified freeze artifact")

    def assert_binding(self, role: str, records: Sequence[Mapping[str, str]]) -> None:
        observed = bind_role(role, records)
        expected = self.bindings.get(role)
        if expected is None or expected != observed:
            raise ManifestMismatch(
                f"role manifest mismatch for {role}: expected={expected}, observed={observed}"
            )


def build_role_contract(roles: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, Any]:
    bindings = {name: bind_role(name, records) for name, records in sorted(roles.items())}
    payload = {
        "schema_version": "mode3-v6-2-role-contract-v1",
        "irreversible_graph": "discovery->full_fit->full_radius->full_select->freeze->sealed",
        "bindings": {name: binding.to_dict() for name, binding in bindings.items()},
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def load_role_contract(path: Path) -> dict[str, RoleBinding]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    registered = payload.pop("contract_sha256", None)
    if registered != canonical_sha256(payload):
        raise ManifestMismatch("role contract hash mismatch")
    return {
        name: RoleBinding(
            role=str(row["role"]), count=int(row["count"]),
            records_sha256=str(row["records_sha256"]),
            source_ids=tuple(map(str, row["source_ids"])),
            document_count=int(row["document_count"]),
        )
        for name, row in payload["bindings"].items()
    }
