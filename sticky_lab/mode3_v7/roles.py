"""V7 data-role graph built on the audited V6.3 allocator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sticky_lab.mode3_v6_3.data_contract import register_v63_roles
from sticky_lab.mode3_v6_3.errors import ManifestMismatch, RoleLeakage

from .config import canonical_sha256


DISCOVERY_ROLES = ("fit", "calibration", "select", "axis_fit_benign")
SEALED_ROLES = (
    "confirm_prefix",
    "confirm_suffix",
    "confirm_benign",
    "confirm_paired",
)
STAGES = ("s0", "full")
CHAINS = ("fit", "calibration", "select")
_TO_V63 = {
    "fit": "fit",
    "calibration": "radius",
    "select": "score",
    "axis_fit_benign": "discovery_benign",
}
_FROM_V63 = {value: key for key, value in _TO_V63.items()}


def records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        {
            "text_id": str(row["text_id"]),
            "document_id": str(row["document_id"]),
            "source_id": str(row["source_id"]),
            "domain": str(row["domain"]),
            "text_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
        }
        for row in records
    ]
    return canonical_sha256(rows)


def _allocator_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    data = value["data"]
    registered = data["stage_sizes"]
    stage_sizes = {
        "s0": {
            "fit": int(registered["s0"]["fit"]),
            "radius": int(registered["s0"]["calibration"]),
            "score": int(registered["s0"]["select"]),
        },
        "s1": {
            "fit": int(registered["s0"]["fit"]),
            "radius": int(registered["s0"]["calibration"]),
            "score": int(registered["s0"]["select"]),
        },
        "s2": {
            "fit": int(registered["s0"]["fit"]),
            "radius": int(registered["s0"]["calibration"]),
            "score": int(registered["s0"]["select"]),
        },
        "full": {
            "fit": int(registered["full"]["fit"]),
            "radius": int(registered["full"]["calibration"]),
            "score": int(registered["full"]["select"]),
        },
    }
    data["search_chain_sizes"] = stage_sizes
    data["discovery_benign"] = int(data["axis_fit_benign"])
    data.setdefault("ood_domains", len(data.get("ood_domains_allowlist", [])))
    data.setdefault("minimum_ood_sources", 0)
    data.setdefault("ood_trigger_per_domain", 0)
    data.setdefault("ood_benign_per_domain", 0)
    return value


def _rename_rows(rows: Sequence[Mapping[str, Any]], old: str, new: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if row.get("role_chain") == old:
            row["role_chain"] = new
        if row.get("registered_role") == old:
            row["registered_role"] = new
        view = str(row.get("registered_view", ""))
        if view.endswith(f"_{old}"):
            row["registered_view"] = view[: -len(old)] + new
        result.append(row)
    return result


def register_v7_roles(
    records: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[str, Any],
]:
    """Allocate disjoint roles without reimplementing near-duplicate controls."""

    transformed = _allocator_config(config)
    old_roles, old_views, allocation = register_v63_roles(
        records, transformed, seed=int(seed)
    )
    roles: dict[str, list[dict[str, Any]]] = {}
    for old, new in _FROM_V63.items():
        roles[new] = _rename_rows(old_roles[old], old, new)
    for role in SEALED_ROLES:
        roles[role] = _rename_rows(old_roles[role], role, role)
    views: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for stage in STAGES:
        views[stage] = {}
        for old, new in (("fit", "fit"), ("radius", "calibration"), ("score", "select")):
            views[stage][new] = _rename_rows(old_views[stage][old], old, new)
    validate_nested_roles(views, config["data"]["stage_sizes"])
    audit = dict(allocation)
    audit.update(
        {
            "schema_version": "mode3-v7-allocation-v1",
            "policy": "fit|calibration|select_disjoint_with_nested_s0_full_views",
            "v6_3_allocator_reused": True,
            "semantic_role_mapping": dict(_TO_V63),
            "random_position_allocated": False,
            "role_counts": {role: len(rows) for role, rows in roles.items()},
        }
    )
    return roles, views, audit


def validate_nested_roles(
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
    full_documents = {
        chain: {str(row["document_id"]) for row in views["full"][chain]}
        for chain in CHAINS
    }
    for index, left in enumerate(CHAINS):
        for right in CHAINS[index + 1 :]:
            if full_documents[left].intersection(full_documents[right]):
                raise RoleLeakage(f"V7 discovery roles {left}/{right} share documents")


def required_unique_capacity(config: Mapping[str, Any]) -> int:
    data = config["data"]
    total = sum(int(value) for value in data["stage_sizes"]["full"].values())
    total += int(data["axis_fit_benign"])
    total += sum(int(value) for value in data["confirm_roles"].values())
    return total


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


def bind_role(
    role: str, role_class: str, records: Sequence[Mapping[str, Any]]
) -> RoleBinding:
    return RoleBinding(
        str(role),
        str(role_class),
        len(records),
        records_sha256(records),
        len({(str(row["source_id"]), str(row["document_id"])) for row in records}),
        tuple(sorted({str(row["source_id"]) for row in records})),
    )


def build_role_manifest(
    bindings: Iterable[RoleBinding],
    nested: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "mode3-v7-role-manifest-v1",
        "graph": (
            "triggered_fit + benign_calibration + triggered_select + axis_fit_benign "
            "-> freeze(token,beta,center,radius) -> confirm_prefix|confirm_suffix|"
            "confirm_benign|confirm_paired"
        ),
        "nested_search_views": nested,
        "bindings": {
            binding.role: binding.to_dict()
            for binding in sorted(bindings, key=lambda item: item.role)
        },
        "positions": ["prefix", "suffix"],
        "random_position_enabled": False,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


class RoleAccessGuard:
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
        sealed = [str(role) for role in roles if str(role) in SEALED_ROLES]
        if not sealed:
            return
        if str(phase) in {"prepare", "enumerate", "reuse", "s0", "full", "ranking", "freeze"}:
            raise RoleLeakage(f"{phase} attempted sealed V7 roles {sealed}")
        if not self.freeze_path.is_file() or not self.grant_path.is_file():
            raise RoleLeakage("V7 confirm remains sealed before freeze and grant")
        freeze_sha256 = hashlib.sha256(self.freeze_path.read_bytes()).hexdigest()
        grant = json.loads(self.grant_path.read_text(encoding="utf-8"))
        if grant.get("freeze_sha256") != freeze_sha256:
            raise RoleLeakage("V7 sealed grant is not bound to current freeze")
        if grant.get("role_manifest_sha256") != self.role_manifest_sha256:
            raise RoleLeakage("V7 sealed grant role-manifest hash mismatch")
