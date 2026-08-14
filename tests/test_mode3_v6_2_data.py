from __future__ import annotations

import json
from pathlib import Path

import pytest

from sticky_lab.mode3_v6.deduplication import audit_role_leakage
from sticky_lab.mode3_v6_2.data import NearDuplicateIndex, register_v62_roles
from sticky_lab.mode3_v6_2.errors import ManifestMismatch, ProtocolViolation
from sticky_lab.mode3_v6_2.roles import RoleGraph, build_role_contract, load_role_contract


def row(index: int, source: str, domain: str = "iid") -> dict[str, str]:
    return {"text": f"unique sample {source} {index} alpha beta gamma delta epsilon zeta", "text_id": f"{source}-{index}", "document_id": f"{source}-doc-{index}", "source_id": source, "domain": domain, "language": "en", "text_type": "sentence", "license": "test"}


def test_near_duplicate_index_verifies_candidates() -> None:
    index = NearDuplicateIndex(.80); index.add("one two three four five six seven eight nine ten")
    assert index.conflicts("one two three four five six seven eight nine ten eleven") == (True, "near_duplicate")


def test_registration_requires_real_iid_sources_and_disjoint_ood() -> None:
    records = [row(i, source) for source in ("iid_a", "iid_b") for i in range(30)]
    records += [row(i, "ood_source", "ood_news") for i in range(20)]
    config = {"data": {"maximum_near_duplicate_jaccard": .80, "allocation_policy": "document_disjoint_online_near_duplicate_rejection", "roles": {"s0_fit": 5, "s0_radius": 5, "s0_score": 5}, "minimum_iid_sources": 2, "ood_domains": 1, "ood_domains_allowlist": ["ood_news"], "ood_trigger_per_domain": 4, "ood_benign_per_domain": 4, "near_duplicate_manual_audit_pairs": 10}}
    roles, audit = register_v62_roles(records, config, seed=5)
    assert set(audit["iid_sources"]) == {"iid_a", "iid_b"}
    for role in ("s0_fit", "s0_radius", "s0_score"):
        counts = audit["allocation"][role]["source_counts"]
        assert set(counts) == {"iid_a", "iid_b"}
        assert max(counts.values()) - min(counts.values()) <= 1
    assert not audit_role_leakage(roles, .80)
    assert len({row["document_id"] for values in roles.values() for row in values}) == sum(map(len, roles.values()))


def test_registration_refuses_fake_single_iid_source() -> None:
    records = [row(i, "only") for i in range(30)]
    config = {"data": {"maximum_near_duplicate_jaccard": .80, "allocation_policy": "x", "roles": {"s0_fit": 2}, "minimum_iid_sources": 2, "ood_domains": 0, "ood_domains_allowlist": [], "ood_trigger_per_domain": 0, "ood_benign_per_domain": 0}}
    with pytest.raises(RuntimeError, match="requires 2 IID sources"): register_v62_roles(records, config, seed=1)


def test_role_contract_detects_tampering_and_seals_confirmation(tmp_path: Path) -> None:
    roles = {"s0_fit": [row(1, "a")], "confirm_trigger": [row(2, "b")]}
    contract = build_role_contract(roles); path = tmp_path / "roles.json"; path.write_text(json.dumps(contract), encoding="utf-8")
    bindings = load_role_contract(path); graph = RoleGraph(bindings, tmp_path / "freeze.json")
    with pytest.raises(ProtocolViolation): graph.assert_access("s0", ["confirm_trigger"])
    with pytest.raises(ProtocolViolation): graph.assert_access("confirm", ["confirm_trigger"])
    (tmp_path / "freeze.json").write_text("{}")
    graph.assert_access("confirm", ["confirm_trigger"])
    with pytest.raises(ManifestMismatch): graph.assert_binding("s0_fit", [row(3, "a")])
