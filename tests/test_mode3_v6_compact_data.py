from __future__ import annotations

import pytest

from sticky_lab.mode3_v6.deduplication import audit_role_leakage
from sticky_lab.mode3_v6_compact.data import NearDuplicateIndex, register_compact_roles


def row(index: int, text: str, *, domain: str = "iid", source: str = "iid_source") -> dict[str, str]:
    return {
        "text": text,
        "text_id": str(index),
        "document_id": f"doc-{index}",
        "source_id": source,
        "domain": domain,
        "language": "en",
        "text_type": "sentence",
        "license": "test",
    }


def test_near_duplicate_index_rejects_verified_paraphrase() -> None:
    index = NearDuplicateIndex(0.80)
    original = "one two three four five six seven eight nine ten"
    index.add(original)
    conflict, reason = index.conflicts("one two three four five six seven eight nine ten eleven")
    assert conflict
    assert reason == "near_duplicate"


def test_compact_registration_prevents_cross_role_leakage() -> None:
    iid = [row(i, f"iid unique sentence number {i} alpha beta gamma delta epsilon") for i in range(16)]
    # An exact/near duplicate appears later in the document order and must be rejected.
    iid.append(row(100, iid[0]["text"]))
    ood0 = [row(1000 + i, f"news unique sentence {i} apple banana cherry delta echo", domain="ood_a", source="ood_a_source") for i in range(8)]
    config = {
        "data": {
            "maximum_near_duplicate_jaccard": 0.80,
            "allocation_policy": "document_disjoint_online_near_duplicate_rejection",
            "roles": {"s0_fit": 3, "s0_eval": 3, "discovery_benign": 3},
            "ood_domains": 1,
            "ood_domains_allowlist": ["ood_a"],
            "ood_trigger_per_domain": 2,
            "ood_benign_per_domain": 2,
        }
    }
    roles, audit = register_compact_roles([*iid, *ood0], config, seed=9)
    assert not audit_role_leakage(roles, 0.80)
    assert audit["accepted_rows"] == sum(map(len, roles.values()))
    documents = [entry["document_id"] for values in roles.values() for entry in values]
    assert len(documents) == len(set(documents))


def test_compact_registration_fails_instead_of_resampling() -> None:
    config = {
        "data": {
            "maximum_near_duplicate_jaccard": 0.80,
            "allocation_policy": "document_disjoint_online_near_duplicate_rejection",
            "roles": {"s0_fit": 10},
            "ood_domains": 0,
            "ood_domains_allowlist": [],
            "ood_trigger_per_domain": 0,
            "ood_benign_per_domain": 0,
        }
    }
    with pytest.raises(RuntimeError, match="capacity gap"):
        register_compact_roles([row(1, "only one record alpha beta gamma delta epsilon")], config, seed=1)
