import hashlib
import json

import pytest

from sticky_lab.mode3_v6_3.errors import RoleLeakage
from sticky_lab.mode3_v6_3.roles import RoleAccessGuard, validate_nested_search_chains


def _row(i, source="s"):
    return {"text_id": f"t{i}", "document_id": f"d{i}", "source_id": source, "domain": "iid", "text": f"text {i}"}


def test_nested_search_chains_are_nested_and_independent():
    views = {}
    sizes = {}
    for stage, count in (("s0", 1), ("s1", 2), ("s2", 3), ("full", 4)):
        views[stage] = {
            "fit": [_row(i) for i in range(count)],
            "radius": [_row(10 + i) for i in range(count)],
            "score": [_row(20 + i) for i in range(count)],
        }
        sizes[stage] = {"fit": count, "radius": count, "score": count}
    validate_nested_search_chains(views, sizes)


def test_role_chain_overlap_is_fatal():
    views = {stage: {chain: [_row(0)] for chain in ("fit", "radius", "score")} for stage in ("s0", "s1", "s2", "full")}
    sizes = {stage: {chain: 1 for chain in ("fit", "radius", "score")} for stage in views}
    with pytest.raises(RoleLeakage):
        validate_nested_search_chains(views, sizes)


def test_confirm_roles_unreadable_before_freeze(tmp_path):
    guard = RoleAccessGuard(tmp_path, "roles")
    with pytest.raises(RoleLeakage):
        guard.assert_access("confirm", ["confirm_trigger"])


def test_sealed_grant_must_match_freeze(tmp_path):
    primary = tmp_path / "freeze" / "primary.json"
    primary.parent.mkdir()
    primary.write_text("{}\n")
    grant = tmp_path / "sealed" / "SEALED_ACCESS_GRANT.json"
    grant.parent.mkdir()
    grant.write_text(json.dumps({
        "freeze_sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
        "role_manifest_sha256": "roles",
    }))
    RoleAccessGuard(tmp_path, "roles").assert_access("confirm", ["confirm_trigger"])
    primary.write_text("{\"changed\":true}\n")
    with pytest.raises(RoleLeakage):
        RoleAccessGuard(tmp_path, "roles").assert_access("confirm", ["confirm_trigger"])
