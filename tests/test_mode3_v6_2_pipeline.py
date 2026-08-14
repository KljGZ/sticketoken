from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sticky_lab.mode3_v6_2.freeze import create_freeze, load_freeze, save_freeze
from sticky_lab.mode3_v6_2.funnel import build_cap_archives, select_stage_models
from sticky_lab.mode3_v6_2.geometry import FrozenCapModel
from sticky_lab.mode3_v6_2.errors import CacheCorruption
from sticky_lab.mode3_v6_2.oracle import load_embedding_cache, records_sha256, write_embedding_cache
from sticky_lab.mode3_v6_2.statistics import p2_position_certificates


def _row(token: int, caps: int, coverage: float) -> dict:
    return {"token_id": token, "cap_count": caps, "status": "valid", "coverage_margin": coverage - .9, "worst_position_coverage": coverage - .01, "outside_to_inside": coverage - .03, "semantic_anomaly": token / 1000, "radius_degrees": 10 + caps, "benign_occupancy": .001 * caps, "benign_occupancy_1_10": .002 * caps, "occupancy_auc_1_1_5": .003 * caps, "center_drift_from_previous": .001 * token}


def test_funnel_keeps_independent_one_and_multicap_archives() -> None:
    rows = [_row(token, caps, .91 + token / 1000) for token in range(12) for caps in (1, 2, 3, 4)]
    archives = build_cap_archives(rows)
    assert archives["one_cap_pareto"] and archives["multi_cap_pareto"]
    selected, audit = select_stage_models(rows, 8)
    assert len(selected) == 8 and len({token for token, _ in selected}) == 8
    assert any("one_cap" in label for labels in audit["provenance"].values() for label in labels)
    assert any("multi_cap" in label for labels in audit["provenance"].values() for label in labels)


def test_embedding_cache_is_bound_to_role_and_records(tmp_path: Path) -> None:
    records = [{"text_id": "a", "text": "one"}, {"text_id": "b", "text": "two"}]
    path = tmp_path / "x.npy"; write_embedding_cache(path, np.eye(2), role="role", records_hash=records_sha256(records), model_revision="r")
    assert np.allclose(load_embedding_cache(path, expected_role="role", expected_records_hash=records_sha256(records), mmap=False), np.eye(2))


def test_embedding_cache_rejects_content_tampering(tmp_path: Path) -> None:
    records = [{"text_id": "a", "text": "one"}]
    path = tmp_path / "x.npy"
    write_embedding_cache(path, np.ones((1, 2)), role="role", records_hash=records_sha256(records), model_revision="r")
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(CacheCorruption, match="SHA-256"):
        load_embedding_cache(path, expected_role="role", expected_records_hash=records_sha256(records), mmap=False)


def test_freeze_is_content_addressed(tmp_path: Path) -> None:
    cap = FrozenCapModel(3, " x", "P3_ST_FCA_Core", np.array([[1., 0.]]), np.array([.2]), .92, "fit", "radius", 1)
    artifact = create_freeze(cap, tokenizer_hash="t", model_hash="m", code_commit="c", data_role_hashes={"confirm": "h"}, position_manifest_hash="p", random_boundary_manifest_hash="r", source_weights={"a": .5, "b": .5}, selection_metrics={"coverage": .93}, certification_thresholds={"p3_balanced_coverage_lcb": .9})
    path = tmp_path / "freeze.json"; save_freeze(path, artifact); loaded = load_freeze(path)
    assert loaded.freeze_sha256 == artifact.freeze_sha256 and loaded.cap.token_id == 3


def test_p2_uses_simultaneous_position_correction() -> None:
    values = {position: {source: np.ones(200, dtype=bool) for source in ("a", "b")} for position in ("prefix", "suffix", "random")}
    result = p2_position_certificates(values, familywise_alpha=.05)
    assert result["simultaneous_all_positions"] and result["correction"] == "bonferroni"
    assert all(result[position]["familywise_position_alpha"] == pytest.approx(.05 / 3) for position in values)
    assert all(result[position]["worst_source_lcb"] <= result[position]["balanced_lcb"] for position in values)
