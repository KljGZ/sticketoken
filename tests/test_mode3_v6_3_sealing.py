import hashlib
import inspect
import json

import numpy as np
import pytest

from sticky_lab.mode3_v6_3 import confirm
from sticky_lab.mode3_v6_3.errors import ManifestMismatch, RoleLeakage
from sticky_lab.mode3_v6_3.freeze import FreezeArtifact, load_freeze
from sticky_lab.mode3_v6_3.geometry import FrozenCap
from sticky_lab.mode3_v6_3.sealing import assert_still_sealed


def _artifact():
    cap = FrozenCap(1, "x", np.asarray([1.0, 0, 0]), .2, "fit", "radius", "top100").to_dict()
    return FreezeArtifact(
        "mode3-v6-3-freeze-v1", "primary", cap, {"token_id": 1}, "commit",
        "config", "roles", {"fit": "fit"}, {"confirm_trigger": "ct"},
        "tokenizer", "revision", "calls", {
            "balanced_coverage_lcb": .9, "worst_position_lcb": .85,
            "worst_source_lcb": .8, "independent_benign_core_ucb": .01,
            "outside_to_inside_lcb": .85, "conditional_outside_origin_lcb": .95,
            "maximum_radius_degrees": 35, "moat_occupancy_1_10_ucb": .05,
            "basin_lambda_star": 1.5, "basin_occupancy_auc_1_1_5": .03,
            "central_collapse_median_depth": .8,
        }, False,
    )


def _write(tmp_path, artifact):
    path = tmp_path / "primary.json"
    path.write_text(json.dumps(artifact.to_dict(), sort_keys=True) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_no_confirm_base_embedding_before_freeze(tmp_path):
    (tmp_path / "embedding_cache" / "confirm").mkdir(parents=True)
    (tmp_path / "embedding_cache" / "confirm" / "bad.npy").write_bytes(b"x")
    with pytest.raises(RoleLeakage):
        assert_still_sealed(tmp_path)


def test_confirm_requires_freeze_hash(tmp_path):
    path, _ = _write(tmp_path, _artifact())
    with pytest.raises(ManifestMismatch):
        load_freeze(path, "0" * 64)


def test_confirm_refuses_modified_center(tmp_path):
    artifact = _artifact()
    artifact.cap["center"] = [0, 1, 0]
    path, digest = _write(tmp_path, artifact)
    with pytest.raises(ManifestMismatch):
        load_freeze(path, digest)


def test_confirm_refuses_modified_radius(tmp_path):
    artifact = _artifact()
    artifact.cap["radius"] = .4
    path, digest = _write(tmp_path, artifact)
    with pytest.raises(ManifestMismatch):
        load_freeze(path, digest)


def test_confirm_refuses_modified_non_geometry_freeze_field(tmp_path):
    artifact = _artifact()
    path, _ = _write(tmp_path, artifact)
    payload = json.loads(path.read_text())
    payload["selection_rank"] = 99
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ManifestMismatch):
        load_freeze(path, digest)


def test_confirm_imports_no_fit_functions():
    source = inspect.getsource(confirm)
    assert "fit_single_cap" not in source
    assert "fit_robust_shared_center" not in source
