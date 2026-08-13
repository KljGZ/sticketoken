from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from sticky_lab.mode3_v6_compact.evaluate import DiscoveryMetric, attach_benign_metrics
from sticky_lab.mode3_v6_compact.oracle import load_embedding_cache, records_sha256, write_embedding_cache


def test_embedding_cache_binds_role_and_records(tmp_path: Path) -> None:
    records = [{"text_id": "a", "text": "one"}, {"text_id": "b", "text": "two"}]
    vectors = np.eye(2, dtype=np.float32)
    path = tmp_path / "role.npy"
    write_embedding_cache(path, vectors, role="role", records_hash=records_sha256(records), model_revision="r")
    loaded = load_embedding_cache(path, expected_role="role", expected_records_hash=records_sha256(records), mmap=False)
    assert np.allclose(loaded, vectors)


def test_vectorized_benign_metrics_use_frozen_high_dimensional_centers() -> None:
    metric = DiscoveryMetric(1, "x", "s0", "valid", np.pi / 4, 45.0, 0.9, 0.9, 0.0, 0.9, 0.8, 0.8)
    benign = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    values = attach_benign_metrics([metric], np.array([[1.0, 0.0]]), np.array([np.pi / 4]), benign, device="cpu")
    assert values[0].benign_occupancy == pytest.approx(1 / 3)
    assert values[0].search_margin_m90_1 is not None


def test_discovery_metric_has_no_reduced_dimension_fields() -> None:
    names = set(DiscoveryMetric.__dataclass_fields__)
    assert "pca" not in names
    assert "umap" not in names
    assert {"radius_radians", "triggered_coverage", "benign_occupancy"} <= names
