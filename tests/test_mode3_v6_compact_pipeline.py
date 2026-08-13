from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from sticky_lab.mode3_v6_compact.evaluate import DiscoveryMetric, attach_benign_metrics
from sticky_lab.mode3_v6_compact.oracle import load_embedding_cache, records_sha256, write_embedding_cache
from sticky_lab.mode3_v6_compact.track_whitebox import _active_embedding_output
from sticky_lab.mode3_v6_compact.workers import _enumerate_limited_single_tokens


class _TinyTokenizer:
    all_special_ids: list[int] = []

    def __init__(self) -> None:
        self.decode_calls = 0

    def get_vocab(self) -> dict[str, int]:
        return {f"t{index}": index for index in range(100)}

    def decode(self, token_ids: list[int], **_: object) -> str:
        self.decode_calls += 1
        return f"t{token_ids[0]}"

    def encode(self, text: str, **_: object) -> list[int]:
        return [int(text[1:])]


def test_dry_enumeration_limit_stops_after_enough_legal_tokens() -> None:
    tokenizer = _TinyTokenizer()
    unrestricted, visible = _enumerate_limited_single_tokens(
        tokenizer,
        context_records=[],
        manifest=object(),
        role="s0_fit",
        exclude_special=True,
        limit=7,
    )
    assert [row.token_id for row in unrestricted] == list(range(7))
    assert [row.token_id for row in visible] == list(range(7))
    assert tokenizer.decode_calls == 7


def test_whitebox_selects_the_captured_output_with_an_active_gradient() -> None:
    import torch

    inactive = torch.zeros((2, 3, 4), requires_grad=True)
    active = torch.ones((2, 3, 4), requires_grad=True)
    active.retain_grad()
    active.sum().backward()
    mask = torch.zeros((2, 3), dtype=torch.bool)
    assert _active_embedding_output([inactive, active], mask) is active


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
