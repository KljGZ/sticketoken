from __future__ import annotations

from pathlib import Path
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from sticky_lab.mode3_v6_2.freeze import create_freeze, load_freeze, save_freeze
from sticky_lab.mode3_v6_2.funnel import build_cap_archives, select_stage_models
from sticky_lab.mode3_v6_2.geometry import FrozenCapModel
from sticky_lab.mode3_v6_2.common import verified_checksum_tree
from sticky_lab.mode3_v6_2.encoding import pretruncate_source
from sticky_lab.mode3_v6_2.errors import CacheCorruption, ProtocolViolation
from sticky_lab.mode3_v6_2.oracle import load_embedding_cache, records_sha256, write_embedding_cache
from sticky_lab.mode3_v6_2.semantic import _bind_registered_nltk_resources
from sticky_lab.mode3_v6_2.statistics import p2_position_certificates


class _CanonicalizingTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": [0, 1] if add_special_tokens else self.encode(text, add_special_tokens=False)}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return {
            "raw": [10, 20, 30],
            "canonical-one": [10, 21, 30],
            "canonical-two": [10, 21, 30],
        }[text]

    def decode(self, values: list[int], **_: object) -> str:
        return "canonical-one" if values == [10, 20, 30] else "canonical-two"


class _CyclingTokenizer(_CanonicalizingTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return {"raw": [10], "left": [20], "right": [10]}[text]

    def decode(self, values: list[int], **_: object) -> str:
        return "left" if values == [10] else "right"


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


def test_pretruncate_freezes_tokenizer_canonicalization_fixed_point() -> None:
    text, ids, original_count = pretruncate_source(
        _CanonicalizingTokenizer(), "raw", maximum_length=8, trigger_overhead=1
    )
    assert text == "canonical-two"
    assert ids == [10, 21, 30]
    assert original_count == 3


def test_pretruncate_rejects_tokenizer_canonicalization_cycle() -> None:
    with pytest.raises(ProtocolViolation, match="canonicalization cycle"):
        pretruncate_source(_CyclingTokenizer(), "raw", maximum_length=8, trigger_overhead=1)


def test_nltk_runtime_is_bound_to_registered_archives(tmp_path: Path) -> None:
    root = tmp_path / "nltk_data"
    corpora = root / "corpora"
    corpora.mkdir(parents=True)
    archives = {
        "wordnet": corpora / "wordnet.zip",
        "omw-1.4": corpora / "omw-1.4.zip",
    }
    for name, path in archives.items():
        path.write_bytes(name.encode("ascii"))

    class FakeData:
        path = [str(tmp_path / "unregistered")]

        @staticmethod
        def find(resource: str) -> str:
            name = "wordnet" if "wordnet.zip" in resource else "omw-1.4"
            return f"{archives[name]}/{name}"

    config = {
        "resources": {
            "nltk_data": str(root),
            "files": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in archives.values()
            ],
        }
    }
    runtime = SimpleNamespace(data=FakeData())
    pointers = _bind_registered_nltk_resources(runtime, config)
    assert runtime.data.path[0] == str(root.resolve())
    assert set(pointers) == {"wordnet", "omw-1.4"}
    archives["wordnet"].write_bytes(b"tampered")
    with pytest.raises(ProtocolViolation, match="hash mismatch"):
        _bind_registered_nltk_resources(runtime, config)


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


def test_model_checksum_manifest_binds_exact_tree(tmp_path: Path) -> None:
    import hashlib

    root = tmp_path / "model"; root.mkdir()
    (root / "a").write_bytes(b"one"); (root / "b").write_bytes(b"two")
    manifest = tmp_path / "model.sha256"
    manifest.write_text("\n".join(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {root / name}"
        for name in ("a", "b")
    ) + "\n", encoding="utf-8")
    result = verified_checksum_tree(root, manifest)
    assert result["file_count"] == 2 and len(result["tree_sha256"]) == 64
    (root / "unregistered").write_text("x")
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        verified_checksum_tree(root, manifest)


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
