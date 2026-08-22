from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import re

import pytest

from sticky_lab.mode3_v6_3.errors import ManifestMismatch, ProtocolViolation, RoleLeakage
from sticky_lab.mode3_v6_3.report import atomic_json
from sticky_lab.mode3_v7.candidate_ranking import choose_primary_and_secondaries
from sticky_lab.mode3_v7 import cli as v7_cli
from sticky_lab.mode3_v7.config import (
    assert_output_leaf,
    config_for_profile,
    load_config,
)
from sticky_lab.mode3_v7.confirm import confirm_frozen_operating_point
from sticky_lab.mode3_v7.encoding import (
    build_confirm_call_space,
    build_discovery_call_space,
)
from sticky_lab.mode3_v7.freeze import load_freeze, write_freeze
from sticky_lab.mode3_v7.reuse import aggregate_fallback_candidates
from sticky_lab.mode3_v7.roles import (
    SEALED_ROLES,
    RoleAccessGuard,
    register_v7_roles,
    required_unique_capacity,
)
from sticky_lab.mode3_v7.tokenizer_audit import tokenizer_sha256

from test_mode3_v7_core import _synthetic_frontier, _vectors


class _Tokenizer:
    all_special_ids = [100, 101]

    def __init__(self):
        words = ["alpha", "beta", "gamma", "delta", "trigger"]
        self.vocab = {word: index + 1 for index, word in enumerate(words)}

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        values = [self.vocab.get(match.group(), 999) for match in re.finditer(r"\S+", str(text))]
        return [100] + values + [101] if add_special_tokens else values

    def decode(self, ids, **kwargs):
        reverse = {value: key for key, value in self.vocab.items()}
        return " ".join(reverse.get(int(value), "unknown") for value in ids)

    def __call__(self, text, add_special_tokens=True, **kwargs):
        matches = list(re.finditer(r"\S+", str(text)))
        ids = [self.vocab.get(match.group(), 999) for match in matches]
        offsets = [(match.start(), match.end()) for match in matches]
        if add_special_tokens:
            ids = [100] + ids + [101]
            offsets = [(0, 0)] + offsets + [(0, 0)]
        return {
            "input_ids": ids,
            "offset_mapping": offsets,
            "attention_mask": [1] * len(ids),
            "special_tokens_mask": [int(value in self.all_special_ids) for value in ids],
        }


def _config(profile: str = "formal") -> dict:
    base = load_config(Path("configs/v7_mode3_occupancy_frontier.yaml"))
    return config_for_profile(base, profile)


def _record(text_id: str, source: str = "s") -> dict[str, str]:
    return {
        "text_id": text_id,
        "document_id": f"doc-{text_id}",
        "source_id": source,
        "domain": "iid",
        "language": "en",
        "text_type": "sentence",
        "license": "test",
        "text": "alpha beta gamma delta",
    }


def _metric(token_id: int) -> dict:
    return {
        "token_id": token_id,
        "token_text": f"T{token_id}",
        "balanced_coverage": 0.9,
        "worst_position_coverage": 0.85,
        "benign_occupancy_core": 0.01,
        "radius_degrees": 20.0,
        "outside_to_inside": 0.8,
        "center_restart_spread": 0.01,
    }


def _metadata() -> dict:
    return {
        "maximum_radius_degrees": 35.0,
        "code_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "protocol_lock_sha256": "c" * 64,
        "role_manifest_sha256": "d" * 64,
        "discovery_role_hashes": {
            "fit": "fit-hash",
            "calibration": "calibration-hash",
            "select": "select-hash",
            "axis_fit_benign": "axis-hash",
        },
        "confirm_role_hashes": {
            "confirm_prefix": "confirm-prefix-hash",
            "confirm_suffix": "confirm-suffix-hash",
            "confirm_benign": "confirm-benign-hash",
            "confirm_paired": "confirm-paired-hash",
        },
        "tokenizer_sha256": "e" * 64,
        "model_revision": "f" * 40,
        "call_space_sha256": "1" * 64,
        "e_star_sha256": "2" * 64,
        "full_frontier_sha256": "3" * 64,
        "certification_thresholds": {
            "prefix_coverage_lcb": 0.8,
            "suffix_coverage_lcb": 0.8,
            "maximum_radius_degrees": 35.0,
            "actual_trigger_token_length": 1,
            "one_insertion_only": True,
        },
    }


def test_aggregate_fallback_always_fills_the_registered_512():
    metrics = [_metric(token_id) for token_id in range(1000)]
    selected, audit = aggregate_fallback_candidates(
        metrics, keep=512, deterministic_audit=64, seed=17
    )
    replay, _ = aggregate_fallback_candidates(
        metrics, keep=512, deterministic_audit=64, seed=17
    )
    assert selected == replay
    assert len(selected) == len(set(selected)) == 512
    assert audit["deterministic_audit"] == 64
    assert any(
        "deterministic_consensus_fill" in row["reasons"] for row in audit["selected"]
    )


def test_call_spaces_contain_no_random_and_keep_confirm_positions_disjoint():
    tokenizer = _Tokenizer()
    config = _config("dry_run")
    config["model"]["tokenizer_sha256"] = tokenizer_sha256(
        tokenizer, algorithm=config["model"]["tokenizer_hash_algorithm"]
    )
    discovery = build_discovery_call_space(
        tokenizer,
        {
            "fit": [_record("fit")],
            "calibration": [_record("cal")],
            "select": [_record("select")],
            "axis_fit_benign": [_record("axis")],
        },
        config,
    )
    assert {entry.key.position for entry in discovery.entries} == {
        "clean",
        "prefix",
        "suffix",
    }
    assert len(discovery.entries) == 7
    confirm = build_confirm_call_space(
        tokenizer,
        {
            "confirm_prefix": [_record("cp")],
            "confirm_suffix": [_record("cs")],
            "confirm_benign": [_record("cb")],
            "confirm_paired": [_record("pair")],
        },
        config,
    )
    positions = {
        role: {
            entry.key.position for entry in confirm.entries if entry.key.role == role
        }
        for role in SEALED_ROLES
    }
    assert positions["confirm_prefix"] == {"clean", "prefix"}
    assert positions["confirm_suffix"] == {"clean", "suffix"}
    assert positions["confirm_benign"] == {"clean"}
    assert positions["confirm_paired"] == {"clean", "prefix", "suffix"}


def test_dry_role_registry_is_document_disjoint_and_nested():
    config = _config("dry_run")
    required = required_unique_capacity(config)
    records = []
    for index in range(required + 200):
        digest = hashlib.sha256(f"payload-{index}".encode()).hexdigest()
        records.append(
            {
                **_record(str(index), source=f"source-{index % 4}"),
                "text": f"record {index} carries isolated payload {digest}",
            }
        )
    roles, views, audit = register_v7_roles(records, config, seed=29)
    documents = [
        str(row["document_id"])
        for rows in roles.values()
        for row in rows
    ]
    assert len(documents) == len(set(documents)) == required
    for chain in ("fit", "calibration", "select"):
        assert {
            row["text_id"] for row in views["s0"][chain]
        }.issubset({row["text_id"] for row in views["full"][chain]})
    assert audit["random_position_allocated"] is False


def test_output_and_sealed_access_fail_closed(tmp_path: Path):
    config = _config()
    assert_output_leaf(tmp_path / "mode3_v7_occupancy_frontier", config)
    with pytest.raises(ProtocolViolation):
        assert_output_leaf(tmp_path / "mode3_v6_3_light", config)
    with pytest.raises(ProtocolViolation):
        assert_output_leaf(
            tmp_path / "mode3_v6_3_light" / "mode3_v7_occupancy_frontier",
            config,
        )
    with pytest.raises(RoleLeakage):
        RoleAccessGuard(tmp_path, "role-hash").assert_access(
            "full", ["confirm_prefix"]
        )


def test_freeze_roundtrip_and_independent_confirmation(tmp_path: Path):
    base = _synthetic_frontier()
    frontiers = []
    for token_id in range(5):
        row = copy.deepcopy(base)
        row["token_id"] = token_id
        row["token_text"] = f"T{token_id}"
        frontiers.append(row)
    primary, secondaries = choose_primary_and_secondaries(frontiers)
    assert int(primary["token_id"]) == 0
    assert len(secondaries) == 4
    primary_path, freeze_sha = write_freeze(
        tmp_path, frontiers=frontiers, metadata=_metadata()
    )
    artifact = load_freeze(primary_path, freeze_sha)
    assert artifact.frozen_model().center_sha256 == base["center_hash"]

    prefix_rows = [
        _record(f"prefix-{source}-{index}", source)
        for source in ("a", "b")
        for index in range(500)
    ]
    suffix_rows = [
        _record(f"suffix-{source}-{index}", source)
        for source in ("a", "b")
        for index in range(500)
    ]
    benign_rows = [
        _record(f"benign-{source}-{index}", source)
        for source in ("a", "b")
        for index in range(5000)
    ]
    prefix_triggered = _vectors(0.03, len(prefix_rows))
    suffix_triggered = _vectors(-0.03, len(suffix_rows))
    prefix_clean = _vectors(1.0, len(prefix_rows))
    suffix_clean = _vectors(1.0, len(suffix_rows))
    benign = _vectors(1.0, len(benign_rows))
    hashes = {
        "confirm_prefix": "confirm-prefix-hash",
        "confirm_suffix": "confirm-suffix-hash",
        "confirm_benign": "confirm-benign-hash",
    }
    result = confirm_frozen_operating_point(
        artifact,
        prefix_rows=prefix_rows,
        prefix_triggered_vectors=prefix_triggered,
        prefix_clean_vectors=prefix_clean,
        suffix_rows=suffix_rows,
        suffix_triggered_vectors=suffix_triggered,
        suffix_clean_vectors=suffix_clean,
        benign_rows=benign_rows,
        benign_vectors=benign,
        observed_role_hashes=hashes,
        freeze_sha256=freeze_sha,
    )
    assert result["status"] == "CERTIFIED_V7_OCFCA_80"
    assert result["certified"] is True
    assert result["refit_performed"] is False

    failed = confirm_frozen_operating_point(
        artifact,
        prefix_rows=prefix_rows,
        prefix_triggered_vectors=prefix_triggered,
        prefix_clean_vectors=prefix_clean,
        suffix_rows=suffix_rows,
        suffix_triggered_vectors=_vectors(1.0, len(suffix_rows)),
        suffix_clean_vectors=suffix_clean,
        benign_rows=benign_rows,
        benign_vectors=benign,
        observed_role_hashes=hashes,
        freeze_sha256=freeze_sha,
    )
    assert failed["status"] == "VALID_PRIMARY_NOT_CERTIFIED"
    assert failed["gates"]["suffix_coverage"] is False
    with pytest.raises(ManifestMismatch):
        confirm_frozen_operating_point(
            artifact,
            prefix_rows=prefix_rows,
            prefix_triggered_vectors=prefix_triggered,
            prefix_clean_vectors=prefix_clean,
            suffix_rows=suffix_rows,
            suffix_triggered_vectors=suffix_triggered,
            suffix_clean_vectors=suffix_clean,
            benign_rows=benign_rows,
            benign_vectors=benign,
            observed_role_hashes={**hashes, "confirm_prefix": "wrong"},
            freeze_sha256=freeze_sha,
        )


def test_cache_compaction_removes_only_nonselected_v7_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "mode3_v7_occupancy_frontier"
    atomic_json(output / "stages" / "full" / "COMPLETE.json", {"status": "complete"})
    atomic_json(
        output / "diagnostics" / "post_selection" / "COMPLETE.json",
        {"status": "complete"},
    )
    atomic_json(
        output / "v7_top20_token_beta_pairs.json",
        {"pairs": [{"token_id": 1}, {"token_id": 2}]},
    )
    for token_id in (-2, 1, 2, 3, 4):
        directory = output / "embedding_cache" / f"token_{token_id}"
        directory.mkdir(parents=True)
        (directory / "chunk.vectors.npy").write_bytes(b"v7-cache")
    protected = tmp_path / "mode3_v6_3_light" / "embedding_cache" / "token_3"
    protected.mkdir(parents=True)
    (protected / "keep.bin").write_bytes(b"v6")
    monkeypatch.setattr(
        v7_cli,
        "_validated_full_selection",
        lambda output, config: (
            [],
            [{"token_id": 1}, {"token_id": 2}],
        ),
    )

    v7_cli.command_compact_cache(
        argparse.Namespace(output=str(output)), _config("dry_run")
    )

    assert (output / "embedding_cache" / "token_-2").is_dir()
    assert (output / "embedding_cache" / "token_1").is_dir()
    assert (output / "embedding_cache" / "token_2").is_dir()
    assert not (output / "embedding_cache" / "token_3").exists()
    assert not (output / "embedding_cache" / "token_4").exists()
    assert (protected / "keep.bin").read_bytes() == b"v6"
    complete = (
        output / "cache_compaction" / "COMPLETE.json"
    ).read_text(encoding="utf-8")
    assert '"v6_paths_touched": false' in complete
