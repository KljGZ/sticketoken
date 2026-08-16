import json
from pathlib import Path

import numpy as np
import pytest

from sticky_lab.mode3_v6_3.budget import BudgetLedger
from sticky_lab.mode3_v6_3.cache import (
    CacheKey,
    CallRegistry,
    CallSpace,
    CallSpaceEntry,
    EmbeddingCache,
)
from sticky_lab.mode3_v6_3.errors import BudgetHardStop, DuplicateEncoderCallConflict
from sticky_lab.mode3_v6_3.config import config_for_profile, load_config
from sticky_lab.mode3_v6_3.report import atomic_json, result_inventory


def _settings(hard=100):
    return {"warning_limit": max(1, hard - 1), "hard_limit": hard, "forbidden_limit": hard + 10}


def _space():
    entries = []
    for index, position in enumerate(("prefix", "random")):
        key = CacheKey(
            "revision", "tokenizer", -1, f"text-{index}", "fit", position,
            position, "source-hash", "insertion-hash", "float32", "eager",
        )
        entries.append(CallSpaceEntry(index, key))
    return CallSpace(entries)


def test_duplicate_encoder_call_rejected(tmp_path):
    ledger = BudgetLedger(tmp_path, _settings())
    registry = CallRegistry(tmp_path, _space(), ledger)
    registry.reserve(7, [0], phase="s0")
    with pytest.raises(DuplicateEncoderCallConflict):
        registry.reserve(7, [0], phase="s0")


def test_budget_reserved_before_model_call(tmp_path):
    ledger = BudgetLedger(tmp_path, _settings())
    reservation = ledger.reserve(phase="s0", raw_items=3)
    assert reservation.total_after == 3
    assert json.loads((tmp_path / "budget" / "observed.json").read_text())["submitted_text_equivalent"] == 3


def test_hard_stop_blocks_new_calls(tmp_path):
    ledger = BudgetLedger(tmp_path, _settings(hard=2))
    ledger.reserve(phase="s0", raw_items=2)
    with pytest.raises(BudgetHardStop):
        ledger.reserve(phase="s0", raw_items=1)
    assert ledger.state()["new_model_calls_allowed"] is False


def test_cache_key_contains_position_boundary_and_source_hash():
    key = _space().entries[1].realized_key(42).to_dict()
    assert key["token_id"] == 42
    assert key["position"] == "random"
    assert key["random_boundary_id"] == "random"
    assert key["pretruncated_source_ids_hash"] == "source-hash"
    assert key["model_revision"] == "revision"


def test_cache_manifest_shape_and_hash(tmp_path):
    space = _space()
    cache = EmbeddingCache(tmp_path, space)
    cache.store(3, [0, 1], np.asarray([[1, 0], [0, 1]], dtype=np.float32), phase="s0")
    found, missing = cache.fetch(3, [0, 1])
    assert not missing
    assert np.allclose(found[0], [1, 0])
    manifest = json.loads(next((tmp_path / "embedding_cache" / "token_3").glob("*.json")).read_text())
    assert manifest["shape"] == [2, 2]
    assert len(manifest["vectors_sha256"]) == 64


def test_config_hard_disables_physical_gpus_zero_through_three():
    config = load_config(Path("configs/v6_3_mode3_light.yaml"))
    assert config["resources"]["allowed_physical_gpus"] == [4, 5, 6, 7]
    assert config["resources"]["forbidden_physical_gpus"] == [0, 1, 2, 3]
    assert config["data"]["ood_domains"] == 4


def test_engineering_profiles_reduce_resources_without_changing_formal():
    config = load_config(Path("configs/v6_3_mode3_light.yaml"))
    dry = config_for_profile(config, "dry_run")
    assert dry["resources"]["estimated_peak_cache_bytes"] < config["resources"]["estimated_peak_cache_bytes"]
    assert dry["data"]["ood_trigger_per_domain"] < config["data"]["ood_trigger_per_domain"]
    assert config["funnel"]["s0_keep"] == 12000


def test_result_inventory_has_recoverable_triple_identity(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    inventory = result_inventory(tmp_path)
    atomic_json(tmp_path / "result_inventory.json", inventory)
    again = result_inventory(tmp_path)
    assert inventory == again
    assert inventory["files"][0]["relative_path"] == "a.txt"
    assert len(inventory["root_sha256"]) == 64
