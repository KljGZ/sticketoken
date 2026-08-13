from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "sticky_lab" / "mode3_v6"


def test_v6_config_encodes_nonnegotiable_scope_and_primary_question() -> None:
    config = yaml.safe_load((ROOT / "configs" / "v6_mode3.yaml").read_text(encoding="utf-8"))
    assert config["protocol_version"] == 6
    assert config["scope"]["only_mode"] == 3
    assert config["tokenizer"]["exact_actual_length"] == 1
    assert config["geometry"]["metric"] == "angular_distance_radians"
    assert config["certification"]["triggered_coverage_lcb"] == 0.90
    assert config["certification"]["independent_benign_occupancy_ucb"] == 0.01
    assert config["enumeration"]["full_search_candidates_minimum"] >= 2000
    assert config["positions"]["shared_equal_weights"] == [1/3, 1/3, 1/3]
    assert config["data"]["allow_resampling"] is False


def test_v6_has_no_old_mode_imports() -> None:
    for path in V6.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom): imports.append(node.module or "")
        assert not any(any(value in name for value in ("mode1", "mode2", "mode3_v3", "mode3_v4", "mode3_v5")) for name in imports)


def test_blackbox_modules_cannot_access_privileged_model_state() -> None:
    forbidden = {"backward", "parameters", "named_parameters", "get_input_embeddings", "grad", "hidden_states"}
    for path in (V6 / "oracle_blackbox.py", V6 / "blackbox_search.py", V6 / "track_blackbox.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert forbidden.isdisjoint(attrs), (path, forbidden.intersection(attrs))


def test_formal_geometry_never_uses_pca_or_umap() -> None:
    source = (V6 / "geometry.py").read_text(encoding="utf-8").lower()
    assert "pca" not in source and "umap" not in source


def test_sealed_worker_has_no_fit_calls() -> None:
    source = (V6 / "sealed_worker.py").read_text(encoding="utf-8")
    assert "fit_robust" not in source
    assert '"refit_performed": False' in source


def test_indexed_sealed_results_cannot_overwrite_each_other() -> None:
    source = (V6 / "run.py").read_text(encoding="utf-8")
    assert 'f"{phase}_{int(payload[\'index\']):02d}"' in source
