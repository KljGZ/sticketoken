from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "sticky_lab" / "mode3_v5"


def test_registered_v5_scope_and_every_actual_length() -> None:
    config = yaml.safe_load((ROOT / "configs" / "v5_mode3.yaml").read_text(encoding="utf-8"))
    assert config["protocol_version"] == 5
    assert config["lengths"] == {
        "exhaustive_single_token": True,
        "minimum": 1,
        "maximum": 30,
        "step": 1,
        "stop_search_after_first_certified": False,
        "test_shortest_per_certificate_and_protocol": True,
    }
    assert config["search"]["tasks"] == ["prefix", "suffix", "random", "conditional", "shared"]
    assert config["validation"]["bootstrap_replicates"] == 500
    assert config["structure"]["maximum_cluster_count"] == 4
    assert config["insertion"]["protocol"] == "shared_literal_insert_once"


def test_query_only_source_has_no_parameter_gradient_or_old_mode_dependency() -> None:
    forbidden_attributes = {
        "backward",
        "parameters",
        "named_parameters",
        "get_input_embeddings",
        "hidden_states",
        "output_hidden_states",
        "grad",
    }
    for path in sorted(V5.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        attributes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Attribute):
                attributes.append(node.attr)
        assert not any("mode1" in value or "mode2" in value or "mode3_v3" in value or "mode3_v4" in value for value in imports)
        assert forbidden_attributes.isdisjoint(attributes), (path, forbidden_attributes.intersection(attributes))


def test_test_and_ood_are_sealed_until_validation_freeze() -> None:
    source = (V5 / "run.py").read_text(encoding="utf-8")
    prepare_region = source[source.index("def command_prepare") : source.index("def command_calibrate")]
    assert '"test_trigger"' in prepare_region
    assert "test/OOD embeddings were encoded before the validation gate" in prepare_region
    assert "oracle.encode(iid_roles[role]" in prepare_region
    allowed_region = prepare_region[prepare_region.index("allowed_roles") : prepare_region.index("if any(")]
    assert "test_trigger" not in allowed_region
    assert "ood_trigger" not in allowed_region
    sealed_region = source[source.index("def _encode_sealed_phase") : source.index("def command_retrieval")]
    assert "validation freeze must complete before test/OOD" in sealed_region
    assert "refit_performed" not in sealed_region
