from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_heavy_and_only_selects_mode3() -> None:
    config = yaml.safe_load((ROOT / "configs/v6_mode3_compact.yaml").read_text(encoding="utf-8"))
    assert config["scope"]["only_mode"] == 3
    assert config["scope"]["immutable_heavy_commit"] == "346feae1aaf2fe15f5f14b512a0fab06771ab7b6"
    assert config["scope"]["output_leaf"] == "mode3_v6_compact"
    assert config["blackbox"]["whitebox_seeded"] is False
    assert config["funnel"]["validation"]["maximum_caps"] == 2


def test_blackbox_track_does_not_import_whitebox_modules() -> None:
    path = ROOT / "sticky_lab/mode3_v6_compact/track_blackbox.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("whitebox" in name for name in imports)


def test_compact_source_never_imports_mode1_or_mode2() -> None:
    for path in (ROOT / "sticky_lab/mode3_v6_compact").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "sticky_lab.mode1" not in text
        assert "sticky_lab.mode2" not in text


def test_runner_only_invokes_compact_test_files() -> None:
    text = (ROOT / "scripts/run_v6_mode3_compact_remote.sh").read_text(encoding="utf-8")
    assert "pytest -q" in text
    assert "tests/test_mode3_v6_compact" in text
    assert "tests/test_mode3_v6_core.py" not in text
    assert "sticky_lab.mode3_v6_compact" in text
