from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_config_is_v62_mode3_and_preserves_compact_history() -> None:
    config = yaml.safe_load((ROOT / "configs/v6_2_mode3.yaml").read_text(encoding="utf-8"))
    assert config["protocol_version"] == "6.2" and config["scope"]["only_mode"] == 3
    assert config["scope"]["immutable_compact_commit"].startswith("fa0ccb7")
    assert "results_publication/v6_compact" in config["scope"]["protected_paths"]
    assert config["positions"]["average_random_vectors"] is False
    assert config["geometry"]["design_coverage"] == .92
    assert config["data"]["minimum_iid_sources"] == config["data"]["minimum_ood_sources"] == 4


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig")); result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): result.append(node.module or "")
    return result


def test_confirmation_module_has_no_fit_imports() -> None:
    path = ROOT / "sticky_lab/mode3_v6_2/confirm.py"
    imports = _imports(path)
    assert not any(name.endswith("evaluate") or "selection" in name for name in imports)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not any(name.startswith("fit_") or name.startswith("calibrate_") for name in called)


def test_blackbox_track_is_physically_whitebox_isolated() -> None:
    assert not any("whitebox" in name for name in _imports(ROOT / "sticky_lab/mode3_v6_2/track_blackbox.py"))


def test_v62_namespace_never_imports_mode1_or_mode2() -> None:
    for path in (ROOT / "sticky_lab/mode3_v6_2").glob("*.py"):
        text = path.read_text(encoding="utf-8-sig")
        assert "sticky_lab.mode1" not in text and "sticky_lab.mode2" not in text


def test_remote_runner_targets_only_v62() -> None:
    text = (ROOT / "scripts/run_v6_2_mode3_remote.sh").read_text(encoding="utf-8-sig")
    assert "sticky_lab.mode3_v6_2" in text and "tests/test_mode3_v6_2" in text
    assert "sticky_lab.mode3_v6_compact" not in text
    assert "merge-enumeration" in text and "V6_2_ENUM_WORKERS" in text


def test_status_counts_every_funnel_stage_under_funnel_root() -> None:
    text = (ROOT / "scripts/status_v6_2_mode3.py").read_text(encoding="utf-8-sig")
    for stage in ("s0", "s1", "s2", "full", "stability"):
        assert f'"funnel/{stage}/shard_*/COMPLETE.json"' in text
