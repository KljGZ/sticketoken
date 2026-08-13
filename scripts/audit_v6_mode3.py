from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PREFIXES = (
    "configs/v6_mode3.yaml", "sticky_lab/mode3_v6/", "tests/test_mode3_v6_",
    "scripts/audit_v6_mode3.py", "scripts/run_v6_mode3_remote.sh",
    "scripts/publish_v6_results.py", "scripts/fetch_v6_full_results.py",
    "scripts/extract_v5_single_token_history.py",
    "docs/sticky_attractor_v6_", "results/sticky_lab/sentence_t5_base/mode3_v6/",
)


def names_since(base: str) -> list[str]:
    committed = subprocess.check_output(["git", "diff", "--name-only", f"{base}..HEAD"], cwd=ROOT, text=True).splitlines()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    uncommitted = [line[3:].replace("\\", "/") for line in status if len(line) > 3]
    return sorted(set(path.replace("\\", "/") for path in committed + uncommitted))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    base = str(config["scope"]["protected_baseline_commit"])
    changed = names_since(base)
    forbidden_changes = [path for path in changed if not path.startswith(ALLOWED_PREFIXES)]
    blackbox_forbidden = {"backward", "parameters", "named_parameters", "get_input_embeddings", "grad", "hidden_states", "input_embeddings"}
    violations: list[str] = []
    for path in sorted((ROOT / "sticky_lab" / "mode3_v6").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imports = []
        attrs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom): imports.append(node.module or "")
            elif isinstance(node, ast.Attribute): attrs.append(node.attr)
        if any(any(old in name for old in ("mode1", "mode2", "mode3_v3", "mode3_v4", "mode3_v5")) for name in imports):
            violations.append(f"old-mode import: {path}")
        if "blackbox" in path.name and blackbox_forbidden.intersection(attrs):
            violations.append(f"blackbox privileged attribute: {path}: {sorted(blackbox_forbidden.intersection(attrs))}")
    geometry_source = (ROOT / "sticky_lab" / "mode3_v6" / "geometry.py").read_text(encoding="utf-8").lower()
    for forbidden in ("pca", "umap", "1 - cos", "1-cos"):
        if forbidden in geometry_source:
            violations.append(f"formal geometry mentions visualization/surrogate: {forbidden}")
    result = {
        "protocol_version": config["protocol_version"], "base": base, "changed_paths": changed,
        "forbidden_changes": forbidden_changes, "source_violations": violations,
        "scope_pass": not forbidden_changes and not violations,
    }
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["scope_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
