#!/usr/bin/env python3
"""Scope/config audit for the isolated V6 Compact branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sticky_lab.mode3_v6_compact.common import load_config, sha256_file



def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3_compact.yaml", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    base = str(config["scope"]["immutable_heavy_commit"])
    changed = [value for value in git("diff", "--name-only", f"{base}..HEAD").splitlines() if value]
    allowed_prefixes = (
        "configs/v6_mode3_compact.yaml",
        "sticky_lab/mode3_v6_compact/",
        "tests/test_mode3_v6_compact",
        "scripts/run_v6_mode3_compact_remote.sh",
        "scripts/stop_v6_mode3_compact_remote.sh",
        "scripts/status_v6_mode3_compact.py",
        "scripts/budget_v6_mode3_compact.py",
        "scripts/audit_v6_mode3_compact.py",
        "scripts/dry_run_v6_mode3_compact.py",
        "docs/sticky_cap_v6_compact_",
    )
    forbidden = [path for path in changed if not path.startswith(allowed_prefixes)]
    payload = {
        "schema_version": "mode3-v6-compact-scope-audit-v1",
        "head": git("rev-parse", "HEAD"),
        "base": base,
        "branch": git("branch", "--show-current"),
        "config_sha256": sha256_file(args.config),
        "changed_paths": changed,
        "forbidden_changes": forbidden,
        "only_mode": config["scope"]["only_mode"],
        "passed": not forbidden and config["scope"]["only_mode"] == 3,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not payload["passed"]:
        raise SystemExit("V6 Compact scope audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
