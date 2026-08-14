#!/usr/bin/env python3
"""Estimate or inspect the V6.2 global query budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sticky_lab.mode3_v6_2.budget import estimate_budget


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_2_mode3.yaml", type=Path)
    parser.add_argument("--legal-vocab", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = estimate_budget(config, args.legal_vocab)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not result["within_planned_limit"]:
        raise SystemExit("planned V6.2 budget exceeds preregistered 12.5T_V5 limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
