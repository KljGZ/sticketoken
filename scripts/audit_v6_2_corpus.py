#!/usr/bin/env python3
"""Read-only full-capacity allocation audit for the registered V6.2 corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sticky_lab.mode3_v6.data import load_registered_records
from sticky_lab.mode3_v6.deduplication import audit_role_leakage
from sticky_lab.mode3_v6_2.common import load_config, write_json
from sticky_lab.mode3_v6_2.data import register_v62_roles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_2_mode3.yaml", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    records = load_registered_records(str(data["input_glob"]), list(data["required_columns"]))
    roles, allocation = register_v62_roles(
        records, config, seed=int(config["positions"]["random_seed"])
    )
    leaks = audit_role_leakage(roles, float(data["maximum_near_duplicate_jaccard"]))
    payload = {
        "schema_version": "mode3-v6-2-corpus-allocation-audit-v1",
        "input_records": len(records),
        "role_counts": {name: len(rows) for name, rows in sorted(roles.items())},
        "iid_sources": allocation["iid_sources"],
        "accepted_documents": allocation["accepted_documents"],
        "verified_near_duplicate_leaks": len(leaks),
        "allocation": allocation,
        "passed": not leaks,
    }
    write_json(args.output, payload)
    if leaks:
        raise SystemExit(f"V6.2 full allocation audit found {len(leaks)} leaks")
    print(f"V6.2 full allocation audit passed: {allocation['accepted_documents']} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
