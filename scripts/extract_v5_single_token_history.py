from __future__ import annotations

import argparse
import json
from pathlib import Path


def visit(value: object, *, length_one_context: bool, output: set[int]) -> None:
    if isinstance(value, dict):
        actual = value.get("actual_length", value.get("tokenizer_length", value.get("length")))
        local_length_one = length_one_context or actual == 1
        if local_length_one and isinstance(value.get("token_id"), int): output.add(int(value["token_id"]))
        ids = value.get("token_ids")
        if local_length_one and isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int): output.add(int(ids[0]))
        for child in value.values(): visit(child, length_one_context=local_length_one, output=output)
    elif isinstance(value, list):
        for child in value: visit(child, length_one_context=length_one_context, output=output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-results", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.v5_results); token_ids: set[int] = set(); files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}: continue
        files += 1; context = any(part in {"length_01", "length_1", "L01", "L1"} for part in path.parts)
        try:
            if path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip(): visit(json.loads(line), length_one_context=context, output=token_ids)
            else: visit(json.loads(path.read_text(encoding="utf-8")), length_one_context=context, output=token_ids)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    if not token_ids: raise RuntimeError("no auditable V5 actual-length-one token history found")
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"token_ids": sorted(token_ids), "source": str(root.resolve()), "scanned_files": files, "search_seed_use": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
