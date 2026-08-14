#!/usr/bin/env python3
"""Create a read-only human/audit summary from completed V6.2 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))
def read_jsonl(path: Path) -> list[dict[str, Any]]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--results-root", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); root = args.results_root.resolve(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    final = read(root / "FINAL_STATUS.json"); confirmation = read(root / "confirmation" / "COMPLETE.json")
    p1 = read(root / "confirmation" / "core" / "P1" / "result.json"); p2 = read(root / "confirmation" / "core" / "P2" / "result.json")
    p3 = read_jsonl(root / "confirmation" / "core" / "P3" / "results.jsonl")
    semantic = read(root / "semantic_confirmation" / "COMPLETE.json"); freezes = read(root / "freezes" / "INDEX.json")
    stages = {name: read(root / "funnel" / name / "COMPLETE.json") for name in ("s0", "s1", "s2", "full", "stability")}
    summary = {
        "schema_version": "mode3-v6-2-readable-summary-v1", "final": final,
        "funnel": {name: {"candidate_tokens": value["candidate_tokens"], "valid_models": value["valid_models"], "selected_tokens": value["selected_tokens"]} for name, value in stages.items()},
        "freeze_count": freezes["freeze_count"], "confirmation": confirmation,
        "P1_simultaneous": p1["simultaneous_all_positions"], "P2_simultaneous": p2["simultaneous_all_positions"],
        "P3": p3, "semantic_confirmation": semantic,
        "budget": {"planned": read(root / "budget" / "planned.json"), "observed": read(root / "budget" / "observed.json")},
        "interpretation": "A negative endpoint is specific to this frozen model, corpus, protocol and budget; it is not a universal nonexistence proof.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = []
    for value in p3:
        rows.append({"token_id": value["token_id"], "token_text": value["token_text"], "cap_count": value["cap_count"], "core": value["levels"]["B_ST_FCA_Core"], "moat": value["levels"]["C_ST_FCA_Moat"], "basin": value["levels"]["D_ST_FCA_Basin"], "collapse": value["levels"]["E_ST_Central_Collapse"], "coverage_lcb": value["coverage"]["balanced_lower"], "worst_position_lcb": value["coverage"]["worst_position_lower"], "worst_source_lcb": value["coverage"]["worst_source_lower"], "worst_benign_ucb": value["occupancy"]["worst_source_ucb"]})
    with (output / "p3_certificates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# V6.2 result summary", "", f"- P1 simultaneous certificate: `{p1['simultaneous_all_positions']}`", f"- P2 simultaneous certificate: `{p2['simultaneous_all_positions']}`", f"- P3 core certificates: `{confirmation['P3_core_certified']}`", f"- Frozen semantic anomalies supported: `{semantic['supported']}/{semantic['candidates']}`", f"- Final negative endpoint: `{final['negative_endpoint']}`", "", "P3 candidate details are in `p3_certificates.csv`; all decisions reference immutable freeze hashes in the source result tree."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); print(json.dumps({"output": str(output), "p3": len(rows)}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
