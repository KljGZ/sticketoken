"""Create a compact, auditable comparison of the registered three-mode runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_RUNS = {
    "single_sticky": Path("results/sticky_lab/sentence_t5_base/single_sticky_v1"),
    "multi_booster": Path("results/sticky_lab/sentence_t5_base/multi_booster_v1"),
    "repulsive_attractor": Path("results/sticky_lab/sentence_t5_base/repulsive_attractor_v1"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("results/sticky_lab/sentence_t5_base/comparison_v1"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for expected_mode, relative in DEFAULT_RUNS.items():
        summary_path = root / relative / "run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("mode") != expected_mode:
            raise ValueError(f"Mode mismatch in {summary_path}")
        records.append(
            {
                "mode": expected_mode,
                "best_trigger_json": json.dumps(summary.get("best_trigger", ""), ensure_ascii=False),
                "certified_count": int(summary.get("validated_count", summary.get("certified_count", 0))),
                "candidate_count": int(summary.get("candidate_count", summary.get("test_candidate_count", 0))),
                "best_certified": bool(summary.get("best_certified", summary.get("validated_count", 0) > 0)),
                "best_feasible": bool(summary.get("best_feasible", summary.get("validated_count", 0) > 0)),
                "best_constraint_violation": float(summary.get("best_constraint_violation", 0.0)),
                "runtime_seconds": float(summary["runtime_seconds"]),
                "model_revision": summary.get("model_revision"),
                "seed": int(summary["seed"]),
                "precision": summary["precision"],
                "insertion_modes": ",".join(summary["insertion_modes"]),
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output / "three_mode_summary.csv", index=False)
    table_columns = ["mode", "best_trigger_json", "certified_count", "best_certified", "best_constraint_violation", "runtime_seconds"]
    table_rows = [
        "| " + " | ".join(table_columns) + " |",
        "| " + " | ".join(["---"] * len(table_columns)) + " |",
    ]
    for _, row in frame[table_columns].iterrows():
        table_rows.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in table_columns) + " |")
    markdown = [
        "# Sentence-T5-base 三模式第一版实验汇总",
        "",
        "> `certified` 只表示各模式注册约束在独立 test 分区上通过；三种模式的认证定义不同，数量不能直接当作同一指标横比。",
        "",
        *table_rows,
        "",
        "详细条件见 `docs/three_mode_experiments.md`，逐候选证据见各模式目录中的 CSV 和 `run_summary.json`。",
    ]
    (output / "comparison_summary.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
