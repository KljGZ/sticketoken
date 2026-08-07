"""Create compact, auditable tables from completed Sticky / Attractor V2 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame[columns].iterrows():
        values = [str(row[column]).replace("|", "\\|").replace("\n", " ") for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sticky_lab/sentence_t5_base/comparison_v2"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    base = root / "results/sticky_lab/sentence_t5_base"
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    mode1 = _read_json(base / "single_sticky_v2/full_summary.json")
    mode2 = _read_json(base / "multi_booster_v2/finalize_summary.json")
    mode3 = _read_json(base / "repulsive_attractor_v2/finalize_summary.json")
    records = [
        {
            "mode": "single_sticky",
            "selected_length": 1,
            "selected_trigger_json": json.dumps(mode1["frozen_token"], ensure_ascii=False),
            "validation_core_certified": bool(mode1["test"].get("coverage_certified", False)),
            "test_core_certified": bool(mode1["test"].get("test_coverage_certified", False)),
            "minimum_effective_repeat_count": mode1.get("minimum_effective_repeat_count"),
            "runtime_seconds": float(mode1["runtime_seconds"]),
        },
        {
            "mode": "multi_booster",
            "selected_length": int(mode2["selected_length"]),
            "selected_trigger_json": json.dumps(mode2["selected_trigger"], ensure_ascii=False),
            "validation_core_certified": bool(mode2["validation_core_certified"]),
            "test_core_certified": bool(mode2["test_core_certified"]),
            "minimum_effective_repeat_count": None,
            "runtime_seconds": float(mode2["runtime_seconds"]),
        },
        {
            "mode": "repulsive_attractor",
            "selected_length": int(mode3["selected_length"]),
            "selected_trigger_json": json.dumps(mode3["selected_trigger"], ensure_ascii=False),
            "validation_core_certified": bool(mode3["validation_core_certified"]),
            "test_core_certified": bool(mode3["test_core_certified"]),
            "minimum_effective_repeat_count": None,
            "runtime_seconds": float(mode3["runtime_seconds"]),
        },
    ]
    summary = pd.DataFrame.from_records(records)
    summary.to_csv(output / "three_mode_summary.csv", index=False)
    for name in ("multi_booster_v2", "repulsive_attractor_v2"):
        frontier = pd.read_csv(base / name / "length_frontier.csv")
        frontier.to_csv(output / f"{name}_length_frontier.csv", index=False)
    columns = [
        "mode",
        "selected_length",
        "selected_trigger_json",
        "validation_core_certified",
        "test_core_certified",
        "runtime_seconds",
    ]
    markdown = [
        "# Sentence-T5-base Sticky / Attractor V2 实验汇总",
        "",
        "模式 1 的长度表示一个 token；其剂量由重复次数曲线单独报告。模式 2、3 的长度表示只插入一次的组合组件数。",
        "",
        *_markdown_table(summary, columns),
        "",
        "`test_core_certified` 只来自 validation 冻结后的一次 test 评估；test 未参与候选或长度选择。",
    ]
    (output / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

