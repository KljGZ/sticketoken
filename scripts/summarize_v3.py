#!/usr/bin/env python3
"""Render the registered Mode 3 V3 outputs as a compact Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _reported_metric(record: dict[str, Any], key: str, *, universal: bool) -> Any:
    if not universal:
        return record.get(key)
    per_position = record.get("test_per_position_metrics", {})
    values = [metrics.get(key) for metrics in per_position.values() if metrics.get(key) is not None]
    if not values:
        return None
    # A universal certificate is governed by the worst registered position.
    return max(values) if key == "compact_radius_q95" else min(values)


def render(root: Path) -> str:
    audit = _load(root / "data_audit.json")
    rows: list[tuple[str, dict[str, Any]]] = []
    for position in ("prefix", "suffix", "random"):
        for protocol in ("separator", "blank"):
            path = root / "validation" / position / protocol / "test_result.json"
            if path.exists():
                rows.append((f"{protocol}/{position}", _load(path)))
    for protocol in ("separator", "blank"):
        path = root / "validation" / "universal" / protocol / "test_result.json"
        if path.exists():
            rows.append((f"{protocol}/universal", _load(path)))

    lines = [
        "# Sticky / Attractor V3 registered result summary",
        "",
        (
            f"Data: search={audit['split_sizes']['search']}, validation={audit['split_sizes']['validation']}, "
            f"test={audit['split_sizes']['test']}, OOD={audit['ood_size']}; "
            f"sentence/group overlaps={max(audit['overlap'].values())}."
        ),
        "",
        "| task | L | trigger | selection | val | test | OOD | full | M_sep | rho95 | M_sample | M_cluster | M_density |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, record in rows:
        universal = task.endswith("universal")
        if universal:
            test_ok = record.get("test_position_universal_certified")
            ood_ok = record.get("ood_position_universal_certified")
        else:
            test_ok = record.get("test_certified")
            ood_ok = record.get("ood_certified")
        values = [
            task,
            record.get("component_length"),
            repr(record.get("trigger", "")),
            record.get("selection_status"),
            record.get("validation_certified", record.get("validation_position_universal_certified")),
            test_ok,
            ood_ok,
            record.get("full_generalized"),
            _reported_metric(record, "separation_margin", universal=universal),
            _reported_metric(record, "compact_radius_q95", universal=universal),
            _reported_metric(record, "sample_blank_margin", universal=universal),
            _reported_metric(record, "cluster_blank_margin", universal=universal),
            _reported_metric(record, "density_blank_margin", universal=universal),
        ]
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = render(args.root)
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
