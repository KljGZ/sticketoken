#!/usr/bin/env python3
"""Build the auditable, human-readable V6 Compact result bundle.

This script is deliberately post-run only: it reads the immutable formal result
tree and writes summaries and figures to a separate publication directory.  It
never fits a cap, encodes text, or changes a gate decision.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(f"expected object at {path}:{line_number}")
            records.append(payload)
    return records


def fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def gate_count(gates: dict[str, Any]) -> int:
    return sum(bool(value) for value in gates.values())


def single_row(record: dict[str, Any]) -> dict[str, Any]:
    result = record["single_cap"]
    counts = result["counts"]
    bounds = result["bounds"]
    cap = result["cap"]
    migration = result["migration"]
    gates = result["gates"]
    p1 = record["layers"]["P1_position_specific"]
    return {
        "token_id": record["token_id"],
        "token_text": record["token_text"],
        "cap_count": cap["cap_count"],
        "radius_degrees": ";".join(f"{value:.12g}" for value in cap["radius_degrees"]),
        "triggered_inside": counts["triggered_inside"],
        "triggered_total": counts["triggered"],
        "triggered_coverage": fraction(counts["triggered_inside"], counts["triggered"]),
        "coverage_lcb": bounds["coverage_lcb"],
        "independent_benign_inside": counts["independent_benign_inside"],
        "independent_benign_total": counts["independent_benign"],
        "independent_benign_occupancy": fraction(
            counts["independent_benign_inside"], counts["independent_benign"]
        ),
        "benign_occupancy_ucb": bounds["benign_occupancy_ucb"],
        "paired_clean_inside": counts["paired_clean_inside"],
        "paired_clean_total": counts["paired_clean"],
        "outside_to_inside": migration["outside_to_inside"],
        "outside_to_inside_lcb": bounds["outside_to_inside_lcb"],
        "conditional_outside_origin_lcb": bounds["conditional_outside_origin_lcb"],
        "gate_coverage": bool(gates["coverage"]),
        "gate_low_core_occupancy": bool(gates["low_core_occupancy"]),
        "gate_outside_to_inside": bool(gates["outside_to_inside"]),
        "gate_conditional_outside_origin": bool(gates["conditional_outside_origin"]),
        "gates_passed": gate_count(gates),
        "single_cap_certified": bool(result["certified"]),
        "p1_prefix_certified": bool(p1["prefix"]["certified"]),
        "p1_suffix_certified": bool(p1["suffix"]["certified"]),
        "p1_random_certified": bool(p1["random"]["certified"]),
        "p2_certified": bool(record["layers"]["P2_conditional"]["certified"]),
        "p3_certified": bool(record["layers"]["P3_shared"]["certified"]),
    }


def rescue_row(record: dict[str, Any]) -> dict[str, Any] | None:
    rescue = record.get("two_cap_rescue", {})
    if not rescue.get("attempted") or not isinstance(rescue.get("result"), dict):
        return None
    result = rescue["result"]
    counts = result["counts"]
    bounds = result["bounds"]
    cap = result["cap"]
    gates = result["gates"]
    return {
        "token_id": record["token_id"],
        "token_text": record["token_text"],
        "cap_count": cap["cap_count"],
        "radius_degrees": ";".join(f"{value:.12g}" for value in cap["radius_degrees"]),
        "triggered_inside": counts["triggered_inside"],
        "triggered_total": counts["triggered"],
        "triggered_coverage": fraction(counts["triggered_inside"], counts["triggered"]),
        "coverage_lcb": bounds["coverage_lcb"],
        "independent_benign_inside": counts["independent_benign_inside"],
        "independent_benign_total": counts["independent_benign"],
        "independent_benign_occupancy": fraction(
            counts["independent_benign_inside"], counts["independent_benign"]
        ),
        "benign_occupancy_ucb": bounds["benign_occupancy_ucb"],
        "outside_to_inside_lcb": bounds["outside_to_inside_lcb"],
        "conditional_outside_origin_lcb": bounds["conditional_outside_origin_lcb"],
        "gates_passed": gate_count(gates),
        "certified": bool(result["certified"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def radial_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    radial = record["single_cap"]["radial"]
    rows: list[dict[str, Any]] = []
    for population in ("triggered", "paired_clean", "independent_benign"):
        for shell in radial[population]:
            rows.append({"population": population, **shell})
    return rows


def plot_bundle(
    output: Path,
    funnel: list[tuple[str, int]],
    single_rows: list[dict[str, Any]],
    rescue_rows: list[dict[str, Any]],
    exemplar: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 180, "font.size": 10})

    labels = [label for label, _ in funnel]
    values = [value for _, value in funnel]
    fig, axis = plt.subplots(figsize=(8.2, 4.5))
    bars = axis.bar(labels, values, color=["#355C7D", "#4F86A6", "#6CA6C1", "#84B8CC", "#F6AE2D", "#C84630"])
    axis.set_ylabel("Candidates")
    axis.set_title("V6 Compact candidate funnel")
    axis.set_yscale("symlog", linthresh=1)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, max(value, 1) * 1.08, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output / "funnel_counts.png")
    plt.close(fig)

    fig, (overview, zoom) = plt.subplots(1, 2, figsize=(12.0, 5.1), gridspec_kw={"width_ratios": [1.15, 1]})
    for axis in (overview, zoom):
        for row in single_rows:
            x = 100 * row["benign_occupancy_ucb"]
            y = 100 * row["coverage_lcb"]
            axis.scatter(x, y, s=34, color="#355C7D", alpha=0.82)
        for row in rescue_rows:
            x = 100 * row["benign_occupancy_ucb"]
            y = 100 * row["coverage_lcb"]
            axis.scatter(x, y, s=54, marker="^", color="#D95F02", alpha=0.9)
        axis.axvline(1.0, color="#C84630", linestyle="--", label="occupancy UCB < 1%")
        axis.axhline(90.0, color="#2A9D8F", linestyle="--", label="coverage LCB at least 90%")
        axis.grid(alpha=0.22)
    overview.set_xscale("log")
    overview.set_xlim(0.55, 105)
    overview.set_xlabel("Benign occupancy UCB (%) — log scale")
    overview.set_ylabel("Triggered coverage LCB (%)")
    overview.set_title("All finalists")
    overview.legend(loc="upper right", fontsize=8)

    zoom.set_xlim(0.55, 2.0)
    zoom.set_ylim(89.55, 90.30)
    zoom.set_xlabel("Benign occupancy UCB (%)")
    zoom.set_title("Registered-gate neighborhood")
    annotation_offsets = {
        "racist": (-54, 14),
        "vegan": (-5, -18),
        "boycott": (5, 28),
    }
    for row in single_rows:
        if row["token_text"] in annotation_offsets:
            zoom.annotate(
                f"{row['token_text']} (1 cap)",
                (100 * row["benign_occupancy_ucb"], 100 * row["coverage_lcb"]),
                xytext=annotation_offsets[row["token_text"]],
                textcoords="offset points",
                fontsize=8,
            )
    for row in rescue_rows:
        if row["token_text"] == "racist":
            zoom.annotate(
                "racist (2 caps)",
                (100 * row["benign_occupancy_ucb"], 100 * row["coverage_lcb"]),
                xytext=(-78, 6),
                textcoords="offset points",
                fontsize=8,
            )
    fig.suptitle("Registered validation gates: one-cap candidates and two-cap rescue")
    fig.tight_layout()
    fig.savefig(output / "validation_gate_frontier.png")
    plt.close(fig)

    radial = exemplar["single_cap"]["radial"]
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    styles = {
        "triggered": ("Triggered", "#D95F02"),
        "paired_clean": ("Paired clean", "#7570B3"),
        "independent_benign": ("Independent benign", "#1B9E77"),
    }
    for population, (label, color) in styles.items():
        shells = radial[population]
        axis.plot(
            [entry["upper_multiplier"] for entry in shells],
            [100 * entry["cumulative_fraction"] for entry in shells],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=label,
            color=color,
        )
    axis.axvline(1.0, color="#333333", linestyle="--", linewidth=1, label="Frozen radius rho")
    axis.set_xlim(0.1, 2.0)
    axis.set_ylim(-1, 101)
    axis.set_xticks([index / 10 for index in range(1, 21)])
    axis.set_xlabel("Angular distance / frozen radius")
    axis.set_ylabel("Cumulative population (%)")
    axis.set_title(f"Radial profile of near-miss token {exemplar['token_text']!r}")
    axis.grid(alpha=0.22)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "exemplar_radial_profile.png")
    plt.close(fig)

    migration = exemplar["single_cap"]["migration"]
    categories = ["outside_to_inside", "inside_to_inside", "outside_to_outside", "inside_to_outside"]
    labels = ["Outside to inside", "Inside to inside", "Outside to outside", "Inside to outside"]
    colors = ["#2A9D8F", "#457B9D", "#E9C46A", "#E76F51"]
    fig, axis = plt.subplots(figsize=(8.2, 2.6))
    left = 0.0
    for category, label, color in zip(categories, labels, colors):
        value = 100 * migration[category]
        axis.barh(["Paired samples"], [value], left=left, color=color, label=f"{label}: {value:.3f}%")
        if value >= 3:
            axis.text(left + value / 2, 0, f"{value:.2f}%", ha="center", va="center", fontsize=9)
        left += value
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of paired validation samples (%)")
    axis.set_title(f"Clean-to-triggered migration for {exemplar['token_text']!r}")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "exemplar_migration.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.results_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    final_status = read_json(root / "FINAL_STATUS.json")
    validation_complete = read_json(root / "validation" / "COMPLETE.json")
    sealed = read_json(root / "sealed" / "NOT_RUN.json")
    contract = read_json(root / "registration" / "run_contract.json")
    enumeration = read_json(root / "enumeration" / "COMPLETE.json")
    whitebox = read_json(root / "tracks" / "whitebox" / "COMPLETE.json")
    blackbox = read_json(root / "tracks" / "blackbox" / "COMPLETE.json")
    observed = read_json(root / "budget" / "observed.json")
    planned = read_json(root / "budget" / "planned.json")
    data_audit = read_json(root / "registration" / "data_capacity_audit.json")
    duplicate_audit = read_json(root / "registration" / "near_duplicate_audit.json")
    records = read_jsonl(root / "validation" / "all_results.jsonl")
    if len(records) != int(validation_complete["tested"]):
        raise RuntimeError("validation record count does not match COMPLETE.json")

    singles = [single_row(record) for record in records]
    singles.sort(key=lambda row: (-row["gates_passed"], row["benign_occupancy_ucb"], -row["coverage_lcb"], row["token_id"]))
    rescues = [row for record in records if (row := rescue_row(record)) is not None]
    rescues.sort(key=lambda row: (-row["gates_passed"], row["benign_occupancy_ucb"], -row["coverage_lcb"], row["token_id"]))
    if not singles:
        raise RuntimeError("no validation candidates")
    exemplar_row = singles[0]
    exemplar = next(record for record in records if record["token_id"] == exemplar_row["token_id"])

    funnel = [
        ("Legal vocabulary", int(enumeration["visible_count"])),
        ("S0", int(read_json(root / "funnel" / "s0" / "COMPLETE.json")["selected"])),
        ("S1", int(read_json(root / "funnel" / "s1" / "COMPLETE.json")["selected"])),
        ("S2", int(read_json(root / "funnel" / "s2" / "COMPLETE.json")["selected"])),
        ("S3/validation", int(read_json(root / "funnel" / "s3" / "COMPLETE.json")["selected"])),
        ("Certified", int(validation_complete["certified"])),
    ]
    budget_ratio = fraction(int(observed["submitted_text_equivalent"]), int(planned["planned_submitted_text_equivalent"]))
    summary = {
        "schema_version": "mode3-v6-compact-readable-summary-v1",
        "scope": "mode3_v6_compact_only",
        "run_code_commit": contract["run_code_commit"],
        "config_sha256": contract["config_sha256"],
        "endpoint": {
            "gate_open": final_status["gate_open"],
            "negative_endpoint": final_status["negative_endpoint"],
            "sealed_followups_encoded": contract["test_ood_encoded"],
            "reason": sealed["reason"],
        },
        "funnel": dict(funnel),
        "tracks": {"whitebox": whitebox, "blackbox": blackbox},
        "validation": {
            "tested": validation_complete["tested"],
            "certified": validation_complete["certified"],
            "frozen": validation_complete["frozen"],
            "one_cap_candidates": len(singles),
            "two_cap_rescues": len(rescues),
            "best_registered_near_miss": exemplar_row,
            "best_two_cap_rescue": rescues[0] if rescues else None,
        },
        "data": {**data_audit, "near_duplicate_audit": duplicate_audit},
        "budget": {
            "observed": observed,
            "planned": planned,
            "observed_fraction_of_planned_estimate": budget_ratio,
        },
        "interpretation_limit": (
            "No candidate was certified under the registered confidence-bound gates. "
            "This is a protocol-, model-, data-, and budget-specific negative endpoint, "
            "not a proof that no such token can exist in every setting."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output / "validation_one_cap_candidates.csv", singles)
    if rescues:
        write_csv(output / "validation_two_cap_rescues.csv", rescues)
    write_csv(output / "exemplar_radial_shells.csv", radial_rows(exemplar))

    compact_evidence = {
        "single_cap": {
            "token_id": exemplar["token_id"],
            "token_text": exemplar["token_text"],
            "cap": {key: value for key, value in exemplar["single_cap"]["cap"].items() if key != "centers"},
            "bounds": exemplar["single_cap"]["bounds"],
            "counts": exemplar["single_cap"]["counts"],
            "gates": exemplar["single_cap"]["gates"],
            "migration": exemplar["single_cap"]["migration"],
            "depth": exemplar["single_cap"]["depth"],
            "certified": exemplar["single_cap"]["certified"],
        },
        "two_cap_rescue": None,
    }
    if exemplar["two_cap_rescue"].get("attempted"):
        rescue = exemplar["two_cap_rescue"]["result"]
        compact_evidence["two_cap_rescue"] = {
            "cap": {key: value for key, value in rescue["cap"].items() if key != "centers"},
            "bounds": rescue["bounds"],
            "counts": rescue["counts"],
            "gates": rescue["gates"],
            "migration": rescue["migration"],
            "certified": rescue["certified"],
        }
    (output / "exemplar_evidence.json").write_text(
        json.dumps(compact_evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_bundle(output, funnel, singles, rescues, exemplar)
    print(json.dumps({"output": str(output), "tested": len(singles), "certified": validation_complete["certified"], "exemplar": exemplar["token_text"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
