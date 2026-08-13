from __future__ import annotations

import csv
from pathlib import Path

from sticky_lab.mode3_v6.data import audit_csv_corpus, register_document_disjoint_roles
from sticky_lab.mode3_v6.deduplication import audit_role_leakage


def test_capacity_audit_fails_closed_without_provenance(tmp_path: Path) -> None:
    path = tmp_path / "legacy.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sentence1", "sentence2"])
        writer.writeheader(); writer.writerow({"sentence1": "a", "sentence2": "b"})
    audit = audit_csv_corpus(str(path), ["text", "document_id", "source_id", "domain"], 10, 4)
    assert not audit.formal_ready
    assert {gap.code for gap in audit.gaps} >= {"missing_columns", "insufficient_unique_texts", "missing_document_identity"}


def test_document_roles_never_split_documents() -> None:
    records = [
        {"text": f"text {index}", "document_id": f"doc{index // 2}", "source_id": "s", "domain": "d", "text_id": str(index)}
        for index in range(12)
    ]
    roles = register_document_disjoint_roles(records, {"a": 3, "b": 3}, seed=4)
    a = {row["document_id"] for row in roles["a"]}; b = {row["document_id"] for row in roles["b"]}
    assert not a.intersection(b)


def test_near_duplicate_audit_crosses_roles() -> None:
    roles = {
        "fit": [{"text": "the quick brown fox jumps over the lazy dog", "text_id": "1"}],
        "test": [{"text": "the quick brown fox jumps over the lazy dog", "text_id": "2"}],
    }
    leaks = audit_role_leakage(roles)
    assert len(leaks) == 1 and leaks[0].normalized_exact
