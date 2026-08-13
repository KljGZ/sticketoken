"""V6 data registry with fail-closed document isolation and capacity checks."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import glob
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Mapping, Sequence


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DataGap:
    code: str
    required: object
    observed: object
    detail: str


@dataclass(frozen=True)
class CapacityAudit:
    files: int
    rows: int
    exact_unique_texts: int
    normalized_unique_texts: int
    columns: tuple[str, ...]
    source_ids: int
    document_ids: int
    domains: int
    gaps: tuple[DataGap, ...]

    @property
    def formal_ready(self) -> bool:
        return not self.gaps

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["formal_ready"] = self.formal_ready
        return result


def audit_csv_corpus(pattern: str, required_columns: Sequence[str], minimum_unique: int, minimum_ood_sources: int) -> CapacityAudit:
    paths = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
    all_columns: set[str] = set()
    exact: set[str] = set()
    normalized: set[str] = set()
    sources: set[str] = set()
    documents: set[str] = set()
    domains: set[str] = set()
    rows = 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            all_columns.update(reader.fieldnames or [])
            for row in reader:
                rows += 1
                if "text" in row:
                    values = [row.get("text", "")]
                else:
                    values = [row.get("sentence1", ""), row.get("sentence2", "")]
                for value in values:
                    if value:
                        exact.add(value)
                        normalized.add(normalized_text(value))
                if row.get("source_id"):
                    sources.add(str(row["source_id"]))
                if row.get("document_id"):
                    documents.add(str(row["document_id"]))
                if row.get("domain"):
                    domains.add(str(row["domain"]))
    gaps: list[DataGap] = []
    missing = sorted(set(required_columns) - all_columns)
    if missing:
        gaps.append(DataGap("missing_columns", list(required_columns), sorted(all_columns), f"missing: {missing}"))
    if len(normalized) < int(minimum_unique):
        gaps.append(DataGap("insufficient_unique_texts", minimum_unique, len(normalized), "resampling is forbidden"))
    if len(sources) < int(minimum_ood_sources):
        gaps.append(DataGap("insufficient_sources", minimum_ood_sources, len(sources), "multi-source OOD cannot be registered"))
    if not documents:
        gaps.append(DataGap("missing_document_identity", ">0 real document ids", 0, "sentence-as-document fallback is forbidden"))
    return CapacityAudit(len(paths), rows, len(exact), len(normalized), tuple(sorted(all_columns)), len(sources), len(documents), len(domains), tuple(gaps))


def write_capacity_audit(audit: CapacityAudit, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_formal_capacity(audit: CapacityAudit) -> None:
    if not audit.formal_ready:
        codes = ", ".join(gap.code for gap in audit.gaps)
        raise RuntimeError(f"V6 formal data contract failed: {codes}; see capacity audit")


def required_unique_capacity(config: Mapping[str, object]) -> int:
    data = config["data"]
    assert isinstance(data, Mapping)
    roles = data["roles"]
    assert isinstance(roles, Mapping)
    base = sum(int(value) for value in roles.values())
    iid_extra = (int(data["iid_replications"]) - 1) * (int(roles["iid_test"]) + int(roles["iid_test_benign"]))
    ood = int(data["ood_domains"]) * (int(data["ood_trigger_per_domain"]) + int(data["ood_benign_per_domain"]))
    return base + iid_extra + ood


def load_registered_records(pattern: str, required_columns: Sequence[str]) -> list[dict[str, str]]:
    """Load a canonical V6 corpus only; legacy sentence-pair fallback is forbidden."""
    records: list[dict[str, str]] = []
    for path_value in sorted(glob.glob(pattern, recursive=True)):
        path = Path(path_value)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(required_columns) - set(reader.fieldnames or [])
            if missing:
                raise RuntimeError(f"{path}: missing V6 columns {sorted(missing)}")
            for row_index, row in enumerate(reader):
                value = {name: str(row.get(name, "")) for name in required_columns}
                if any(not value[name] for name in required_columns):
                    raise RuntimeError(f"{path}:{row_index + 2}: empty required V6 field")
                value["text_id"] = text_sha256(value["text"])
                value["input_file"] = path.as_posix()
                value["input_row"] = str(row_index + 2)
                records.append(value)
    return records


def register_document_disjoint_roles(
    records: Sequence[Mapping[str, str]],
    role_sizes: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    """Assign complete documents to roles; never truncate with resampling."""
    documents: dict[str, list[dict[str, str]]] = {}
    seen_text: set[str] = set()
    for source in records:
        row = dict(source)
        digest = text_sha256(row["text"])
        if digest in seen_text:
            continue
        seen_text.add(digest)
        documents.setdefault(row["document_id"], []).append(row)
    ordered = sorted(
        documents.items(),
        key=lambda item: hashlib.sha256(f"{seed}\0{item[0]}".encode()).hexdigest(),
    )
    result: dict[str, list[dict[str, str]]] = {role: [] for role in role_sizes}
    document_roles: dict[str, str] = {}
    cursor = 0
    for role, required in role_sizes.items():
        while len(result[role]) < int(required) and cursor < len(ordered):
            document_id, rows = ordered[cursor]
            cursor += 1
            document_roles[document_id] = role
            result[role].extend(rows)
        if len(result[role]) < int(required):
            raise RuntimeError(f"insufficient document-disjoint records for {role}: {len(result[role])}/{required}")
        # Whole-document rule means role sizes may exceed the target, never slice.
    if len(document_roles) != len(set(document_roles)):
        raise AssertionError("document leakage")
    return result


def build_all_role_sizes(config: Mapping[str, object]) -> dict[str, int]:
    data = config["data"]
    assert isinstance(data, Mapping)
    roles = {str(key): int(value) for key, value in dict(data["roles"]).items()}
    for replication in range(1, int(data["iid_replications"])):
        roles[f"iid_test_replication_{replication}"] = int(roles["iid_test"])
        roles[f"iid_test_benign_replication_{replication}"] = int(roles["iid_test_benign"])
    return roles


def register_v6_roles(records: Sequence[Mapping[str, str]], config: Mapping[str, object], *, seed: int) -> dict[str, list[dict[str, str]]]:
    """Create IID roles plus source-isolated multi-domain OOD roles."""
    data = config["data"]
    assert isinstance(data, Mapping)
    domain_groups: dict[str, list[Mapping[str, str]]] = {}
    for row in records:
        domain_groups.setdefault(str(row["domain"]), []).append(row)
    needed_ood = int(data["ood_domains"])
    per_domain = int(data["ood_trigger_per_domain"]) + int(data["ood_benign_per_domain"])
    eligible = [domain for domain, rows in domain_groups.items() if len({text_sha256(row["text"]) for row in rows}) >= per_domain]
    eligible.sort(key=lambda domain: hashlib.sha256(f"{seed}\0ood\0{domain}".encode()).hexdigest())
    if len(eligible) < needed_ood:
        raise RuntimeError(f"only {len(eligible)} OOD domains have {per_domain} unique texts; need {needed_ood}")
    ood_domains = eligible[:needed_ood]
    iid_records = [row for row in records if str(row["domain"]) not in set(ood_domains)]
    result = register_document_disjoint_roles(iid_records, build_all_role_sizes(config), seed=seed)
    for index, domain in enumerate(ood_domains):
        sizes = {
            f"ood_{index}_trigger": int(data["ood_trigger_per_domain"]),
            f"ood_{index}_benign": int(data["ood_benign_per_domain"]),
        }
        allocated = register_document_disjoint_roles(domain_groups[domain], sizes, seed=seed + index + 1)
        for role, rows in allocated.items():
            for row in rows:
                row["registered_ood_domain"] = domain
            result[role] = rows
    # Assert source isolation between IID and every OOD domain.
    iid_sources = {row["source_id"] for role, rows in result.items() if not role.startswith("ood_") for row in rows}
    for role, rows in result.items():
        if role.startswith("ood_") and iid_sources.intersection(row["source_id"] for row in rows):
            raise RuntimeError(f"source leakage into {role}")
    return result
