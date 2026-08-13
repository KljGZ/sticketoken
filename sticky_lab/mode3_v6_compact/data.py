"""Deterministic Compact data registration with near-duplicate-safe allocation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

from sticky_lab.mode3_v6.data import text_sha256
from sticky_lab.mode3_v6.deduplication import (
    jaccard,
    minhash_signature,
    shingles,
    simhash64,
)


@dataclass(frozen=True)
class RejectionStats:
    exact_or_normalized: int = 0
    near_duplicate: int = 0
    already_allocated_document: int = 0


class NearDuplicateIndex:
    """Verified LSH index for records already assigned to any formal role."""

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.normalized_hashes: set[str] = set()
        self.texts: list[str] = []
        self.shingle_sets: list[set[str]] = []
        self.simhashes: list[int] = []
        self.simhash_buckets: dict[tuple[int, int], list[int]] = {}
        self.minhash_buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}

    @staticmethod
    def _features(text: str) -> tuple[set[str], int, tuple[int, ...]]:
        return shingles(text), simhash64(text), minhash_signature(text)

    def conflicts(self, text: str) -> tuple[bool, str | None]:
        digest = text_sha256(text)
        if digest in self.normalized_hashes:
            return True, "exact_or_normalized"
        current_shingles, current_simhash, current_minhash = self._features(text)
        candidates: set[int] = set()
        for band in range(4):
            candidates.update(
                self.simhash_buckets.get(
                    (band, (current_simhash >> (16 * band)) & 0xFFFF), []
                )
            )
        for band in range(16):
            key = (band, current_minhash[band * 4 : (band + 1) * 4])
            candidates.update(self.minhash_buckets.get(key, []))
        for index in candidates:
            if jaccard(current_shingles, self.shingle_sets[index]) >= self.threshold:
                return True, "near_duplicate"
        return False, None

    def add(self, text: str) -> None:
        digest = text_sha256(text)
        if digest in self.normalized_hashes:
            raise ValueError("attempted to add duplicate text to registry")
        current_shingles, simhash, minhash = self._features(text)
        index = len(self.texts)
        self.normalized_hashes.add(digest)
        self.texts.append(str(text))
        self.shingle_sets.append(current_shingles)
        self.simhashes.append(simhash)
        for band in range(4):
            self.simhash_buckets.setdefault(
                (band, (simhash >> (16 * band)) & 0xFFFF), []
            ).append(index)
        for band in range(16):
            key = (band, minhash[band * 4 : (band + 1) * 4])
            self.minhash_buckets.setdefault(key, []).append(index)


def _document_key(row: Mapping[str, str]) -> str:
    return f"{row['source_id']}\0{row['document_id']}"


def _group_documents(records: Sequence[Mapping[str, str]]) -> dict[str, list[dict[str, str]]]:
    documents: dict[str, list[dict[str, str]]] = {}
    document_hashes: dict[str, set[str]] = {}
    for source in records:
        key = _document_key(source)
        digest = text_sha256(source["text"])
        hashes = document_hashes.setdefault(key, set())
        if digest in hashes:
            continue
        hashes.add(digest)
        documents.setdefault(key, []).append(dict(source))
    return documents


def _ordered_documents(
    records: Sequence[Mapping[str, str]], *, seed: int, namespace: str
) -> list[tuple[str, list[dict[str, str]]]]:
    documents = _group_documents(records)
    return sorted(
        documents.items(),
        key=lambda item: hashlib.sha256(
            f"{seed}\0{namespace}\0{item[0]}".encode("utf-8")
        ).hexdigest(),
    )


def _allocate_roles(
    ordered: Sequence[tuple[str, list[dict[str, str]]]],
    role_sizes: Mapping[str, int],
    *,
    index: NearDuplicateIndex,
    allocated_documents: set[str],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, int]]]:
    result: dict[str, list[dict[str, str]]] = {role: [] for role in role_sizes}
    stats: dict[str, dict[str, int]] = {}
    cursor = 0
    for role, required_value in role_sizes.items():
        required = int(required_value)
        counters = {
            "documents_examined": 0,
            "documents_accepted": 0,
            "documents_rejected_exact": 0,
            "documents_rejected_near": 0,
        }
        while len(result[role]) < required and cursor < len(ordered):
            document_key, rows = ordered[cursor]
            cursor += 1
            counters["documents_examined"] += 1
            if document_key in allocated_documents:
                continue
            reason: str | None = None
            for row in rows:
                conflict, reason = index.conflicts(row["text"])
                if conflict:
                    break
            if reason is not None:
                key = (
                    "documents_rejected_exact"
                    if reason == "exact_or_normalized"
                    else "documents_rejected_near"
                )
                counters[key] += 1
                continue
            allocated_documents.add(document_key)
            counters["documents_accepted"] += 1
            for row in rows:
                index.add(row["text"])
                result[role].append(row)
        if len(result[role]) < required:
            raise RuntimeError(
                f"near-duplicate-safe capacity gap for {role}: "
                f"{len(result[role])}/{required}; examined={cursor}/{len(ordered)}"
            )
        counters["target_rows"] = required
        counters["allocated_rows"] = len(result[role])
        stats[role] = counters
    return result, stats


def required_unique_capacity(config: Mapping[str, Any]) -> int:
    data = config["data"]
    return sum(int(value) for value in data["roles"].values()) + int(
        data["ood_domains"]
    ) * (int(data["ood_trigger_per_domain"]) + int(data["ood_benign_per_domain"]))


def register_compact_roles(
    records: Sequence[Mapping[str, str]], config: Mapping[str, Any], *, seed: int
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    """Allocate whole documents and reject near duplicates before role entry.

    The index is shared across IID and all OOD allocations.  Therefore a row
    can never be admitted to a later role if it is an exact, normalized, or
    verified Jaccard near duplicate of a previously admitted row.
    """

    data = config["data"]
    threshold = float(data["maximum_near_duplicate_jaccard"])
    index = NearDuplicateIndex(threshold)
    allocated_documents: set[str] = set()
    allowlist = list(map(str, data["ood_domains_allowlist"]))
    if len(allowlist) != int(data["ood_domains"]) or len(set(allowlist)) != len(allowlist):
        raise RuntimeError("Compact OOD allowlist must exactly match ood_domains")
    ood_set = set(allowlist)
    iid_records = [row for row in records if str(row["domain"]) not in ood_set]
    roles, allocation_stats = _allocate_roles(
        _ordered_documents(iid_records, seed=seed, namespace="iid"),
        {str(key): int(value) for key, value in data["roles"].items()},
        index=index,
        allocated_documents=allocated_documents,
    )
    ood_audit: dict[str, Any] = {}
    iid_sources = {row["source_id"] for rows in roles.values() for row in rows}
    seen_ood_sources: set[str] = set()
    for ood_index, domain in enumerate(allowlist):
        domain_rows = [row for row in records if str(row["domain"]) == domain]
        sizes = {
            f"ood_{ood_index}_trigger": int(data["ood_trigger_per_domain"]),
            f"ood_{ood_index}_benign": int(data["ood_benign_per_domain"]),
        }
        allocated, stats = _allocate_roles(
            _ordered_documents(
                domain_rows, seed=seed + ood_index + 1, namespace=f"ood:{domain}"
            ),
            sizes,
            index=index,
            allocated_documents=allocated_documents,
        )
        current_sources = {row["source_id"] for rows in allocated.values() for row in rows}
        if current_sources & iid_sources:
            raise RuntimeError(f"source leakage between IID and OOD domain {domain}")
        if current_sources & seen_ood_sources:
            raise RuntimeError(f"source leakage across OOD domains at {domain}")
        seen_ood_sources.update(current_sources)
        for role, rows in allocated.items():
            for row in rows:
                row["registered_ood_domain"] = domain
            roles[role] = rows
        ood_audit[domain] = {"sources": sorted(current_sources), "allocation": stats}
    return roles, {
        "schema_version": "mode3-v6-compact-allocation-v1",
        "policy": data["allocation_policy"],
        "threshold": threshold,
        "accepted_rows": len(index.texts),
        "accepted_documents": len(allocated_documents),
        "role_counts": {role: len(rows) for role, rows in roles.items()},
        "allocation": allocation_stats,
        "ood": ood_audit,
        "postcondition": "no later role contains an LSH-candidate verified near duplicate of an earlier role",
    }
