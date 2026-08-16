"""Deterministic V6.3 data registration with near-duplicate-safe allocation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
        self.conflict_audit: list[dict[str, Any]] = []

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
            similarity = jaccard(current_shingles, self.shingle_sets[index])
            if similarity >= self.threshold:
                if len(self.conflict_audit) < 1000:
                    self.conflict_audit.append({
                        "left_sha256": text_sha256(text),
                        "right_sha256": text_sha256(self.texts[index]),
                        "verified_shingle_jaccard": similarity,
                    })
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


def _allocate_balanced_iid_roles(
    records: Sequence[Mapping[str, str]],
    role_sizes: Mapping[str, int],
    *,
    index: NearDuplicateIndex,
    allocated_documents: set[str],
    seed: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, Any]]]:
    """Allocate every IID role equally across the registered sources.

    Source balancing is part of the sampling contract, not a downstream
    reweighting trick. V6.3 corpora use one complete record per document; the
    explicit check below prevents a multi-row document from silently breaking
    the exact per-source quotas.
    """
    source_ids = sorted({str(row["source_id"]) for row in records})
    if not source_ids:
        raise RuntimeError("no IID sources available")
    ordered = {
        source_id: _ordered_documents(
            [row for row in records if str(row["source_id"]) == source_id],
            seed=seed,
            namespace=f"iid:{source_id}",
        )
        for source_id in source_ids
    }
    cursors = {source_id: 0 for source_id in source_ids}
    result: dict[str, list[dict[str, str]]] = {str(role): [] for role in role_sizes}
    stats: dict[str, dict[str, Any]] = {}
    for role_index, (role_value, required_value) in enumerate(role_sizes.items()):
        role = str(role_value)
        required = int(required_value)
        quotient, remainder = divmod(required, len(source_ids))
        # Rotate remainder ownership so no lexicographically early source is
        # systematically overrepresented across the role sequence.
        rotated = source_ids[role_index % len(source_ids):] + source_ids[:role_index % len(source_ids)]
        quotas = {source_id: quotient + int(source_id in set(rotated[:remainder])) for source_id in source_ids}
        counters: dict[str, Any] = {
            "documents_examined": 0,
            "documents_accepted": 0,
            "documents_rejected_exact": 0,
            "documents_rejected_near": 0,
            "source_quotas": quotas,
            "source_counts": {},
        }
        for source_id in source_ids:
            accepted: list[dict[str, str]] = []
            candidates = ordered[source_id]
            while len(accepted) < quotas[source_id] and cursors[source_id] < len(candidates):
                document_key, rows = candidates[cursors[source_id]]
                cursors[source_id] += 1
                counters["documents_examined"] += 1
                if len(rows) != 1:
                    raise RuntimeError(
                        "balanced IID allocation requires exactly one complete record per document: "
                        f"{document_key} has {len(rows)} rows"
                    )
                if document_key in allocated_documents:
                    continue
                conflict, reason = index.conflicts(rows[0]["text"])
                if conflict:
                    key = "documents_rejected_exact" if reason == "exact_or_normalized" else "documents_rejected_near"
                    counters[key] += 1
                    continue
                allocated_documents.add(document_key)
                index.add(rows[0]["text"])
                accepted.append(rows[0])
                counters["documents_accepted"] += 1
            if len(accepted) != quotas[source_id]:
                raise RuntimeError(
                    f"balanced capacity gap for {role}/{source_id}: "
                    f"{len(accepted)}/{quotas[source_id]}; "
                    f"examined={cursors[source_id]}/{len(candidates)}"
                )
            counters["source_counts"][source_id] = len(accepted)
            result[role].extend(accepted)
        if len(result[role]) != required:
            raise RuntimeError(f"internal balanced allocation mismatch for {role}")
        counters["target_rows"] = required
        counters["allocated_rows"] = len(result[role])
        stats[role] = counters
    return result, stats


def required_unique_capacity(config: Mapping[str, Any]) -> int:
    data = config["data"]
    search = data["search_chain_sizes"]["full"]
    iid = sum(int(value) for value in search.values())
    iid += int(data["discovery_benign"])
    iid += sum(int(value) for value in data["confirm_roles"].values())
    ood = len(data["ood_domains_allowlist"]) * (
        int(data["ood_trigger_per_domain"]) + int(data["ood_benign_per_domain"])
    )
    return iid + ood


def _annotate_chain(
    records: Sequence[Mapping[str, str]], *, chain: str, seed: int
) -> list[dict[str, str]]:
    """Give every full-chain record a stable source-local position rank."""
    output: list[dict[str, str]] = []
    sources = sorted({str(row["source_id"]) for row in records})
    for source in sources:
        rows = [dict(row) for row in records if str(row["source_id"]) == source]
        rows.sort(key=lambda row: hashlib.sha256(
            f"{seed}\0v6.3\0{chain}\0{source}\0{row['text_id']}".encode("utf-8")
        ).hexdigest())
        for rank, row in enumerate(rows):
            row["role_chain"] = str(chain)
            row["source_position_rank"] = str(rank)
            row["registered_role"] = str(chain)
            output.append(row)
    return output


def _nested_view(
    full_records: Sequence[Mapping[str, str]], required: int
) -> list[dict[str, str]]:
    sources = sorted({str(row["source_id"]) for row in full_records})
    quotient, remainder = divmod(int(required), len(sources))
    result: list[dict[str, str]] = []
    for index, source in enumerate(sources):
        quota = quotient + int(index < remainder)
        rows = sorted(
            (dict(row) for row in full_records if str(row["source_id"]) == source),
            key=lambda row: int(row["source_position_rank"]),
        )
        if len(rows) < quota:
            raise RuntimeError(f"nested source capacity gap for {source}: {len(rows)}/{quota}")
        result.extend(rows[:quota])
    if len(result) != int(required):
        raise RuntimeError("nested role view has the wrong size")
    return result


def register_v63_roles(
    records: Sequence[Mapping[str, str]], config: Mapping[str, Any], *, seed: int
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, dict[str, list[dict[str, str]]]],
    dict[str, Any],
]:
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
        raise RuntimeError("V6.3 OOD allowlist must exactly match ood_domains")
    ood_set = set(allowlist)
    iid_records = [row for row in records if str(row["domain"]) not in ood_set]
    observed_iid_sources = {str(row["source_id"]) for row in iid_records}
    if len(observed_iid_sources) < int(data["minimum_iid_sources"]):
        raise RuntimeError(
            f"V6.3 requires {data['minimum_iid_sources']} IID sources, "
            f"observed {len(observed_iid_sources)}"
        )
    full_sizes = {str(key): int(value) for key, value in data["search_chain_sizes"]["full"].items()}
    iid_sizes: dict[str, int] = {
        "fit": full_sizes["fit"],
        "radius": full_sizes["radius"],
        "score": full_sizes["score"],
        "discovery_benign": int(data["discovery_benign"]),
    }
    iid_sizes.update({str(key): int(value) for key, value in data["confirm_roles"].items()})
    roles, allocation_stats = _allocate_balanced_iid_roles(
        iid_records,
        iid_sizes,
        index=index,
        allocated_documents=allocated_documents,
        seed=seed,
    )
    for chain in ("fit", "radius", "score"):
        roles[chain] = _annotate_chain(roles[chain], chain=chain, seed=seed)
    for role in data["confirm_roles"]:
        roles[str(role)] = _annotate_chain(
            roles[str(role)], chain=str(role), seed=seed + 1000
        )
    views: dict[str, dict[str, list[dict[str, str]]]] = {}
    for stage in ("s0", "s1", "s2", "full"):
        views[stage] = {}
        for chain in ("fit", "radius", "score"):
            view = _nested_view(
                roles[chain], int(data["search_chain_sizes"][stage][chain])
            )
            for row in view:
                row["registered_view"] = f"{stage}_{chain}"
            views[stage][chain] = view
    ood_audit: dict[str, Any] = {}
    iid_sources = {row["source_id"] for rows in roles.values() for row in rows}
    if iid_sources != observed_iid_sources:
        raise RuntimeError(
            "balanced IID allocation did not retain every registered source: "
            f"registered={sorted(observed_iid_sources)}, allocated={sorted(iid_sources)}"
        )
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
            roles[role] = _annotate_chain(
                rows, chain=role, seed=seed + 2000 + ood_index
            )
        ood_audit[domain] = {"sources": sorted(current_sources), "allocation": stats}
    required_ood_sources = int(data.get("minimum_ood_sources", data["ood_domains"]))
    if len(seen_ood_sources) < required_ood_sources:
        raise RuntimeError(
            f"V6.3 requires {required_ood_sources} OOD sources, "
            f"observed {len(seen_ood_sources)}"
        )
    return roles, views, {
        "schema_version": "mode3-v6-3-allocation-v1",
        "policy": "three_disjoint_source_balanced_chains_with_nested_stage_views",
        "threshold": threshold,
        "accepted_rows": len(index.texts),
        "accepted_documents": len(allocated_documents),
        "role_counts": {role: len(rows) for role, rows in roles.items()},
        "nested_view_counts": {
            stage: {chain: len(rows) for chain, rows in chains.items()}
            for stage, chains in views.items()
        },
        "allocation": allocation_stats,
        "ood": ood_audit,
        "iid_sources": sorted(iid_sources),
        "verified_near_duplicate_conflict_sample": index.conflict_audit[
            : int(data.get("near_duplicate_manual_audit_pairs", 1000))
        ],
        "postcondition": "no later role contains an LSH-candidate verified near duplicate of an earlier role",
    }
