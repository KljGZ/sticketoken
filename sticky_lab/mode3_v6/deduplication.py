"""Exact, normalized, MinHash, and SimHash leakage auditing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Mapping, Sequence

import numpy as np

from .data import normalized_text


def shingles(text: str, width: int = 5) -> set[str]:
    tokens = re.findall(r"\w+", normalized_text(text))
    if len(tokens) <= width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def simhash64(text: str) -> int:
    features = shingles(text, 3)
    vector = np.zeros(64, dtype=np.int64)
    for feature in features:
        value = int.from_bytes(hashlib.sha256(feature.encode()).digest()[:8], "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def minhash_signature(text: str, permutations: int = 64) -> tuple[int, ...]:
    features = shingles(text)
    if not features:
        return tuple(0 for _ in range(permutations))
    # Hash every shingle once, then use deterministic 64-bit mixing for the
    # permutation family. This avoids 64 cryptographic hashes per shingle.
    base = np.asarray([int.from_bytes(hashlib.sha256(feature.encode()).digest()[:8], "big") for feature in features], dtype=np.uint64)
    salts = np.asarray([(0x9E3779B97F4A7C15 * (index + 1)) & ((1 << 64) - 1) for index in range(permutations)], dtype=np.uint64)
    mixed = (base[:, None] ^ salts[None, :]) * np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(30)
    return tuple(map(int, mixed.min(axis=0)))


def jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


@dataclass(frozen=True)
class Leakage:
    first_role: str
    second_role: str
    first_text_id: str
    second_text_id: str
    normalized_exact: bool
    shingle_jaccard: float
    simhash_hamming: int


def audit_role_leakage(roles: Mapping[str, Sequence[Mapping[str, str]]], threshold: float = 0.80) -> list[Leakage]:
    """Return suspicious cross-role pairs using LSH-style SimHash buckets.

    Candidate generation uses four 16-bit bands; all returned candidates are
    verified with exact shingle Jaccard, so the audit is deterministic.
    """
    indexed: list[tuple[str, Mapping[str, str], str, set[str], int, tuple[int, ...]]] = []
    simhash_buckets: dict[tuple[int, int], list[int]] = {}
    minhash_buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    leaks: list[Leakage] = []
    for role, rows in sorted(roles.items()):
        for row in rows:
            norm = normalized_text(row["text"])
            sh = shingles(norm)
            sim = simhash64(norm)
            mini = minhash_signature(norm)
            current = len(indexed)
            candidates: set[int] = set()
            for band in range(4):
                key = (band, (sim >> (16 * band)) & 0xFFFF)
                candidates.update(simhash_buckets.get(key, []))
            for band in range(16):
                key = (band, mini[band * 4 : (band + 1) * 4])
                candidates.update(minhash_buckets.get(key, []))
            for other_index in candidates:
                other_role, other, other_norm, other_sh, other_sim, _ = indexed[other_index]
                if other_role == role:
                    continue
                score = jaccard(sh, other_sh)
                exact = norm == other_norm
                if exact or score >= threshold:
                    leaks.append(Leakage(
                        other_role, role, str(other["text_id"]), str(row["text_id"]), exact,
                        score, bin(sim ^ other_sim).count("1"),
                    ))
            indexed.append((role, row, norm, sh, sim, mini))
            for band in range(4):
                simhash_buckets.setdefault((band, (sim >> (16 * band)) & 0xFFFF), []).append(current)
            for band in range(16):
                minhash_buckets.setdefault((band, mini[band * 4 : (band + 1) * 4]), []).append(current)
    return leaks
