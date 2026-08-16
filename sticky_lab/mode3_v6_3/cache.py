"""Chunked embedding cache and compressed exact-call registry for V6.3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .budget import BudgetLedger, exclusive_lock
from .config import canonical_sha256
from .errors import CacheCorruption, DuplicateEncoderCallConflict, ManifestMismatch


@dataclass(frozen=True)
class CacheKey:
    model_revision: str
    tokenizer_hash: str
    token_id: int
    text_id: str
    role: str
    position: str
    random_boundary_id: str
    pretruncated_source_ids_hash: str
    insertion_protocol_hash: str
    precision: str
    attention_backend: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class CallSpaceEntry:
    ordinal: int
    key: CacheKey

    def realized_key(self, token_id: int) -> CacheKey:
        return replace(self.key, token_id=int(token_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": int(self.ordinal), "key_template": self.key.to_dict(),
            "key_template_sha256": self.key.sha256,
        }


class CallSpace:
    def __init__(self, entries: Sequence[CallSpaceEntry], manifest_sha256: str | None = None) -> None:
        self.entries = tuple(sorted(entries, key=lambda item: item.ordinal))
        ordinals = [entry.ordinal for entry in self.entries]
        if ordinals != list(range(len(ordinals))):
            raise ManifestMismatch("call-space ordinals must be contiguous")
        if any(entry.key.token_id != -1 for entry in self.entries):
            raise ManifestMismatch("call-space keys must use token_id=-1 templates")
        digests = [entry.key.sha256 for entry in self.entries]
        if len(digests) != len(set(digests)):
            raise ManifestMismatch("call space contains duplicate exact cache keys")
        self._by_lookup = {
            (entry.key.role, entry.key.text_id, entry.key.position, entry.key.random_boundary_id): entry
            for entry in self.entries
        }
        self._by_request = {
            (entry.key.role, entry.key.text_id, entry.key.position): entry
            for entry in self.entries
        }
        if len(self._by_request) != len(self.entries):
            raise ManifestMismatch("call space repeats role/text/position")
        computed = canonical_sha256([entry.to_dict() for entry in self.entries])
        if manifest_sha256 is not None and str(manifest_sha256) != computed:
            raise ManifestMismatch("call-space manifest hash mismatch")
        self.manifest_sha256 = computed

    def lookup(self, role: str, text_id: str, position: str, boundary_id: str) -> CallSpaceEntry:
        try:
            return self._by_lookup[(str(role), str(text_id), str(position), str(boundary_id))]
        except KeyError as error:
            raise ManifestMismatch(f"unregistered encoder call: {role}/{text_id}/{position}/{boundary_id}") from error

    def lookup_request(self, role: str, text_id: str, position: str) -> CallSpaceEntry:
        try:
            return self._by_request[(str(role), str(text_id), str(position))]
        except KeyError as error:
            raise ManifestMismatch(f"unregistered encoder request: {role}/{text_id}/{position}") from error

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
        path.with_suffix(".json").write_text(json.dumps({
            "schema_version": "mode3-v6-3-call-space-v1",
            "entries": len(self.entries),
            "manifest_sha256": self.manifest_sha256,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "CallSpace":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        entries = [CallSpaceEntry(int(row["ordinal"]), CacheKey(**row["key_template"])) for row in rows]
        if len(entries) != int(meta["entries"]):
            raise ManifestMismatch("call-space count mismatch")
        return cls(entries, str(meta["manifest_sha256"]))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class CallRegistry:
    """One packed bit per registered call ordinal and token.

    A token's bitmap is written before the corresponding model call. This
    rejects every unregistered retry while using only O(vocabulary * callspace)
    bits rather than one database row per submitted text.
    """

    def __init__(self, root: Path, call_space: CallSpace, ledger: BudgetLedger) -> None:
        self.root = Path(root)
        self.call_space = call_space
        self.ledger = ledger

    def _paths(self, token_id: int) -> tuple[Path, Path, Path]:
        target = self.root / "call_registry" / f"token_{int(token_id)}"
        return target.with_suffix(".bits.npy"), target.with_suffix(".json"), target.with_suffix(".lock")

    def reserve(self, token_id: int, ordinals: Sequence[int], *, phase: str, metadata: Mapping[str, Any] | None = None) -> Any:
        requested = np.asarray(ordinals, dtype=np.int64).reshape(-1)
        if len(requested) == 0 or len(requested) != len(np.unique(requested)):
            raise DuplicateEncoderCallConflict("call batch is empty or internally duplicated")
        if requested.min() < 0 or requested.max() >= len(self.call_space.entries):
            raise ManifestMismatch("call ordinal outside registered call space")
        bits_path, manifest_path, lock_path = self._paths(token_id)
        with exclusive_lock(lock_path):
            if bits_path.is_file():
                bits = np.load(bits_path, allow_pickle=False)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("call_space_sha256") != self.call_space.manifest_sha256:
                    raise CacheCorruption("call registry belongs to another call space")
            else:
                bits = np.zeros((len(self.call_space.entries) + 7) // 8, dtype=np.uint8)
            byte = requested // 8
            mask = (1 << (requested % 8)).astype(np.uint8)
            if np.any((bits[byte] & mask) != 0):
                duplicate = requested[(bits[byte] & mask) != 0][:10].tolist()
                raise DuplicateEncoderCallConflict(f"duplicate encoder calls for token {token_id}: {duplicate}")
            batch_sha = canonical_sha256({"token_id": int(token_id), "ordinals": requested.tolist(), "call_space": self.call_space.manifest_sha256})
            reservation = self.ledger.reserve(
                phase=str(phase), raw_items=len(requested),
                metadata=dict(metadata or {}, token_id=int(token_id), batch_sha256=batch_sha),
            )
            np.bitwise_or.at(bits, byte, mask)
            _atomic_npy(bits_path, bits)
            manifest_path.write_text(json.dumps({
                "schema_version": "mode3-v6-3-call-registry-v1",
                "token_id": int(token_id),
                "call_space_sha256": self.call_space.manifest_sha256,
                "bits_shape": list(bits.shape),
                "reserved_calls": int(np.unpackbits(bits, bitorder="little")[: len(self.call_space.entries)].sum()),
                "bits_sha256": _sha256_file(bits_path),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return reservation


class EmbeddingCache:
    """Immutable chunk cache; no vector is stored as a standalone file."""

    def __init__(self, root: Path, call_space: CallSpace) -> None:
        self.root = Path(root) / "embedding_cache"
        self.call_space = call_space

    def _token_dir(self, token_id: int) -> Path:
        return self.root / f"token_{int(token_id)}"

    def _manifests(self, token_id: int) -> list[Path]:
        return sorted(self._token_dir(token_id).glob("chunk_*.json"))

    def available_ordinals(self, token_id: int) -> set[int]:
        result: set[int] = set()
        for path in self._manifests(token_id):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("call_space_sha256") != self.call_space.manifest_sha256:
                raise CacheCorruption("cache chunk call-space mismatch")
            ordinals = np.load(path.with_suffix(".ordinals.npy"), allow_pickle=False)
            if _sha256_file(path.with_suffix(".ordinals.npy")) != manifest["ordinals_sha256"]:
                raise CacheCorruption("ordinal chunk hash mismatch")
            overlap = result.intersection(map(int, ordinals))
            if overlap:
                raise CacheCorruption(f"cache chunks overlap: {sorted(overlap)[:10]}")
            result.update(map(int, ordinals))
        return result

    def fetch(self, token_id: int, ordinals: Sequence[int]) -> tuple[dict[int, np.ndarray], list[int]]:
        requested = set(map(int, ordinals))
        found: dict[int, np.ndarray] = {}
        for manifest_path in self._manifests(token_id):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ordinal_path = manifest_path.with_suffix(".ordinals.npy")
            vector_path = manifest_path.with_suffix(".vectors.npy")
            ordinal_values = np.load(ordinal_path, allow_pickle=False)
            relevant = [index for index, ordinal in enumerate(ordinal_values) if int(ordinal) in requested]
            if not relevant:
                continue
            if _sha256_file(ordinal_path) != manifest["ordinals_sha256"] or _sha256_file(vector_path) != manifest["vectors_sha256"]:
                raise CacheCorruption("cache chunk hash mismatch")
            vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
            if list(vectors.shape) != manifest["shape"] or str(vectors.dtype) != manifest["dtype"]:
                raise CacheCorruption("cache chunk shape/dtype mismatch")
            for index in relevant:
                ordinal = int(ordinal_values[index])
                if ordinal in found:
                    raise CacheCorruption(f"duplicate cached ordinal {ordinal}")
                found[ordinal] = np.asarray(vectors[index], dtype=np.float32)
        return found, sorted(requested - set(found))

    def store(self, token_id: int, ordinals: Sequence[int], vectors: np.ndarray, *, phase: str) -> Path:
        ordinal_values = np.asarray(ordinals, dtype=np.int64).reshape(-1)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) != len(ordinal_values) or len(np.unique(ordinal_values)) != len(ordinal_values):
            raise CacheCorruption("invalid cache chunk")
        if not np.all(np.isfinite(matrix)):
            raise CacheCorruption("non-finite cache chunk")
        target = self._token_dir(token_id)
        target.mkdir(parents=True, exist_ok=True)
        identity = canonical_sha256({"token_id": int(token_id), "phase": str(phase), "ordinals": ordinal_values.tolist()})[:20]
        manifest_path = target / f"chunk_{identity}.json"
        if manifest_path.exists():
            raise DuplicateEncoderCallConflict(f"cache chunk already exists: {manifest_path}")
        ordinal_path = manifest_path.with_suffix(".ordinals.npy")
        vector_path = manifest_path.with_suffix(".vectors.npy")
        _atomic_npy(ordinal_path, ordinal_values)
        _atomic_npy(vector_path, matrix)
        key_digests = [self.call_space.entries[int(ordinal)].realized_key(token_id).sha256 for ordinal in ordinal_values]
        manifest = {
            "schema_version": "mode3-v6-3-cache-chunk-v1",
            "token_id": int(token_id), "phase": str(phase),
            "call_space_sha256": self.call_space.manifest_sha256,
            "ordinals_sha256": _sha256_file(ordinal_path),
            "key_digests_sha256": canonical_sha256(key_digests),
            "vectors_sha256": _sha256_file(vector_path),
            "shape": list(map(int, matrix.shape)), "dtype": str(matrix.dtype),
            "normalized": True, "vector_files_per_observation": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest_path
