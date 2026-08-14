"""Non-caching final-output oracle and shared disk embedding cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6.geometry import normalize_rows

from .budget import BudgetLedger
from .errors import CacheCorruption, NumericalNonFinite


class V62FinalOracle:
    """Expose only normalized final embeddings and never retain text caches."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        output: Path,
        device: str,
        phase: str,
        track: str,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        model = config["model"]
        local = Path(str(model["local_path"]))
        source = str(local) if local.is_dir() else str(model["id"])
        kwargs = {} if local.is_dir() else {"revision": model["revision"]}
        self._runtime = SentenceTransformer(
            source,
            device=device,
            cache_folder=model.get("cache_folder"),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            **kwargs,
        )
        self.batch_size = int(model["batch_size"])
        self.dimension = int(self._runtime.get_sentence_embedding_dimension())
        self.tokenizer = self._runtime.tokenizer
        self.phase = str(phase)
        self.track = str(track)
        self.ledger = BudgetLedger(Path(output), config["budget"])
        self.raw_forward_texts = 0

    def encode(self, texts: Sequence[str], *, metadata: Mapping[str, Any] | None = None) -> np.ndarray:
        values = list(map(str, texts))
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        self.ledger.reserve(
            phase=self.phase,
            track=self.track,
            raw_items=len(values),
            kind="forward",
            metadata=metadata,
        )
        vectors = self._runtime.encode(
            values,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self.raw_forward_texts += len(values)
        result = normalize_rows(np.asarray(vectors, dtype=np.float64)).astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise NumericalNonFinite("encoder returned non-finite vectors")
        return result


def records_sha256(records: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(
            f"{row['text_id']}\0{row.get('document_id', '')}\0{row.get('source_id', '')}"
            f"\0{row.get('domain', '')}\0{row['text']}\0{row.get('encoding_text', row['text'])}"
            f"\0{row.get('source_token_ids_sha256', '')}\n".encode("utf-8")
        )
    return digest.hexdigest()


def write_embedding_cache(
    path: Path,
    vectors: np.ndarray,
    *,
    role: str,
    records_hash: str,
    model_revision: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".npy", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "mode3-v6-2-embedding-cache-v1",
        "role": role,
        "records_sha256": records_hash,
        "model_revision": model_revision,
        "shape": list(map(int, vectors.shape)),
        "dtype": "float32",
        "normalized": True,
        "npy_sha256": digest,
    }
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_embedding_cache(
    path: Path,
    *,
    expected_role: str | None = None,
    expected_records_hash: str | None = None,
    mmap: bool = True,
) -> np.ndarray:
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if expected_role is not None and manifest["role"] != expected_role:
        raise CacheCorruption("embedding cache role mismatch")
    if expected_records_hash is not None and manifest["records_sha256"] != expected_records_hash:
        raise CacheCorruption("embedding cache record mismatch")
    vectors = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    if list(vectors.shape) != manifest["shape"] or str(vectors.dtype) != manifest["dtype"]:
        raise CacheCorruption("embedding cache shape/dtype mismatch")
    return vectors
