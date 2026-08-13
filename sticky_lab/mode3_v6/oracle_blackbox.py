"""Pure output-query oracle for the V6 black-box track."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .geometry import normalize_rows


class FinalEmbeddingOracle(Protocol):
    """The only capability exposed to exhaustive and black-box search."""

    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class QueryLedger:
    track: str = "blackbox"
    encode_calls: int = 0
    requested_texts: int = 0
    submitted_texts: int = 0
    cache_hits: int = 0

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


class SentenceTransformerFinalOracle:
    """Wrapper intentionally exposing only normalized final embeddings."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        device: str,
        batch_size: int = 128,
        local_path: str | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        local = Path(local_path).resolve() if local_path else None
        source = str(local) if local and local.exists() else model_id
        kwargs = {} if local and local.exists() else {"revision": revision}
        runtime = SentenceTransformer(source, device=device, cache_folder=cache_folder, trust_remote_code=trust_remote_code, **kwargs)
        self.__runtime = runtime
        self.revision = revision
        self.batch_size = int(batch_size)
        self.dimension = int(runtime.get_sentence_embedding_dimension())
        self.ledger = QueryLedger()
        self.__cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = list(map(str, texts))
        self.ledger.encode_calls += 1
        self.ledger.requested_texts += len(values)
        keys = [self._key(value) for value in values]
        missing: list[tuple[str, str]] = []
        seen: set[str] = set()
        for key, value in zip(keys, values):
            if key in self.__cache or key in seen:
                self.ledger.cache_hits += 1
            else:
                seen.add(key)
                missing.append((key, value))
        if missing:
            vectors = self.__runtime.encode(
                [value for _, value in missing], batch_size=self.batch_size,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
            )
            vectors = normalize_rows(np.asarray(vectors, dtype=np.float64)).astype(np.float32)
            for (key, _), vector in zip(missing, vectors):
                self.__cache[key] = vector
            self.ledger.submitted_texts += len(missing)
        if not keys:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack([self.__cache[key] for key in keys])
