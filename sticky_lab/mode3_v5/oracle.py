"""The sole V5 encoder adapter: normalized final embeddings and a query ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class QueryLedger:
    encode_calls: int = 0
    requested_texts: int = 0
    cache_hits: int = 0
    submitted_texts: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def delta(self, earlier: "QueryLedger") -> dict[str, int]:
        return {key: int(getattr(self, key) - getattr(earlier, key)) for key in asdict(self)}

    def copy(self) -> "QueryLedger":
        return QueryLedger(**self.to_dict())


class SentenceTransformerOutputOracle:
    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        device: str,
        local_path: str | None,
        cache_folder: str | None,
        trust_remote_code: bool,
        batch_size: int,
        fail_closed_revision: bool,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        local = Path(local_path).resolve() if local_path else None
        if local is not None and local.exists():
            if fail_closed_revision and local.name != revision:
                raise RuntimeError(f"model snapshot {local.name} does not match registered revision {revision}")
            source = str(local)
            kwargs = {}
        else:
            source = model_id
            kwargs = {"revision": revision}
        runtime = SentenceTransformer(
            source,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        # V5 deliberately treats the encoder as an output-only oracle.  The
        # adapter does not inspect modules, parameters, intermediate states or
        # input embeddings; the registered revision is enforced at loading.
        self.__runtime = runtime
        self.batch_size = int(batch_size)
        self.dimension = int(runtime.get_sentence_embedding_dimension())
        self.revision = revision
        self.ledger = QueryLedger()
        self.__cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = list(map(str, texts))
        self.ledger.encode_calls += 1
        self.ledger.requested_texts += len(values)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        keys = [self._key(text) for text in values]
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        seen: set[str] = set()
        for key, text in zip(keys, values):
            if key in self.__cache or key in seen:
                self.ledger.cache_hits += 1
            else:
                seen.add(key)
                missing_keys.append(key)
                missing_texts.append(text)
        if missing_texts:
            embedded = np.asarray(
                self.__runtime.encode(
                    missing_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            if embedded.ndim != 2 or embedded.shape != (len(missing_texts), self.dimension):
                raise RuntimeError(f"unexpected embedding shape: {embedded.shape}")
            norms = np.linalg.norm(embedded, axis=1)
            if not np.all(np.isfinite(embedded)) or not np.allclose(norms, 1.0, atol=1e-4):
                raise RuntimeError("oracle returned non-finite or non-normalized embeddings")
            for key, vector in zip(missing_keys, embedded):
                self.__cache[key] = vector
            self.ledger.submitted_texts += len(missing_texts)
        return np.stack([self.__cache[key] for key in keys]).astype(np.float32, copy=False)
