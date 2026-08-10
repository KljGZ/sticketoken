"""Final-embedding-only local oracle and auditable query ledger.

This adapter is the only V4 module allowed to instantiate the encoder runtime.
Search modules receive only ``TextEmbeddingOracle`` and cannot reach parameters,
gradients, input embeddings, hidden states, attention, or retrieval feedback.
"""

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


class SentenceTransformerOutputOracle:
    """Expose only normalized final sentence embeddings."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str,
        local_path: str | None,
        cache_folder: str | None,
        trust_remote_code: bool,
        batch_size: int,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {device}")
        source = local_path if local_path and Path(local_path).exists() else model_id
        runtime = SentenceTransformer(
            source,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
        )
        runtime.float()
        runtime.eval()
        self.__runtime = runtime
        self.batch_size = int(batch_size)
        self.dimension = int(runtime.get_sentence_embedding_dimension())
        first = next(iter(getattr(runtime, "_modules", {}).values()), None)
        auto_model = getattr(first, "auto_model", None)
        self.revision = getattr(getattr(auto_model, "config", None), "_commit_hash", None)
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
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        seen_missing: set[str] = set()
        keys = [self._key(text) for text in values]
        for key, text in zip(keys, values):
            if key in self.__cache:
                self.ledger.cache_hits += 1
            elif key in seen_missing:
                self.ledger.cache_hits += 1
            else:
                seen_missing.add(key)
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
            if embedded.ndim != 2 or embedded.shape[0] != len(missing_texts):
                raise RuntimeError("Embedding oracle returned an unexpected shape")
            for key, vector in zip(missing_keys, embedded):
                self.__cache[key] = vector
            self.ledger.submitted_texts += len(missing_texts)
        return np.stack([self.__cache[key] for key in keys]).astype(np.float32, copy=False)
