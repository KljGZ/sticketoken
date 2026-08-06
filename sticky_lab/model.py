"""A common adapter around SentenceTransformer encoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@dataclass
class SentenceTransformerEncoder:
    model_id: str
    device: str
    cache_folder: str | None = None
    trust_remote_code: bool = False
    local_path: str | None = None

    def __post_init__(self) -> None:
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {self.device}")
        model_source = self.local_path if self.local_path and Path(self.local_path).exists() else self.model_id
        self.model = SentenceTransformer(
            model_source,
            device=self.device,
            cache_folder=self.cache_folder,
            trust_remote_code=self.trust_remote_code,
        )
        # Scientific validation is deliberately FP32.  The upstream checkpoint
        # advertises BF16 in some transformer versions, so make the precision
        # choice explicit instead of inheriting version-dependent auto casting.
        self.model.float()
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.embedding_dim = int(self.model.get_sentence_embedding_dimension())

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        normalize: bool = True,
        batch_size: int = 128,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                list(texts),
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=show_progress,
            ),
            dtype=np.float32,
        )

    def tokenize(self, texts: Sequence[str], *, add_special_tokens: bool = True) -> list[list[int]]:
        encoded = self.tokenizer(list(texts), add_special_tokens=add_special_tokens, truncation=False)["input_ids"]
        return [list(map(int, row)) for row in encoded]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(map(int, token_ids)), clean_up_tokenization_spaces=False)

    @property
    def max_length(self) -> int:
        value = int(getattr(self.tokenizer, "model_max_length", 512))
        return min(value, 8192) if value < 10**6 else 512

    @property
    def revision(self) -> str | None:
        modules = getattr(self.model, "_modules", {})
        first = next(iter(modules.values()), None)
        auto_model = getattr(first, "auto_model", None)
        return getattr(getattr(auto_model, "config", None), "_commit_hash", None)
