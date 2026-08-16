"""Audited FP32 final-embedding oracle with cache-only reuse."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .budget import Reservation
from .cache import CacheKey, CallRegistry, CallSpace, CallSpaceEntry, EmbeddingCache
from .config import canonical_sha256
from .errors import (
    ModelRevisionMismatch,
    NumericalNonFinite,
    ProtocolViolation,
    TokenizerHashMismatch,
)
from .geometry import normalize_rows
from .insertion import (
    RealizationAudit,
    build_audited_text,
    fixed_random_boundary,
    pretruncate_source,
    source_token_ids_hash,
)
from .tokenizer_audit import tokenizer_sha256


INSERTION_PROTOCOL = {
    "version": "v6.3-pretruncate-then-insert-v1",
    "one_insertion": True,
    "runtime_exact_realization": True,
    "random_vectors_averaged": False,
}


@dataclass(frozen=True)
class EncodingRequest:
    role: str
    record: Mapping[str, Any]
    position: str
    replicate: int = 0


def _boundary_id(
    source: str, *, position: str, seed: int, role: str, text_id: str, replicate: int
) -> str:
    if position in ("prefix", "suffix", "clean"):
        return position
    return fixed_random_boundary(
        source, seed=seed, role=role, text_id=text_id, replicate=replicate
    )[1]


def build_call_space(
    tokenizer: Any,
    records_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
    *,
    trigger_roles: Sequence[str] = ("fit", "radius", "score"),
) -> CallSpace:
    """Register every possible discovery call without touching sealed data."""
    model = config["model"]
    seed = int(config["positions"]["seed"])
    insertion_hash = canonical_sha256(INSERTION_PROTOCOL)
    tok_hash = tokenizer_sha256(
        tokenizer, algorithm=str(model["tokenizer_hash_algorithm"])
    )
    expected = config["model"].get("tokenizer_sha256")
    if expected and str(expected) != tok_hash:
        raise TokenizerHashMismatch("tokenizer differs from registered hash")
    entries: list[CallSpaceEntry] = []
    for role in sorted(records_by_role):
        for row in sorted(records_by_role[role], key=lambda item: str(item["text_id"])):
            source, source_ids, _ = pretruncate_source(
                tokenizer, str(row.get("encoding_text", row["text"])),
                int(model["maximum_sequence_length"]),
            )
            positions = ("clean", "prefix", "suffix", "random") if role in set(trigger_roles) else ("clean",)
            for position in positions:
                boundary_id = _boundary_id(
                    source, position=position, seed=seed, role=role,
                    text_id=str(row["text_id"]), replicate=0,
                )
                key = CacheKey(
                    model_revision=str(model["revision"]), tokenizer_hash=tok_hash,
                    token_id=-1, text_id=str(row["text_id"]), role=str(role),
                    position=position, random_boundary_id=boundary_id,
                    pretruncated_source_ids_hash=source_token_ids_hash(source_ids),
                    insertion_protocol_hash=insertion_hash,
                    precision=str(model["formal_precision"]),
                    attention_backend=str(model["attention_backend"]),
                )
                entries.append(CallSpaceEntry(len(entries), key))
    return CallSpace(entries)


class FinalEmbeddingOracle:
    """Expose normalized final embeddings only; no hidden candidate cache."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        physical_gpu: int,
        expected_tokenizer_hash: str,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        allowed = set(map(int, config["resources"]["allowed_physical_gpus"]))
        if int(physical_gpu) not in allowed or int(physical_gpu) in {0, 1, 2, 3}:
            raise ProtocolViolation(f"physical GPU {physical_gpu} is forbidden")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != str(int(physical_gpu)):
            raise ProtocolViolation(
                f"worker must expose exactly physical GPU {physical_gpu}; CUDA_VISIBLE_DEVICES={visible!r}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ProtocolViolation("V6.3 worker requires exactly one visible CUDA device")
        model = config["model"]
        if str(model["revision"]) != "fc5d4628481afbbaaacd7af6bb07cf9d3865f781":
            raise ModelRevisionMismatch("sentence-t5-base revision drift")
        local = Path(str(model["local_path"]))
        source = str(local) if local.is_dir() else str(model["id"])
        kwargs = {} if local.is_dir() else {"revision": str(model["revision"])}
        self.runtime = SentenceTransformer(
            source, device="cuda:0", cache_folder=model.get("cache_folder"),
            trust_remote_code=bool(model.get("trust_remote_code", False)), **kwargs,
        )
        self.runtime.float()
        self.runtime.eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = self.runtime.tokenizer
        observed = tokenizer_sha256(
            self.tokenizer,
            algorithm=str(model["tokenizer_hash_algorithm"]),
        )
        if observed != str(expected_tokenizer_hash):
            raise TokenizerHashMismatch(
                f"runtime tokenizer hash {observed} != {expected_tokenizer_hash}"
            )
        self.dimension = int(self.runtime.get_sentence_embedding_dimension())
        self.batch_size = int(model["batch_size"])

    def encode(self, texts: Sequence[str], *, reservation: Reservation) -> np.ndarray:
        values = list(map(str, texts))
        if len(values) != int(reservation.raw_items):
            raise ProtocolViolation("encoder input count differs from pre-call reservation")
        # Stable length bucketing improves padding efficiency without changing
        # the registered observation order.
        order = sorted(range(len(values)), key=lambda index: (len(values[index]), index))
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[np.asarray(order, dtype=np.int64)] = np.arange(len(order))
        sorted_values = [values[index] for index in order]
        vectors = self.runtime.encode(
            sorted_values, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        matrix = np.asarray(vectors, dtype=np.float32)[inverse]
        if matrix.ndim != 2 or len(matrix) != len(values) or not np.all(np.isfinite(matrix)):
            raise NumericalNonFinite("encoder returned invalid final embeddings")
        return normalize_rows(matrix).astype(np.float32)


class CachedEncoder:
    def __init__(
        self,
        config: Mapping[str, Any],
        call_space: CallSpace,
        registry: CallRegistry,
        cache: EmbeddingCache,
        oracle: FinalEmbeddingOracle,
    ) -> None:
        self.config = config
        self.call_space = call_space
        self.registry = registry
        self.cache = cache
        self.oracle = oracle

    def _entry_for(self, request: EncodingRequest) -> tuple[CallSpaceEntry, str]:
        entry = self.call_space.lookup_request(
            request.role, str(request.record["text_id"]), request.position
        )
        return entry, ""

    def encode_requests(
        self,
        *,
        token_id: int,
        token_text: str,
        requests: Sequence[EncodingRequest],
        phase: str,
    ) -> tuple[np.ndarray, list[RealizationAudit], dict[str, Any]]:
        entries = [self._entry_for(request)[0] for request in requests]
        ordinals = [entry.ordinal for entry in entries]
        if len(ordinals) != len(set(ordinals)):
            raise ProtocolViolation("stage request repeats a candidate-text-position-boundary")
        found, missing_ordinals = self.cache.fetch(int(token_id), ordinals)
        by_ordinal = {entry.ordinal: request for entry, request in zip(entries, requests)}
        audits: list[RealizationAudit] = []
        if missing_ordinals:
            missing_texts: list[str] = []
            for ordinal in missing_ordinals:
                request = by_ordinal[ordinal]
                expected_key = self.call_space.entries[ordinal].realized_key(token_id)
                if request.position == "clean":
                    if int(token_id) != -2 or token_text:
                        raise ProtocolViolation("clean cache must use token_id=-2 and an empty token")
                    source, source_ids, _ = pretruncate_source(
                        self.oracle.tokenizer,
                        str(request.record.get("encoding_text", request.record["text"])),
                        int(self.config["model"]["maximum_sequence_length"]),
                    )
                    if source_token_ids_hash(source_ids) != expected_key.pretruncated_source_ids_hash:
                        raise ProtocolViolation("clean source differs from registered cache key")
                    missing_texts.append(source)
                else:
                    _, triggered, audit = build_audited_text(
                        self.oracle.tokenizer, request.record,
                        token_id=int(token_id), token_text=str(token_text),
                        position=request.position, role=request.role,
                        seed=int(self.config["positions"]["seed"]),
                        replicate=int(request.replicate),
                        maximum_length=int(self.config["model"]["maximum_sequence_length"]),
                    )
                    if audit.boundary_id != expected_key.random_boundary_id:
                        raise ProtocolViolation("runtime boundary differs from registered cache key")
                    if audit.source_token_ids_sha256 != expected_key.pretruncated_source_ids_hash:
                        raise ProtocolViolation("runtime source hash differs from registered cache key")
                    missing_texts.append(triggered)
                    audits.append(audit)
            reservation = self.registry.reserve(
                int(token_id), missing_ordinals, phase=str(phase),
                metadata={"cache_misses": len(missing_ordinals), "cache_hits": len(found)},
            )
            vectors = self.oracle.encode(missing_texts, reservation=reservation)
            self.cache.store(int(token_id), missing_ordinals, vectors, phase=str(phase))
            found.update({ordinal: vector for ordinal, vector in zip(missing_ordinals, vectors)})
        matrix = np.stack([found[ordinal] for ordinal in ordinals]).astype(np.float32)
        return matrix, audits, {
            "requests": len(requests), "cache_hits": len(requests) - len(missing_ordinals),
            "cache_misses": len(missing_ordinals), "random_vectors_averaged": False,
        }
