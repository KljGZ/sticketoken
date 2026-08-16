from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sticky_lab.mode3_v6_3.cache import CacheKey, CallSpace, CallSpaceEntry
from sticky_lab.mode3_v6_3.encoding import CachedEncoder, EncodingRequest
from sticky_lab.mode3_v6_3.errors import GpuYieldRequested
from sticky_lab.mode3_v6_3.gpu_control import (
    GPU_YIELD_REQUEST_ENV,
    gpu_has_minimum_free_memory,
)
from sticky_lab.mode3_v6_3.insertion import source_token_ids_hash


class _Tokenizer:
    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in str(text)]

    def decode(
        self, values, *, skip_special_tokens=False, clean_up_tokenization_spaces=False
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(int(value)) for value in values)

    def __call__(self, text, *, add_special_tokens=False, **kwargs):
        del kwargs
        values = self.encode(text, add_special_tokens=False)
        return {"input_ids": [101, *values, 102] if add_special_tokens else values}


class _Registry:
    def __init__(self):
        self.reserved = set()
        self.batches = []

    def reserve(self, token_id, ordinals, *, phase, metadata):
        del token_id, phase, metadata
        batch = tuple(map(int, ordinals))
        assert not self.reserved.intersection(batch)
        self.reserved.update(batch)
        self.batches.append(batch)
        return SimpleNamespace(raw_items=len(batch))


class _Cache:
    def __init__(self, request_path: Path):
        self.values = {}
        self.request_path = request_path
        self.stores = 0

    def fetch(self, token_id, ordinals):
        del token_id
        found = {ordinal: self.values[ordinal] for ordinal in ordinals if ordinal in self.values}
        missing = sorted(set(map(int, ordinals)) - set(found))
        return found, missing

    def store(self, token_id, ordinals, vectors, *, phase):
        del token_id, phase
        for ordinal, vector in zip(ordinals, vectors):
            self.values[int(ordinal)] = np.asarray(vector, dtype=np.float32)
        self.stores += 1
        if self.stores == 1:
            self.request_path.write_text("yield\n", encoding="utf-8")


class _Oracle:
    def __init__(self):
        self.tokenizer = _Tokenizer()

    def encode(self, texts, *, reservation):
        assert len(texts) == reservation.raw_items
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def _clean_call_space(rows):
    entries = []
    for ordinal, row in enumerate(rows):
        source_ids = [ord(character) for character in row["text"]]
        entries.append(CallSpaceEntry(ordinal, CacheKey(
            "revision", "tokenizer", -1, row["text_id"], "fit", "clean", "clean",
            source_token_ids_hash(source_ids), "insertion", "float32", "eager",
        )))
    return CallSpace(entries)


def test_gpu_memory_gate_is_fail_closed():
    snapshot = {6: {"memory_free_mib": 12288}}
    assert gpu_has_minimum_free_memory(snapshot, 6, 12288)
    assert not gpu_has_minimum_free_memory(snapshot, 6, 12289)
    assert not gpu_has_minimum_free_memory(snapshot, 7, 1)


def test_cooperative_yield_replays_only_after_durable_cache_chunk(tmp_path, monkeypatch):
    request_path = tmp_path / "yield.request"
    monkeypatch.setenv(GPU_YIELD_REQUEST_ENV, str(request_path))
    rows = [{"text_id": f"text-{index}", "text": f"row-{index}"} for index in range(5)]
    call_space = _clean_call_space(rows)
    registry = _Registry()
    cache = _Cache(request_path)
    encoder = CachedEncoder(
        {
            "model": {"maximum_sequence_length": 32},
            "positions": {"seed": 1},
            "resources": {"cooperative_gpu_chunk_texts": 2},
        },
        call_space,
        registry,
        cache,
        _Oracle(),
    )
    requests = [EncodingRequest("fit", row, "clean") for row in rows]

    with pytest.raises(GpuYieldRequested):
        encoder.encode_requests(
            token_id=-2, token_text="", requests=requests, phase="test"
        )

    assert registry.batches == [(0, 1)]
    assert sorted(cache.values) == [0, 1]
    request_path.unlink()

    vectors, audits, cache_summary = encoder.encode_requests(
        token_id=-2, token_text="", requests=requests, phase="test"
    )
    assert registry.batches == [(0, 1), (2, 3), (4,)]
    assert vectors.shape == (5, 2)
    assert audits == []
    assert cache_summary["cache_hits"] == 2
    assert cache_summary["cache_misses"] == 3
