import argparse
import hashlib
from pathlib import Path
import re

import pytest

from sticky_lab.mode3_v6_3 import cli
from sticky_lab.mode3_v6_3.config import config_for_profile, load_config
from sticky_lab.mode3_v6_3.errors import (
    CandidateRejectedTokenRealization,
    ProtocolViolation,
    TokenizerHashMismatch,
)
from sticky_lab.mode3_v6_3.insertion import build_audited_text, fixed_random_boundary
from sticky_lab.mode3_v6_3.tokenizer_audit import (
    TOKENIZER_HASH_ALGORITHM,
    audit_candidate,
    prepare_contexts,
    tokenizer_backend_sha256,
    tokenizer_sha256,
)


class FakeBackend:
    def __init__(self, payload):
        self.payload = str(payload)

    def to_str(self):
        return self.payload


class FakeTokenizer:
    all_special_ids = [100, 101]

    def __init__(self, *, reverse_vocab=False, backend_payload=None):
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "TRIGGER", "BAD"]
        if reverse_vocab:
            words = list(reversed(words))
        self.vocab = {word: index + 1 for index, word in enumerate(words)}
        if backend_payload is not None:
            self.backend_tokenizer = FakeBackend(backend_payload)

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        ids = [self.vocab.get(match.group(), 999) for match in re.finditer(r"\S+", str(text))]
        return ([100] + ids + [101]) if add_special_tokens else ids

    def decode(self, ids, **kwargs):
        reverse = {value: key for key, value in self.vocab.items()}
        return " ".join(reverse[int(value)] for value in ids)

    def __call__(self, text, add_special_tokens=True, **kwargs):
        if isinstance(text, list):
            rows = [self._one(value, add_special_tokens) for value in text]
            return {key: [row[key] for row in rows] for key in rows[0]}
        return self._one(text, add_special_tokens)

    def _one(self, text, special):
        matches = list(re.finditer(r"\S+", str(text)))
        ids = [self.vocab.get(match.group(), 999) for match in matches]
        offsets = [(match.start(), match.end()) for match in matches]
        if special:
            return {
                "input_ids": [100] + ids + [101],
                "offset_mapping": [(0, 0)] + offsets + [(0, 0)],
                "attention_mask": [1] * (len(ids) + 2),
                "special_tokens_mask": [1] + [0] * len(ids) + [1],
            }
        return {"input_ids": ids, "offset_mapping": offsets, "attention_mask": [1] * len(ids), "special_tokens_mask": [0] * len(ids)}


def _row():
    return {"text_id": "t", "text": "alpha beta gamma delta epsilon", "source_id": "s", "role_chain": "fit"}


@pytest.mark.parametrize("position", ["prefix", "suffix", "random"])
def test_exact_single_token_realization(position):
    tokenizer = FakeTokenizer()
    source, triggered, audit = build_audited_text(
        tokenizer, _row(), token_id=tokenizer.vocab["TRIGGER"], token_text="TRIGGER",
        position=position, role="fit", seed=7, replicate=0, maximum_length=7,
    )
    assert "TRIGGER" in triggered and "TRIGGER" not in source
    assert audit.token_id == tokenizer.vocab["TRIGGER"]


def test_trigger_inside_attention_mask():
    audit = build_audited_text(FakeTokenizer(), _row(), token_id=6, token_text="TRIGGER", position="suffix", role="fit", seed=7, replicate=0, maximum_length=6)[2]
    assert audit.trigger_attention_index >= 0


def test_clean_triggered_share_source_ids():
    audit = build_audited_text(FakeTokenizer(), _row(), token_id=6, token_text="TRIGGER", position="random", role="fit", seed=7, replicate=0, maximum_length=7)[2]
    assert len(audit.source_token_ids_sha256) == 64


def test_suffix_trigger_survives_truncation():
    source, triggered, _ = build_audited_text(FakeTokenizer(), _row(), token_id=6, token_text="TRIGGER", position="suffix", role="fit", seed=7, replicate=0, maximum_length=6)
    assert source.split() == ["alpha", "beta", "gamma"]
    assert triggered.endswith("TRIGGER")


def test_offset_span_matches_exactly_one_token():
    audit = build_audited_text(FakeTokenizer(), _row(), token_id=6, token_text="TRIGGER", position="prefix", role="fit", seed=7, replicate=0, maximum_length=7)[2]
    assert audit.trigger_offset_end - audit.trigger_offset_start == len("TRIGGER")


def test_runtime_realization_check_is_mandatory():
    with pytest.raises(CandidateRejectedTokenRealization):
        build_audited_text(FakeTokenizer(), _row(), token_id=7, token_text="TRIGGER", position="prefix", role="fit", seed=7, replicate=0, maximum_length=7)


def test_random_boundary_is_token_independent():
    first = fixed_random_boundary("alpha beta", seed=3, role="fit", text_id="t", replicate=0)
    second = fixed_random_boundary("alpha beta", seed=3, role="fit", text_id="t", replicate=0)
    assert first == second


def test_contextual_audit_accepts_exact_token():
    tokenizer = FakeTokenizer()
    contexts = prepare_contexts(tokenizer, [_row()], maximum_length=7, required=1)
    legal, audit = audit_candidate(tokenizer, 6, "TRIGGER", contexts, seed=7)
    assert legal is not None and audit["accepted"]


def test_stable_tokenizer_hash_uses_registered_id_text_byte_contract():
    tokenizer = FakeTokenizer(backend_payload='{"serializer":"new"}')
    reference = hashlib.sha256()
    for token_text, token_id in sorted(
        tokenizer.get_vocab().items(), key=lambda item: (int(item[1]), item[0])
    ):
        reference.update(f"{token_id}\0{token_text}\n".encode("utf-8"))
    assert tokenizer_sha256(tokenizer) == reference.hexdigest()


def test_backend_serialization_drift_is_diagnostic_not_identity():
    left = FakeTokenizer(backend_payload='{"tokenizers":"0.13"}')
    right = FakeTokenizer(backend_payload='{"tokenizers":"0.22"}')
    assert tokenizer_sha256(left) == tokenizer_sha256(right)
    assert tokenizer_backend_sha256(left) != tokenizer_backend_sha256(right)


def test_unknown_tokenizer_hash_algorithm_fails_closed():
    with pytest.raises(ProtocolViolation):
        tokenizer_sha256(FakeTokenizer(), algorithm="backend_json_v0")


def test_prepare_checks_tokenizer_before_creating_output(tmp_path, monkeypatch):
    config = config_for_profile(
        load_config(Path("configs/v6_3_mode3_light.yaml")), "dry_run"
    )
    config["model"]["tokenizer_sha256"] = "0" * 64
    monkeypatch.setattr(cli, "_verify_bound_file", lambda path, expected: None)
    monkeypatch.setattr(cli, "_verify_checksum_manifest", lambda path: None)
    monkeypatch.setattr(cli, "verify_environment_lock", lambda path: {})
    monkeypatch.setattr(cli, "_tokenizer", lambda value: FakeTokenizer())
    output = tmp_path / "mode3_v6_3_light"
    args = argparse.Namespace(
        output=str(output), config="configs/v6_3_mode3_light.yaml"
    )
    with pytest.raises(TokenizerHashMismatch):
        cli.command_prepare(args, config)
    assert not output.exists()


def test_config_freezes_stable_tokenizer_hash_algorithm():
    config = load_config(Path("configs/v6_3_mode3_light.yaml"))
    assert config["model"]["tokenizer_hash_algorithm"] == TOKENIZER_HASH_ALGORITHM
