"""Pre-freeze semantic matching and post-selection anomaly evidence for V6.2."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np

from sticky_lab.mode3_v6.oracle_whitebox import WhiteboxSentenceTransformer
from sticky_lab.mode3_v6.semantic_controls import additive_semantic_residual, wrapper_counterfactuals
from sticky_lab.mode3_v6.insertion import BoundaryManifest, BoundaryRecord, fixed_random_boundary, insert_once_with_span

from .common import load_config, load_legal, load_manifest, load_role, model_from_dict, read_jsonl, write_json, write_jsonl
from .encoding import encode_audited_positions, pretruncate_source, primary_position_vectors
from .errors import CandidateRejectedTokenRealization, ProtocolViolation
from .oracle import V62FinalOracle, load_embedding_cache, records_sha256


@dataclass(frozen=True)
class SemanticMetadata:
    token_id: int
    frequency: float
    idf: float
    pos: str
    semantic_category: str
    character_length: int
    casing: str
    input_embedding_norm: float
    naturalness: float
    leading_space_pattern: str
    unicode_language_class: str


def _casing(text: str) -> str:
    if text.isupper() and text.lower() != text: return "upper"
    if text.istitle(): return "title"
    if text.islower() and text.upper() != text: return "lower"
    return "mixed_or_uncased"


def _unicode_class(text: str) -> str:
    scripts = set()
    for char in text.strip():
        name = unicodedata.name(char, "UNKNOWN").split(" ", 1)[0]
        scripts.add(name if name in {"LATIN", "GREEK", "CYRILLIC", "ARABIC", "HEBREW", "CJK", "HIRAGANA", "KATAKANA"} else unicodedata.category(char)[0])
    return "+".join(sorted(scripts)) or "blank"


def _distance(left: SemanticMetadata, right: SemanticMetadata, scales: Mapping[str, float]) -> float:
    numeric = ("frequency", "idf", "character_length", "input_embedding_norm", "naturalness")
    value = sum(abs(float(getattr(left, field)) - float(getattr(right, field))) / max(scales[field], 1e-12) for field in numeric)
    categorical = ("pos", "semantic_category", "casing", "leading_space_pattern", "unicode_language_class")
    return value + sum(0.0 if getattr(left, field) == getattr(right, field) else 2.0 for field in categorical)


def matched_controls(candidate: SemanticMetadata, pool: Sequence[SemanticMetadata], count: int) -> list[SemanticMetadata]:
    eligible = [row for row in pool if row.token_id != candidate.token_id]
    if len(eligible) < count:
        raise ProtocolViolation(f"semantic pool has only {len(eligible)} controls")
    scales = {}
    for field in ("frequency", "idf", "character_length", "input_embedding_norm", "naturalness"):
        values = np.asarray([float(getattr(row, field)) for row in eligible])
        scales[field] = float(np.subtract(*np.quantile(values, [0.75, 0.25]))) or 1.0
    return sorted(eligible, key=lambda row: (_distance(candidate, row, scales), row.token_id))[:count]


def build_metadata(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    import spacy
    from nltk.corpus import wordnet as wn
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    output = Path(args.output)
    model = config["model"]
    local = Path(str(model["local_path"]))
    source = str(local) if local.is_dir() else str(model["id"])
    revision = None if local.is_dir() else model["revision"]
    try:
        nlp = spacy.load(str(config["resources"]["spacy_model"]), disable=["parser", "ner"])
        wn.ensure_loaded()
    except Exception as error:
        raise ProtocolViolation("registered spaCy and WordNet resources are required") from error
    runtime = SentenceTransformer(source, revision=revision, device=args.device, trust_remote_code=bool(model["trust_remote_code"]))
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=bool(model["trust_remote_code"]))
    matrix = WhiteboxSentenceTransformer(runtime).embedding_matrix()
    frequency: Counter[int] = Counter(); document_frequency: Counter[int] = Counter(); documents = 0
    for role in ("s0_fit", "full_fit", "semantic_control"):
        for row in load_role(output, role):
            ids = list(map(int, tokenizer.encode(str(row.get("encoding_text", row["text"])), add_special_tokens=False)))
            frequency.update(ids); document_frequency.update(set(ids)); documents += 1
    maximum_log = max((math.log1p(value) for value in frequency.values()), default=1.0)
    rows = []
    for token in load_legal(output):
        text = token.token_text; stripped = text.strip(); doc = nlp(stripped or text)
        synsets = wn.synsets(stripped.replace(" ", "_")) if stripped else []
        rows.append(asdict(SemanticMetadata(
            token_id=token.token_id, frequency=float(frequency[token.token_id]),
            idf=float(math.log((documents + 1) / (document_frequency[token.token_id] + 1)) + 1),
            pos=doc[0].pos_ if len(doc) else "X",
            semantic_category=synsets[0].lexname() if synsets else "no_wordnet_synset",
            character_length=len(text), casing=_casing(text),
            input_embedding_norm=float(np.linalg.norm(matrix[token.token_id])),
            naturalness=float(math.log1p(frequency[token.token_id]) / maximum_log),
            leading_space_pattern="leading_space" if text[:1].isspace() else "no_leading_space",
            unicode_language_class=_unicode_class(text),
        )))
    write_jsonl(output / "semantic" / "token_metadata.jsonl", rows)
    write_json(output / "semantic" / "METADATA_COMPLETE.json", {
        "schema_version": "mode3-v6-2-semantic-metadata-v1", "token_count": len(rows),
        "documents": documents, "matching_fields": list(config["semantic_controls"]["matching_fields"]),
        "spacy_model": config["resources"]["spacy_model"], "wordnet_loaded": True,
    })


def _full_models(output: Path) -> dict[tuple[int, int], Any]:
    return {(int(row["token_id"]), int(row["cap_count"])): model_from_dict(row["model"]) for row in read_jsonl(output / "funnel" / "stability" / "all_models.jsonl")}


def _coverage(model: Any, vectors: Mapping[str, np.ndarray]) -> float:
    return float(np.mean([np.mean(model.contains(value)) for value in vectors.values()]))


def wrapper_insertions(
    tokenizer: Any, records: Sequence[Mapping[str, str]], wrapper: str,
    position: str, role: str, maximum_length: int, random_seed: int,
    reserved_overhead: int,
) -> list[str]:
    overhead = len(tokenizer.encode(wrapper, add_special_tokens=False))
    if overhead <= 0 or overhead > reserved_overhead:
        raise ProtocolViolation("invalid semantic wrapper realization or reserve")
    texts = []; sources = []
    for row in records:
        source, _, _ = pretruncate_source(
            tokenizer, str(row["text"]), maximum_length=maximum_length,
            trigger_overhead=reserved_overhead,
        )
        sources.append(source)
    local_manifest = BoundaryManifest([
        BoundaryRecord(
            role, str(row["text_id"]), 0,
            fixed_random_boundary(str(row["text_id"]), source, seed=random_seed, replicate=0),
        )
        for row, source in zip(records, sources)
    ])
    for row, source in zip(records, sources):
        value = insert_once_with_span(
            source, wrapper, position, role=role, text_id=str(row["text_id"]),
            manifest=local_manifest, replicate=0,
        )[0]
        if len(tokenizer(value, add_special_tokens=True, truncation=False)["input_ids"]) > maximum_length:
            raise ProtocolViolation("semantic wrapper exceeds model maximum length")
        texts.append(value)
    return texts


def discovery_shard(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output)
    candidates = list(map(int, json.loads((output / "funnel" / "stability" / "selected.json").read_text(encoding="utf-8"))["token_ids"]))
    candidates = [value for index, value in enumerate(candidates) if index % int(args.shards) == int(args.shard)]
    metadata = [SemanticMetadata(**row) for row in read_jsonl(output / "semantic" / "token_metadata.jsonl")]
    by_meta = {row.token_id: row for row in metadata}; legal = {row.token_id: row for row in load_legal(output)}
    models = _full_models(output); records = load_role(output, "semantic_control"); manifest = load_manifest(output, ("semantic_control",))
    clean = load_embedding_cache(output / "base_embeddings" / "semantic_control.npy", expected_role="semantic_control", expected_records_hash=records_sha256(records))
    oracle = V62FinalOracle(config, output=output, device=args.device, phase="semantic_discovery", track="top100_matched_controls")
    rows: list[dict[str, Any]] = []
    count = int(config["semantic_controls"]["controls_per_candidate"])
    maximum_length = int(config["model"]["maximum_sequence_length"])
    for token_id in candidates:
        available = [model for (current, _), model in models.items() if current == token_id]
        if not available:
            raise ProtocolViolation(f"full model missing for semantic candidate {token_id}")
        model = min(available, key=lambda item: (item.cap_count, float(np.max(item.radii))))
        candidate_views, _, _ = encode_audited_positions(
            oracle, records, token_id=token_id, token_text=legal[token_id].token_text,
            role="semantic_control", manifest=manifest, random_replicates=1,
            maximum_length=maximum_length, metadata={"semantic_candidate": token_id},
        )
        candidate_vectors = primary_position_vectors(candidate_views)
        candidate_coverage = _coverage(model, candidate_vectors)
        pool = matched_controls(by_meta[token_id], metadata, max(count * 3, count))
        control_rows = []
        for control in pool:
            if len(control_rows) >= count: break
            try:
                views, _, _ = encode_audited_positions(
                    oracle, records, token_id=control.token_id, token_text=legal[control.token_id].token_text,
                    role="semantic_control", manifest=manifest, random_replicates=1,
                    maximum_length=maximum_length,
                    metadata={"semantic_candidate": token_id, "semantic_control": control.token_id},
                )
            except CandidateRejectedTokenRealization:
                continue
            control_rows.append({"token_id": control.token_id, "coverage": _coverage(model, primary_position_vectors(views))})
        if len(control_rows) != count:
            raise ProtocolViolation(f"candidate {token_id} has only {len(control_rows)}/{count} realizable controls")
        wrappers = {}; wrapper_values = wrapper_counterfactuals(legal[token_id].token_text)
        wrapper_reserve = max(len(oracle.tokenizer.encode(value, add_special_tokens=False)) for value in wrapper_values.values())
        for name, wrapper in wrapper_values.items():
            values = []
            for position in ("prefix", "suffix", "random"):
                texts = wrapper_insertions(
                    oracle.tokenizer, records, wrapper, position, "semantic_control",
                    maximum_length, int(config["positions"]["random_seed"]), wrapper_reserve,
                )
                values.append(model.contains(oracle.encode(texts, metadata={"semantic_wrapper": name, "position": position})))
            wrappers[name] = float(np.mean([np.mean(value) for value in values]))
        q95 = float(np.quantile([row["coverage"] for row in control_rows], 0.95))
        triggered = np.concatenate(list(candidate_vectors.values()))
        token_direction = oracle.encode([legal[token_id].token_text], metadata={"semantic_token_direction": token_id})[0]
        additive = additive_semantic_residual(np.concatenate([np.asarray(clean)] * 3), triggered, token_direction)
        margin = candidate_coverage - q95
        rows.append({
            "token_id": token_id, "token_text": legal[token_id].token_text,
            "cap_count": model.cap_count, "candidate_coverage": candidate_coverage,
            "control_coverage_q95": q95, "semantic_anomaly": margin,
            "matched_controls": control_rows, "wrapper_coverages": wrappers,
            "additive_semantic_model": additive,
            "anomaly_supported": margin >= float(config["semantic_controls"]["minimum_coverage_over_control_q95"])
                and min(wrappers.values()) >= float(config["semantic_controls"]["minimum_wrapper_coverage"]),
        })
    target = output / "semantic" / f"shard_{int(args.shard):02d}"
    write_jsonl(target / "results.jsonl", rows)
    write_json(target / "COMPLETE.json", {"candidates": len(candidates), "results": len(rows), "raw_forward_texts": oracle.raw_forward_texts})


def merge_discovery(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = Path(args.output); rows = []
    for shard in range(int(args.shards)):
        target = output / "semantic" / f"shard_{shard:02d}"
        rows.extend(read_jsonl(target / "results.jsonl"))
    expected = set(map(int, json.loads((output / "funnel" / "stability" / "selected.json").read_text(encoding="utf-8"))["token_ids"]))
    if {int(row["token_id"]) for row in rows} != expected or len(rows) != len(expected):
        raise ProtocolViolation("semantic discovery did not cover the frozen top-100 candidate set")
    write_jsonl(output / "semantic" / "discovery_results.jsonl", rows)
    write_json(output / "semantic" / "COMPLETE.json", {
        "schema_version": "mode3-v6-2-semantic-discovery-v1", "candidates": len(rows),
        "anomaly_supported": sum(bool(row["anomaly_supported"]) for row in rows),
        "selection_feedback_allowed": True, "confirmation_role_encoded": False,
    })


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/v6_2_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6_2"); parser.add_argument("--device", default="cuda:0")
    sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("build-metadata")
    shard = sub.add_parser("discovery-shard"); shard.add_argument("--shard", type=int, required=True); shard.add_argument("--shards", type=int, required=True)
    merge = sub.add_parser("merge-discovery"); merge.add_argument("--shards", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv); config = load_config(Path(args.config))
    {"build-metadata": build_metadata, "discovery-shard": discovery_shard, "merge-discovery": merge_discovery}[args.command](args, config)
    return 0


if __name__ == "__main__": raise SystemExit(main())
