"""Build preregistered semantic-matching metadata for all legal tokens."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .atomic_io import write_jsonl
from .oracle_whitebox import WhiteboxSentenceTransformer
from .semantic_controls import TokenMetadata


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _casing(text: str) -> str:
    if text.isupper() and text.lower() != text: return "upper"
    if text.istitle(): return "title"
    if text.islower() and text.upper() != text: return "lower"
    return "mixed_or_uncased"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    legal = _jsonl(output / "enumeration" / "legal_unrestricted.jsonl")
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    import spacy
    from nltk.corpus import wordnet as wn

    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        wn.ensure_loaded()
    except Exception as error:
        raise RuntimeError("formal semantic matching requires registered spaCy POS and WordNet resources") from error
    model = config["model"]
    runtime = SentenceTransformer(model["local_path"] or model["id"], revision=None if model["local_path"] else model["revision"], device=args.device, trust_remote_code=model["trust_remote_code"])
    tokenizer = AutoTokenizer.from_pretrained(model["local_path"] or model["id"], revision=None if model["local_path"] else model["revision"], trust_remote_code=model["trust_remote_code"])
    matrix = WhiteboxSentenceTransformer(runtime).embedding_matrix()
    frequency: Counter[int] = Counter()
    document_frequency: Counter[int] = Counter()
    role_dir = output / "registration" / "roles"
    corpus_paths = [role_dir / "screen_fit.jsonl", role_dir / "full_search_fit.jsonl", role_dir / "semantic_control.jsonl"]
    documents = 0
    for path in corpus_paths:
        for row in _jsonl(path):
            ids = list(map(int, tokenizer.encode(row["text"], add_special_tokens=False)))
            frequency.update(ids); document_frequency.update(set(ids)); documents += 1
    maximum_log_frequency = max((math.log1p(value) for value in frequency.values()), default=1.0)
    rows = []
    for row in legal:
        token_id, text = int(row["token_id"]), str(row["token_text"])
        stripped = text.strip()
        doc = nlp(stripped or text)
        pos = doc[0].pos_ if len(doc) else "X"
        synsets = wn.synsets(stripped.replace(" ", "_")) if stripped else []
        category = synsets[0].lexname() if synsets else "no_wordnet_synset"
        meta = TokenMetadata(
            token_id, float(frequency[token_id]),
            float(math.log((documents + 1) / (document_frequency[token_id] + 1)) + 1),
            pos, category, len(text), _casing(text), float(np.linalg.norm(matrix[token_id])),
            float(math.log1p(frequency[token_id]) / maximum_log_frequency),
        )
        rows.append(meta.__dict__)
    write_jsonl(output / "semantic" / "token_metadata.jsonl", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
