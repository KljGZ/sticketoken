"""Run controlled retrieval only after all encoder-level gates pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .atomic_io import write_json
from .experiment import encode_all_positions, encode_clean, position_balanced_concat
from .insertion import BoundaryManifest, BoundaryRecord, insert_once
from .oracle_blackbox import SentenceTransformerFinalOracle
from .retrieval import single_poison_retrieval


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_mode3.yaml")
    parser.add_argument("--output", default="results/sticky_lab/sentence_t5_base/mode3_v6")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--payload-output", required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    frozen = json.loads((output / "validation" / "frozen_cap.json").read_text())
    required_evidence = [output / "test" / "result.json", output / "semantic-controls" / "result.json"]
    if not frozen.get("gate_open") or not all(path.exists() for path in required_evidence):
        raise RuntimeError("core, migration, and anomaly evidence must complete before retrieval")
    test = json.loads(required_evidence[0].read_text())
    semantic = json.loads(required_evidence[1].read_text())
    if not test.get("certified") or not semantic.get("anomaly_supported"):
        raise RuntimeError("retrieval gate is closed")
    roles = output / "registration" / "roles"
    queries = _jsonl(roles / "iid_test.jsonl")
    keys = _jsonl(roles / "retrieval_probe.jsonl")
    boundaries = BoundaryManifest([BoundaryRecord(**row) for row in _jsonl(output / "registration" / "random_boundaries.jsonl")])
    token = str(frozen["token_text"])
    model = config["model"]
    oracle = SentenceTransformerFinalOracle(model["id"], revision=model["revision"], device=args.device, batch_size=model["batch_size"], local_path=model["local_path"], cache_folder=model["cache_folder"], trust_remote_code=model["trust_remote_code"])
    views = encode_all_positions(oracle, queries, token, role="iid_test", manifest=boundaries, random_replicates=config["positions"]["random_replicates"])
    query_vectors = position_balanced_concat(views)
    benign_keys = encode_clean(oracle, keys)
    poison_record = keys[0]
    poison_text = insert_once(poison_record["text"], token, "prefix", role="retrieval_probe", text_id=poison_record["text_id"], manifest=boundaries)
    poison_key = oracle.encode([poison_text])[0]
    result = single_poison_retrieval(query_vectors, benign_keys, poison_key)
    write_json(Path(args.payload_output), {
        "real_poison_text_id": poison_record["text_id"], "single_poison_entry": True,
        "poison_top1_rate": result.poison_top1_rate, "poison_top5_rate": result.poison_top5_rate,
        "poison_rank": result.poison_rank.tolist(), "poison_similarity": result.poison_similarity.tolist(),
        "strongest_benign_similarity": result.strongest_benign_similarity.tolist(),
        "search_feedback": False, "refit_performed": False, "query_ledger": oracle.ledger.to_dict(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
