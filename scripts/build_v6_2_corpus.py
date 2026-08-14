#!/usr/bin/env python3
"""Extend V6 to four genuine IID sources from immutable local raw snapshots.

Five V6 CSVs are copied byte-for-byte.  Three existing BEIR-style JSONL
corpora (MS MARCO, HotpotQA and Natural Questions) add independent dataset
source IDs.  Their complete raw SHA-256 hashes are recorded in the manifest.
Selection is streaming, deterministic, document-level and without replacement.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from sticky_lab.mode3_v6.data import normalized_text


@dataclass(frozen=True)
class ExtraSource:
    name: str
    relative_path: str
    source_id: str
    domain: str
    text_type: str
    license: str
    target: int
    provenance_url: str


EXTRAS = (
    ExtraSource("msmarco_passages", "msmarco/corpus.jsonl", "beir:msmarco", "iid_web_passages", "web_passage", "microsoft-research-license-see-source", 20_000, "https://microsoft.github.io/msmarco/"),
    ExtraSource("hotpotqa_contexts", "hotpotqa/corpus.jsonl", "beir:hotpotqa", "iid_multi_hop_contexts", "wikipedia_context_document", "cc-by-sa-4.0-see-source", 20_000, "https://hotpotqa.github.io/"),
    ExtraSource("natural_questions_contexts", "nq/corpus.jsonl", "beir:nq", "iid_natural_questions_contexts", "wikipedia_context_document", "cc-by-sa-3.0-see-source", 20_000, "https://ai.google.com/research/NaturalQuestions"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def copy_base(base: Path, output: Path, seen: set[str]) -> list[dict[str, Any]]:
    base_manifest = base / "corpus_manifest.json"; manifest = json.loads(base_manifest.read_text(encoding="utf-8")); rows = []
    for source in manifest["sources"]:
        relative = Path(source["output_relative_path"]); origin = base / relative; destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(origin, destination)
        if sha256(destination) != source["output_sha256"]: raise RuntimeError(f"base copy hash mismatch: {relative}")
        with destination.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle): seen.add(normalized_text(row["text"]))
        rows.append({**source, "inherited_from_manifest_sha256": sha256(base_manifest)})
    return rows


def _render(row: dict[str, Any]) -> str:
    title = str(row.get("title", "")).strip(); text = str(row.get("text", "")).strip()
    return " ".join((f"{title}. {text}" if title else text).replace("\x00", " ").split()).strip()


def select_jsonl(path: Path, target: int, global_seen: set[str]) -> tuple[list[tuple[str, str, str]], int, int, str]:
    """Keep the lowest normalized-content hashes using O(target) memory."""
    try:
        import orjson
        loads = orjson.loads
    except ImportError:
        loads = json.loads
    heap: list[tuple[int, str, str, str]] = []; selected_norms: set[str] = set(); raw_rows = 0; invalid = 0
    raw_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            raw_digest.update(line)
            if not line.strip(): continue
            raw_rows += 1; row = loads(line); text = _render(row); norm = normalized_text(text)
            if not norm or norm in global_seen: invalid += 1; continue
            digest = int.from_bytes(hashlib.sha256(norm.encode("utf-8")).digest(), "big")
            if norm in selected_norms: continue
            item = (-digest, norm, text, str(row.get("_id", line_number)))
            if len(heap) < target:
                heapq.heappush(heap, item); selected_norms.add(norm)
            elif digest < -heap[0][0]:
                removed = heapq.heapreplace(heap, item); selected_norms.remove(removed[1]); selected_norms.add(norm)
    if len(heap) < target: raise RuntimeError(f"{path}: {len(heap)}/{target} unique eligible documents")
    result = sorted([(norm, text, identity) for _, norm, text, identity in heap], key=lambda value: hashlib.sha256(value[0].encode()).digest())
    global_seen.update(norm for norm, _, _ in result); return result, raw_rows, invalid, raw_digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--base-registered", required=True, type=Path)
    parser.add_argument("--local-corpus-root", required=True, type=Path); parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()): raise SystemExit(f"refusing non-empty output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True); seen: set[str] = set(); source_rows = copy_base(args.base_registered, args.output_root, seen)
    fields = ["text", "document_id", "source_id", "domain", "language", "text_type", "license"]
    for source in EXTRAS:
        raw = args.local_corpus_root / source.relative_path
        if not raw.is_file(): raise FileNotFoundError(raw)
        selected, raw_rows, invalid, raw_sha256 = select_jsonl(raw, source.target, seen)
        destination = args.output_root / source.name / "sampled_sentence_pairs.csv"; destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for norm, text, identity in selected:
                digest = hashlib.sha256(norm.encode()).hexdigest()
                writer.writerow({"text": text, "document_id": f"{source.source_id}:{identity}:{digest}", "source_id": source.source_id, "domain": source.domain, "language": "en", "text_type": source.text_type, "license": source.license})
        source_rows.append({"name": source.name, "rows": len(selected), "raw_rows": raw_rows, "raw_cross_source_or_invalid": invalid, "raw_absolute_path": str(raw), "raw_bytes": raw.stat().st_size, "raw_sha256": raw_sha256, "output_relative_path": destination.relative_to(args.output_root).as_posix(), "output_bytes": destination.stat().st_size, "output_sha256": sha256(destination), "source_id": source.source_id, "domain": source.domain, "license": source.license, "provenance_url": source.provenance_url, "revision": f"local-raw-sha256:{raw_sha256}", "selection": "streaming lowest_sha256(normalized_complete_document), without replacement"})
    manifest = {"schema_version": "mode3-v6-2-corpus-v1", "builder_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(), "python": sys.version.replace("\n"," "), "row_count": sum(int(row["rows"]) for row in source_rows), "normalized_unique_count": len(seen), "iid_source_count": 4, "ood_source_count": 4, "document_identity": "dataset source, original ID, and SHA-256 of complete normalized source record", "sentence_as_document_fallback": False, "resampling": False, "sources": source_rows}
    target = args.output_root / "corpus_manifest.json"; target.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"manifest": str(target), "manifest_sha256": sha256(target), "rows": manifest["row_count"], "sources": len(source_rows)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
