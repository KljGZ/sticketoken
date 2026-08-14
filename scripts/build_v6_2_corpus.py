#!/usr/bin/env python3
"""Extend the immutable V6 corpus to four genuine IID source families.

The five V6 source files are copied byte-for-byte.  Three independently
published, commit-pinned parquet assets add Wikipedia articles, SQuAD source
contexts, and arXiv abstracts.  Selection is deterministic and without
replacement over normalized complete source records.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from sticky_lab.mode3_v6.data import normalized_text


@dataclass(frozen=True)
class ExtraSource:
    name: str
    filename: str
    source_id: str
    domain: str
    text_type: str
    license: str
    target: int
    url: str
    revision: str
    render: Callable[[dict[str, object]], str]


EXTRAS = (
    ExtraSource(
        "wikipedia_20231101_en_00000", "wikipedia-train-00000-of-00041.parquet",
        "hf:wikimedia/wikipedia:20231101.en", "iid_wikipedia",
        "encyclopedia_article", "cc-by-sa-3.0-and-gfdl", 20_000,
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/ad5752b/20231101.en/train-00000-of-00041.parquet",
        "ad5752b", lambda row: f"{str(row.get('title','')).strip()}. {str(row.get('text','')).strip()}",
    ),
    ExtraSource(
        "squad_contexts", "squad-train-0000.parquet",
        "hf:rajpurkar/squad:plain_text", "iid_qa_contexts",
        "source_context_document", "cc-by-sa-4.0", 15_000,
        "https://huggingface.co/datasets/rajpurkar/squad/resolve/bd9801c87f06e5034c669cd0c7d5e94e2cb723e6/plain_text/train/0000.parquet",
        "bd9801c87f06e5034c669cd0c7d5e94e2cb723e6",
        lambda row: f"{str(row.get('title','')).strip()}. {str(row.get('context','')).strip()}",
    ),
    ExtraSource(
        "arxiv_abstracts_00000", "arxiv-document-train-00000-of-00015.parquet",
        "hf:ccdv/arxiv-summarization:document", "iid_scientific_abstracts",
        "scientific_abstract", "other-see-source-card", 10_000,
        "https://huggingface.co/datasets/ccdv/arxiv-summarization/resolve/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/document/train-00000-of-00015.parquet",
        "240aaf1a969b3f8cd0ade6986bfad0cd730ee288",
        lambda row: str(row.get("abstract", "")),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def download(source: ExtraSource, raw_root: Path) -> Path:
    target = raw_root / source.filename; target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file(): return target
    temporary = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(source.url) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    temporary.replace(target); return target


def copy_base(base: Path, output: Path, seen: set[str]) -> list[dict[str, object]]:
    manifest = json.loads((base / "corpus_manifest.json").read_text(encoding="utf-8")); rows = []
    for source in manifest["sources"]:
        relative = Path(source["output_relative_path"]); origin = base / relative; destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(origin, destination)
        if sha256(destination) != source["output_sha256"]: raise RuntimeError(f"base copy hash mismatch: {relative}")
        with destination.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle): seen.add(normalized_text(row["text"]))
        rows.append({**source, "inherited_from_manifest_sha256": sha256(base / "corpus_manifest.json")})
    return rows


def select(frame: pd.DataFrame, source: ExtraSource, seen: set[str]) -> list[tuple[str, str]]:
    candidates = {}
    for row in frame.to_dict(orient="records"):
        text = " ".join(source.render(row).replace("\x00", " ").split()).strip(); normalized = normalized_text(text)
        if normalized and normalized not in seen and normalized not in candidates: candidates[normalized] = text
    ordered = sorted(candidates.items(), key=lambda item: hashlib.sha256(item[0].encode()).digest())
    if len(ordered) < source.target: raise RuntimeError(f"{source.name}: {len(ordered)}/{source.target} unique records")
    result = ordered[: source.target]; seen.update(normalized for normalized, _ in result); return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--base-registered", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path); parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--download", action="store_true"); args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()): raise SystemExit(f"refusing non-empty output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True); seen: set[str] = set(); source_rows = copy_base(args.base_registered, args.output_root, seen)
    fields = ["text", "document_id", "source_id", "domain", "language", "text_type", "license"]
    for source in EXTRAS:
        raw = download(source, args.raw_root) if args.download else args.raw_root / source.filename
        if not raw.is_file(): raise FileNotFoundError(raw)
        frame = pd.read_parquet(raw); raw_rows = len(frame); selected = select(frame, source, seen); del frame
        destination = args.output_root / source.name / "sampled_sentence_pairs.csv"; destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for normalized, text in selected:
                digest = hashlib.sha256(normalized.encode()).hexdigest()
                writer.writerow({"text": text, "document_id": f"{source.source_id}:{digest}", "source_id": source.source_id, "domain": source.domain, "language": "en", "text_type": source.text_type, "license": source.license})
        source_rows.append({"name": source.name, "rows": len(selected), "raw_rows": raw_rows, "raw_relative_path": raw.name, "raw_bytes": raw.stat().st_size, "raw_sha256": sha256(raw), "output_relative_path": destination.relative_to(args.output_root).as_posix(), "output_bytes": destination.stat().st_size, "output_sha256": sha256(destination), "source_id": source.source_id, "domain": source.domain, "license": source.license, "url": source.url, "revision": source.revision, "selection": "lowest_sha256(normalized_complete_source_record), without replacement"})
    manifest = {"schema_version": "mode3-v6-2-corpus-v1", "builder_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(), "python": sys.version.replace("\n"," "), "pandas": pd.__version__, "row_count": sum(int(row["rows"]) for row in source_rows), "normalized_unique_count": len(seen), "iid_source_count": 4, "ood_source_count": 4, "document_identity": "dataset source plus SHA-256 of complete normalized source record", "sentence_as_document_fallback": False, "resampling": False, "sources": source_rows}
    target = args.output_root / "corpus_manifest.json"; target.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"manifest": str(target), "manifest_sha256": sha256(target), "rows": manifest["row_count"], "sources": len(source_rows)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
