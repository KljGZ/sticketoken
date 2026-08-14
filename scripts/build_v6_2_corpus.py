#!/usr/bin/env python3
"""Build the immutable eight-source V6.2 corpus from local raw snapshots.

The registered V6 DBpedia file is inherited byte-for-byte. Three independent
BEIR corpora add genuine IID sources. The four V6 OOD sources are rebuilt from
their complete, manifest-bound Parquet snapshots because the historical 30k
subsets cannot satisfy V6.2's disjoint 15k-trigger/30k-benign contract.
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
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sticky_lab.mode3_v6.data import normalized_text


@dataclass(frozen=True)
class JsonSource:
    name: str
    relative_path: str
    source_id: str
    domain: str
    text_type: str
    license: str
    target: int
    provenance_url: str


@dataclass(frozen=True)
class ParquetSource:
    name: str
    relative_path: str
    source_id: str
    domain: str
    text_type: str
    license: str
    provenance_url: str
    revision: str
    renderer: str


IID_SOURCES = (
    JsonSource("msmarco_passages", "msmarco/corpus.jsonl", "beir:msmarco", "iid_web_passages", "web_passage", "microsoft-research-license-see-source", 100_000, "https://microsoft.github.io/msmarco/"),
    JsonSource("hotpotqa_contexts", "hotpotqa/corpus.jsonl", "beir:hotpotqa", "iid_multi_hop_contexts", "wikipedia_context_document", "cc-by-sa-4.0-see-source", 100_000, "https://hotpotqa.github.io/"),
    JsonSource("natural_questions_contexts", "nq/corpus.jsonl", "beir:nq", "iid_natural_questions_contexts", "wikipedia_context_document", "cc-by-sa-3.0-see-source", 100_000, "https://ai.google.com/research/NaturalQuestions"),
)

OOD_SOURCES = (
    ParquetSource("ag_news", "ag_news/train-0000.parquet", "hf:fancyzhx/ag_news", "ood_news", "news_article", "license-not-stated-on-source-card", "https://huggingface.co/datasets/fancyzhx/ag_news", "fc456aa3b280761d8c49225b0d9e4adae901963b", "text"),
    ParquetSource("imdb", "imdb/unsupervised-0000.parquet", "hf:stanfordnlp/imdb", "ood_movie_reviews", "movie_review", "other-see-source-card", "https://huggingface.co/datasets/stanfordnlp/imdb", "b8beaaa4c7f38e9e597697c92c30f9ccb866552d", "text"),
    ParquetSource("yelp_review_full", "yelp_review_full/test-0000.parquet", "hf:Yelp/yelp_review_full", "ood_business_reviews", "business_review", "other-yelp-license-see-source-card", "https://huggingface.co/datasets/Yelp/yelp_review_full", "adeddac038505c0178d827a06aeb48ac131e1000", "text"),
    ParquetSource("amazon_polarity", "amazon_polarity/test-0000.parquet", "hf:fancyzhx/amazon_polarity", "ood_product_reviews", "product_review", "apache-2.0", "https://huggingface.co/datasets/fancyzhx/amazon_polarity", "c63175c691dc8edb840a88886f26dfebc747b902", "title_content"),
)

FIELDS = ["text", "document_id", "source_id", "domain", "language", "text_type", "license"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    clean = not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if not clean:
        raise RuntimeError("corpus construction requires a clean, committed worktree")
    return commit, clean


def inherit_dbpedia(base: Path, output: Path, seen: set[str]) -> list[dict[str, Any]]:
    base_manifest = base / "corpus_manifest.json"
    manifest = json.loads(base_manifest.read_text(encoding="utf-8"))
    selected = [row for row in manifest["sources"] if row["name"] == "dbpedia14"]
    if len(selected) != 1:
        raise RuntimeError("base manifest must contain exactly one dbpedia14 source")
    source = selected[0]
    relative = Path(source["output_relative_path"])
    origin = base / relative
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origin, destination)
    if sha256(destination) != source["output_sha256"]:
        raise RuntimeError(f"base copy hash mismatch: {relative}")
    with destination.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seen.add(normalized_text(row["text"]))
    return [{**source, "inherited_from_manifest_sha256": sha256(base_manifest)}]


def _render_json(row: dict[str, Any]) -> str:
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    return " ".join((f"{title}. {text}" if title else text).replace("\x00", " ").split()).strip()


def select_jsonl(path: Path, target: int, global_seen: set[str]) -> tuple[list[tuple[str, str, str]], int, int, str]:
    """Keep the lowest normalized-content hashes using O(target) memory."""
    try:
        import orjson
        loads: Callable[[bytes], dict[str, Any]] = orjson.loads
    except ImportError:
        loads = json.loads
    heap: list[tuple[int, str, str, str]] = []
    selected_norms: set[str] = set()
    raw_rows = invalid = 0
    raw_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            raw_digest.update(line)
            if not line.strip():
                continue
            raw_rows += 1
            row = loads(line)
            text = _render_json(row)
            norm = normalized_text(text)
            if not norm or norm in global_seen:
                invalid += 1
                continue
            digest = int.from_bytes(hashlib.sha256(norm.encode("utf-8")).digest(), "big")
            if norm in selected_norms:
                continue
            item = (-digest, norm, text, str(row.get("_id", line_number)))
            if len(heap) < target:
                heapq.heappush(heap, item)
                selected_norms.add(norm)
            elif digest < -heap[0][0]:
                removed = heapq.heapreplace(heap, item)
                selected_norms.remove(removed[1])
                selected_norms.add(norm)
    if len(heap) < target:
        raise RuntimeError(f"{path}: {len(heap)}/{target} unique eligible documents")
    result = sorted(
        [(norm, text, identity) for _, norm, text, identity in heap],
        key=lambda value: hashlib.sha256(value[0].encode()).digest(),
    )
    global_seen.update(norm for norm, _, _ in result)
    return result, raw_rows, invalid, raw_digest.hexdigest()


def write_iid(source: JsonSource, raw: Path, output: Path, seen: set[str]) -> dict[str, Any]:
    selected, raw_rows, invalid, raw_sha = select_jsonl(raw, source.target, seen)
    destination = output / source.name / "sampled_sentence_pairs.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for norm, text, identity in selected:
            digest = hashlib.sha256(norm.encode()).hexdigest()
            writer.writerow({"text": text, "document_id": f"{source.source_id}:{identity}:{digest}", "source_id": source.source_id, "domain": source.domain, "language": "en", "text_type": source.text_type, "license": source.license})
    return {"name": source.name, "rows": len(selected), "raw_rows": raw_rows, "raw_cross_source_or_invalid": invalid, "raw_absolute_path": str(raw), "raw_bytes": raw.stat().st_size, "raw_sha256": raw_sha, "output_relative_path": destination.relative_to(output).as_posix(), "output_bytes": destination.stat().st_size, "output_sha256": sha256(destination), "source_id": source.source_id, "domain": source.domain, "license": source.license, "provenance_url": source.provenance_url, "revision": f"local-raw-sha256:{raw_sha}", "selection": f"lowest {source.target} SHA-256(normalized complete document), without replacement"}


def parquet_records(path: Path, renderer: str) -> Iterator[tuple[int, str]]:
    import pyarrow.parquet as parquet

    index = 0
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=8192):
        columns = batch.to_pydict()
        rows = batch.num_rows
        for offset in range(rows):
            if renderer == "text":
                raw = str(columns["text"][offset] or "")
            elif renderer == "title_content":
                title = str(columns["title"][offset] or "").strip()
                content = str(columns["content"][offset] or "").strip()
                raw = f"{title}. {content}" if title else content
            else:  # pragma: no cover - constant registry
                raise RuntimeError(f"unknown Parquet renderer: {renderer}")
            yield index, " ".join(raw.replace("\x00", " ").split()).strip()
            index += 1


def write_ood(source: ParquetSource, raw: Path, output: Path, seen: set[str]) -> dict[str, Any]:
    raw_sha = sha256(raw)
    destination = output / source.name / "sampled_sentence_pairs.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = invalid = rows = 0
    local_seen: set[str] = set()
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row_index, text in parquet_records(raw, source.renderer):
            raw_rows += 1
            norm = normalized_text(text)
            if not norm or norm in seen or norm in local_seen:
                invalid += 1
                continue
            local_seen.add(norm)
            digest = hashlib.sha256(norm.encode()).hexdigest()
            writer.writerow({"text": text, "document_id": f"{source.source_id}:{row_index}:{digest}", "source_id": source.source_id, "domain": source.domain, "language": "en", "text_type": source.text_type, "license": source.license})
            rows += 1
    seen.update(local_seen)
    if rows < 45_000:
        raise RuntimeError(f"{source.name}: only {rows} unique rows; V6.2 requires 45,000")
    return {"name": source.name, "rows": rows, "raw_rows": raw_rows, "raw_cross_source_or_invalid": invalid, "raw_absolute_path": str(raw), "raw_bytes": raw.stat().st_size, "raw_sha256": raw_sha, "output_relative_path": destination.relative_to(output).as_posix(), "output_bytes": destination.stat().st_size, "output_sha256": sha256(destination), "source_id": source.source_id, "domain": source.domain, "license": source.license, "provenance_url": source.provenance_url, "revision": source.revision, "selection": "all unique eligible complete documents in immutable Parquet row order"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-registered", required=True, type=Path)
    parser.add_argument("--local-corpus-root", required=True, type=Path)
    parser.add_argument("--ood-raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing non-empty output: {args.output_root}")
    commit, clean = git_metadata()
    args.output_root.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    sources = inherit_dbpedia(args.base_registered, args.output_root, seen)
    for source in IID_SOURCES:
        raw = args.local_corpus_root / source.relative_path
        if not raw.is_file():
            raise FileNotFoundError(raw)
        print(f"building IID source {source.name} from {raw}", flush=True)
        sources.append(write_iid(source, raw, args.output_root, seen))
    for source in OOD_SOURCES:
        raw = args.ood_raw_root / source.relative_path
        if not raw.is_file():
            raise FileNotFoundError(raw)
        print(f"building OOD source {source.name} from {raw}", flush=True)
        sources.append(write_ood(source, raw, args.output_root, seen))
    manifest = {
        "schema_version": "mode3-v6-2-corpus-v2",
        "builder_commit": commit,
        "builder_script_sha256": sha256(Path(__file__)),
        "builder_worktree_clean": clean,
        "python": sys.version.replace("\n", " "),
        "row_count": sum(int(row["rows"]) for row in sources),
        "normalized_unique_count": len(seen),
        "iid_source_count": 4,
        "ood_source_count": 4,
        "document_identity": "dataset source, original ID or row index, and SHA-256 of complete normalized source record",
        "sentence_as_document_fallback": False,
        "resampling": False,
        "sources": sources,
    }
    target = args.output_root / "corpus_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(target), "manifest_sha256": sha256(target), "rows": manifest["row_count"], "sources": len(sources)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
