"""Build the immutable V6 canonical corpus from pinned public Parquet assets.

This is a data-registration utility, not an experiment sampler.  Every output
row is a distinct source document selected without replacement by a stable
content hash.  The raw assets and the resulting CSV files are inventoried with
SHA-256 so the external data volume can be reconstructed independently.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

import pandas as pd

from .data import normalized_text


@dataclass(frozen=True)
class SourceSpec:
    name: str
    relative_path: str
    source_id: str
    domain: str
    language: str
    text_type: str
    license: str
    target: int
    url: str
    revision: str
    render: Callable[[dict[str, object]], str]


def _join_title(row: dict[str, object]) -> str:
    title = str(row.get("title", "")).strip()
    content = str(row.get("content", "")).strip()
    return f"{title}. {content}" if title else content


SOURCES = (
    SourceSpec(
        "dbpedia14", "dbpedia14/train-0000.parquet", "hf:fancyzhx/dbpedia_14",
        "iid_knowledge_base", "en", "knowledge_base_abstract", "cc-by-sa-3.0", 550_000,
        "https://huggingface.co/datasets/fancyzhx/dbpedia_14/resolve/refs%2Fconvert%2Fparquet/dbpedia_14/train/0000.parquet",
        "7b2d3998c3e87668c1282ce9b8bfd164a4fca8f5", _join_title,
    ),
    SourceSpec(
        "ag_news", "ag_news/train-0000.parquet", "hf:fancyzhx/ag_news",
        "ood_news", "en", "news_article", "license-not-stated-on-source-card", 30_000,
        "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "fc456aa3b280761d8c49225b0d9e4adae901963b", lambda row: str(row.get("text", "")),
    ),
    SourceSpec(
        "imdb", "imdb/unsupervised-0000.parquet", "hf:stanfordnlp/imdb",
        "ood_movie_reviews", "en", "movie_review", "other-see-source-card", 30_000,
        "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/refs%2Fconvert%2Fparquet/plain_text/unsupervised/0000.parquet",
        "b8beaaa4c7f38e9e597697c92c30f9ccb866552d", lambda row: str(row.get("text", "")),
    ),
    SourceSpec(
        "yelp_review_full", "yelp_review_full/test-0000.parquet", "hf:Yelp/yelp_review_full",
        "ood_business_reviews", "en", "business_review", "other-yelp-license-see-source-card", 30_000,
        "https://huggingface.co/datasets/Yelp/yelp_review_full/resolve/refs%2Fconvert%2Fparquet/yelp_review_full/test/0000.parquet",
        "adeddac038505c0178d827a06aeb48ac131e1000", lambda row: str(row.get("text", "")),
    ),
    SourceSpec(
        "amazon_polarity", "amazon_polarity/test-0000.parquet", "hf:fancyzhx/amazon_polarity",
        "ood_product_reviews", "en", "product_review", "apache-2.0", 30_000,
        "https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/refs%2Fconvert%2Fparquet/amazon_polarity/test/0000.parquet",
        "c63175c691dc8edb840a88886f26dfebc747b902", _join_title,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select(frame: pd.DataFrame, spec: SourceSpec, global_seen: set[str]) -> list[tuple[str, str]]:
    candidates: dict[str, str] = {}
    for row in frame.to_dict(orient="records"):
        text = " ".join(spec.render(row).replace("\x00", " ").split()).strip()
        norm = normalized_text(text)
        if not norm or norm in global_seen or norm in candidates:
            continue
        candidates[norm] = text
    ordered = sorted(candidates.items(), key=lambda item: hashlib.sha256(item[0].encode("utf-8")).digest())
    if len(ordered) < spec.target:
        raise RuntimeError(f"{spec.name}: only {len(ordered)} unique documents, need {spec.target}")
    selected = ordered[: spec.target]
    global_seen.update(norm for norm, _ in selected)
    return selected


def build(raw_root: Path, output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    global_seen: set[str] = set()
    source_records: list[dict[str, object]] = []
    total = 0
    fields = ["text", "document_id", "source_id", "domain", "language", "text_type", "license"]
    for spec in SOURCES:
        raw = raw_root / spec.relative_path
        if not raw.is_file():
            raise FileNotFoundError(raw)
        frame = pd.read_parquet(raw)
        raw_rows = len(frame)
        selected = _select(frame, spec, global_seen)
        del frame
        destination = output_root / spec.name / "sampled_sentence_pairs.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for norm, text in selected:
                digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
                writer.writerow({
                    "text": text,
                    "document_id": f"{spec.source_id}:{digest}",
                    "source_id": spec.source_id,
                    "domain": spec.domain,
                    "language": spec.language,
                    "text_type": spec.text_type,
                    "license": spec.license,
                })
        temporary.replace(destination)
        total += len(selected)
        source_records.append({
            "name": spec.name, "rows": len(selected), "raw_rows": raw_rows,
            "raw_relative_path": spec.relative_path, "raw_bytes": raw.stat().st_size,
            "raw_sha256": _sha256(raw), "output_relative_path": destination.relative_to(output_root).as_posix(),
            "output_bytes": destination.stat().st_size, "output_sha256": _sha256(destination),
            "source_id": spec.source_id, "domain": spec.domain, "license": spec.license,
            "url": spec.url, "revision": spec.revision, "selection": "lowest_sha256(normalized_full_document), without replacement",
        })
    manifest = {
        "schema_version": 1,
        "builder_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip(),
        "python": sys.version.replace("\n", " "),
        "pandas": pd.__version__,
        "row_count": total,
        "normalized_unique_count": len(global_seen),
        "document_identity": "dataset source plus SHA-256 of the complete normalized source record",
        "sentence_as_document_fallback": False,
        "resampling": False,
        "sources": source_records,
    }
    target = output_root / "corpus_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(Path(args.raw_root), Path(args.output_root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
