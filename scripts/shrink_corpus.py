#!/usr/bin/env python3
"""Subsample an existing corpus to fit a smaller memory budget.

Why this exists rather than re-running `build_corpus.py`: the source shards are ~470 MB each and
were deleted after the original build, so rebuilding from scratch means re-downloading ~1.9 GB over
a link that averages 3 MB/s. Subsampling the corpus we already have takes seconds and produces an
identical result, because the original sampling was itself a deterministic hash of `query_id`.

The divisor must be a multiple of the original. The corpus was built at 1-in-16, so 1-in-80 is a
strict subset (80 = 16 x 5) — every query kept here was already present, and its passages and
relevance judgments carry over untouched. A divisor that was not a multiple would select queries
that were never downloaded, silently producing a corpus with missing qrels.

The trade this represents is measured, not assumed. From the precision-vs-size curve in
docs/BUILD_LOG.md, shrinking *improves* retrieval precision — MRR@10 roughly doubles going from
310k to 60k, because in this dataset additional passages belong to other queries and act purely as
distractors. What it costs is coverage: fewer questions have any answer in the corpus at all.

Usage
-----
    python scripts/shrink_corpus.py --in data/corpus --out data/corpus-small --sample-1-in 80
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("shrink_corpus")


def keep_query(query_id: int, sample_1_in: int) -> bool:
    """Identical hash to scripts/build_corpus.py — the subset property depends on it."""
    if sample_1_in <= 1:
        return True
    digest = hashlib.blake2b(str(query_id).encode("ascii"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % sample_1_in == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus-small"))
    ap.add_argument("--sample-1-in", type=int, default=80)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    src, out = args.src, args.out
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((src / "manifest.json").read_text())
    original = int(manifest.get("sample_1_in", 1))
    if args.sample_1_in % original != 0:
        log.error(
            "--sample-1-in %d is not a multiple of the corpus's own %d; the result would reference "
            "queries that were never downloaded", args.sample_1_in, original
        )
        return 2

    queries = pq.read_table(src / "queries.parquet")
    qrels = pq.read_table(src / "qrels.parquet")
    passages = pq.read_table(src / "passages.parquet")

    keep_qids = {
        int(q) for q in queries.column("query_id").to_pylist() if keep_query(int(q), args.sample_1_in)
    }
    log.info("queries: %d -> %d", queries.num_rows, len(keep_qids))

    qmask = [int(q) in keep_qids for q in queries.column("query_id").to_pylist()]
    rmask = [int(q) in keep_qids for q in qrels.column("query_id").to_pylist()]
    kept_queries = queries.filter(pa.array(qmask))
    kept_qrels = qrels.filter(pa.array(rmask))

    # A passage survives if ANY surviving query references it. Filtering on `source_query_id`
    # instead would drop passages that are relevant to a kept query but were first seen under a
    # dropped one — silently deleting ground truth.
    keep_pids = set(kept_qrels.column("passage_id").to_pylist())
    pmask = [p in keep_pids for p in passages.column("passage_id").to_pylist()]
    kept_passages = passages.filter(pa.array(pmask))

    log.info("qrels:    %d -> %d", qrels.num_rows, kept_qrels.num_rows)
    log.info("passages: %d -> %d", passages.num_rows, kept_passages.num_rows)

    pq.write_table(kept_passages, out / "passages.parquet", compression="zstd")
    pq.write_table(kept_queries, out / "queries.parquet", compression="zstd")
    pq.write_table(kept_qrels, out / "qrels.parquet", compression="zstd")

    langs: dict[str, int] = {}
    for lg in kept_passages.column("lang").to_pylist():
        langs[lg] = langs.get(lg, 0) + 1

    manifest["sample_1_in"] = args.sample_1_in
    manifest["derived_from"] = {"sample_1_in": original, "passages": passages.num_rows}
    manifest["stats"] = {
        **manifest.get("stats", {}),
        "passages_kept": kept_passages.num_rows,
        "per_lang_kept": langs,
        "queries_kept": kept_queries.num_rows,
    }
    manifest.pop("embeddings", None)
    manifest.pop("indexes", None)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    for lg, n in sorted(langs.items()):
        log.info("  %-3s %d", lg, n)
    log.info("wrote -> %s  (now run scripts/embed_corpus.py --corpus %s --quantize int8)", out, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
