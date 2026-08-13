#!/usr/bin/env python3
"""Build the SHRUTI corpus from ai4bharat/MSMARCO-XI.

Design constraints that shaped this script
------------------------------------------

**Disk.** The full dataset is ~56 GB and the build machine has ~16 GB free. Train shards are
~3.7 GB each; validation shards are ~0.47 GB and carry the same `is_selected` relevance labels.
So we use validation, download exactly one shard at a time, process it by row group, and delete it
before fetching the next. Free space is checked before every download and the run aborts rather
than filling the disk — a wedged machine two days before a deadline is not an acceptable failure
mode.

**Memory.** Passages are written incrementally through a `ParquetWriter` rather than accumulated
in a list. Holding ~500k multilingual strings in Python would cost on the order of a gigabyte;
flushing in batches keeps the resident set flat and roughly independent of corpus size.

**Structure.** Output follows the standard IR triple rather than one denormalised table:

    passages.parquet   the corpus            passage_id, text, lang, ...
    queries.parquet    the topics            query_id, lang, query, answer, query_type
    qrels.parquet      the relevance labels  query_id, lang, passage_id, is_selected

This is what makes the chunking lab possible: MRR@10 and Recall@k need qrels as a first-class
relation, not a boolean smuggled into a passage row.

**Dedupe.** `query_id` is stable across language files, so every language shard repeats the same
`English_passages`. Content-hash dedupe collapses them, and the observed dedupe rate is logged —
which empirically confirms the shards really do align rather than assuming it.

Usage
-----
    python scripts/build_corpus.py --languages hi gu bn ta --out data/corpus
    python scripts/build_corpus.py --languages hi --limit-queries 500 --out data/corpus-smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import time
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "ai4bharat/MSMARCO-XI"
REPO_TYPE = "dataset"

# Actual filenames in the repo tree. The dataset README lists slightly different stems
# ("guval", "orval"); the tree is authoritative and these were read from it directly.
LANG_FILES: dict[str, str] = {
    "as": "asmval.parquet",
    "bn": "benval.parquet",
    "gu": "gujval.parquet",
    "hi": "hinval.parquet",
    "kn": "kanval.parquet",
    "ml": "malval.parquet",
    "mr": "marval.parquet",
    "ne": "nepval.parquet",
    "or": "orival.parquet",
    "pa": "panval.parquet",
    "sa": "sanval.parquet",
    "ta": "tamval.parquet",
    "te": "telval.parquet",
    "ur": "urdval.parquet",
}

# Abort if free disk would drop below this. Sized to leave the OS room to breathe.
MIN_FREE_BYTES = 5 * 1024**3
# A shard is ~0.5 GB; require comfortable headroom before starting one.
SHARD_HEADROOM_BYTES = 2 * 1024**3

MIN_PASSAGE_CHARS = 20
MAX_PASSAGE_CHARS = 4000
FLUSH_EVERY = 20_000

# Translation-degeneration filter.
#
# MSMARCO-XI is machine-translated, and neural MT degenerates on short or ambiguous inputs by
# looping a phrase. Measured on a 400-query Hindi sample: the English query "suit definition"
# (15 chars) became a 7,783-character Hindi string repeating one clause ~90 times.
#
# Prevalence, measured rather than assumed:
#
#   threshold   passages        queries
#   > 0.3       4.22%           0.25%
#   > 0.5       0.11%           0.25%
#   > 0.7       0.00%           0.25%
#
# 0.5 is the chosen cut. At 0.3 the filter starts eating legitimate repetitive text — enumerations,
# tables, boilerplate — which is why the looser threshold flags 40× more passages while catching no
# additional queries. Degenerate text is poison for retrieval specifically: repeated n-grams inflate
# BM25 term frequencies and drag the embedding toward the repeated phrase, so these passages
# retrieve for far more queries than they deserve.
MAX_REPETITION_RATIO = 0.5
_REP_NGRAM = 5


def repetition_ratio(text: str, n: int = _REP_NGRAM) -> float:
    """Fraction of word n-grams that are repeats. 0.0 = all distinct, ~1.0 = a loop."""
    tokens = text.split()
    if len(tokens) < 2 * n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)

_WS = re.compile(r"\s+")

log = logging.getLogger("build_corpus")


# ---------------------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------------------


def normalize(text: str) -> str:
    """NFC-normalise and collapse whitespace.

    NFC matters for Indic scripts: the same visible Devanagari or Gujarati grapheme can arrive as
    either a precomposed codepoint or a base plus combining mark. Without normalisation those hash
    differently, so dedupe would miss real duplicates and BM25 would tokenise the same word two
    ways.
    """
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def content_id(text: str) -> str:
    """Stable 16-hex-char id derived from content. Doubles as the dedupe key."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def keep_query(query_id: int, sample_1_in: int) -> bool:
    """Deterministic query subsampling, stable across language shards.

    The validation split is far larger than its file size suggests: 97,941 queries per language,
    ~19.4 passages each, which is ~1.9M passages from a single shard and ~4.75M across four
    languages. That is 10-20x more than this system should index — it would mean ~4.9 GB of float32
    embeddings, a multi-gigabyte HNSW graph, and exact search in the 50-100ms range, which spends
    most of the answer budget on a corpus size nobody asked for.

    So we sample. Two properties matter, and neither is satisfied by simply taking the first N rows:

    1. **Unbiased.** Parquet row order is not random — it may correlate with query id, source, or
       collection date. Hashing decorrelates the selection from whatever structure the file has.
    2. **Identical across languages.** The hash depends only on `query_id`, which is stable across
       shards, so every language contributes translations of *the same* queries. This is what makes
       cross-lingual comparison meaningful, and it is what lets the English passages — repeated
       identically in every shard — collapse cleanly under content-hash dedupe instead of
       fragmenting into near-duplicates of different query subsets.
    """
    if sample_1_in <= 1:
        return True
    digest = hashlib.blake2b(str(query_id).encode("ascii"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % sample_1_in == 0


# ---------------------------------------------------------------------------------------
# Disk safety
# ---------------------------------------------------------------------------------------


class DiskGuardError(RuntimeError):
    pass


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def assert_disk_headroom(path: Path, need: int, what: str) -> None:
    free = free_bytes(path)
    if free < need:
        raise DiskGuardError(
            f"refusing to {what}: {free / 1024**3:.1f} GB free, "
            f"need {need / 1024**3:.1f} GB. Free space and retry."
        )


# ---------------------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------------------

PASSAGE_SCHEMA = pa.schema(
    [
        ("passage_id", pa.string()),
        ("text", pa.string()),
        ("lang", pa.string()),
        ("query_type", pa.string()),
        ("source_query_id", pa.int64()),
        ("position", pa.int32()),
        ("n_chars", pa.int32()),
        ("n_words", pa.int32()),
    ]
)

QUERY_SCHEMA = pa.schema(
    [
        ("query_id", pa.int64()),
        ("lang", pa.string()),
        ("query", pa.string()),
        ("answer", pa.string()),
        ("query_type", pa.string()),
        ("eng_query", pa.string()),
        ("eng_answer", pa.string()),
        # Flagged rather than dropped: the query's passages are still valid corpus content and its
        # qrels still describe real relevance. Only the query *text* is unusable, so the benchmark
        # and the chunking lab filter on this column instead of the row vanishing and silently
        # breaking joins.
        ("degenerate", pa.bool_()),
    ]
)

QREL_SCHEMA = pa.schema(
    [
        ("query_id", pa.int64()),
        ("lang", pa.string()),
        ("passage_id", pa.string()),
        ("is_selected", pa.int8()),
        ("position", pa.int32()),
    ]
)


class BufferedWriter:
    """Incremental parquet writer. Keeps memory flat regardless of corpus size."""

    def __init__(self, path: Path, schema: pa.Schema, flush_every: int = FLUSH_EVERY) -> None:
        self.path = path
        self.schema = schema
        self.flush_every = flush_every
        self._rows: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self.n_written = 0

    def add(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=self.schema)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self._writer.write_table(table)
        self.n_written += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# ---------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------


@dataclass
class BuildStats:
    rows_seen: int = 0
    rows_read: int = 0
    passages_seen: int = 0
    passages_kept: int = 0
    passages_deduped: int = 0
    passages_filtered: int = 0
    passages_degenerate: int = 0
    queries_degenerate: int = 0
    per_lang_kept: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_seen": self.rows_seen,
            "rows_read": self.rows_read,
            "passages_seen": self.passages_seen,
            "passages_kept": self.passages_kept,
            "passages_deduped": self.passages_deduped,
            "passages_filtered": self.passages_filtered,
            "passages_degenerate": self.passages_degenerate,
            "queries_degenerate": self.queries_degenerate,
            "per_lang_kept": self.per_lang_kept,
        }


def download_shard(lang: str, cache_dir: Path) -> Path:
    filename = LANG_FILES[lang]
    remote = f"validation/{filename}"
    assert_disk_headroom(cache_dir, SHARD_HEADROOM_BYTES, f"download {remote}")
    log.info("downloading %s (~0.5 GB)", remote)
    t0 = time.perf_counter()
    local = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=remote,
        local_dir=str(cache_dir),
    )
    path = Path(local)
    log.info(
        "downloaded %s -> %.2f GB in %.1fs",
        remote,
        path.stat().st_size / 1024**3,
        time.perf_counter() - t0,
    )
    return path


def iter_rows(path: Path, batch_size: int = 512) -> Iterator[dict[str, Any]]:
    """Stream a shard by record batch so a 0.5 GB file never lands in memory whole."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def build(
    languages: list[str],
    out_dir: Path,
    cache_dir: Path,
    limit_queries: int | None,
    keep_shards: bool,
    sample_1_in: int = 1,
) -> BuildStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    passages = BufferedWriter(out_dir / "passages.parquet", PASSAGE_SCHEMA)
    queries = BufferedWriter(out_dir / "queries.parquet", QUERY_SCHEMA)
    qrels = BufferedWriter(out_dir / "qrels.parquet", QREL_SCHEMA)

    seen: set[str] = set()
    stats = BuildStats()

    def emit_passage(
        raw: str, lang: str, query_type: str, query_id: int, position: int, is_selected: int
    ) -> None:
        """Normalise, filter, dedupe, and record both the passage and its relevance judgment.

        The qrel is written even when the passage itself is a duplicate: the passage exists once in
        the corpus, but it can be relevant to several queries, and dropping those links would
        silently shrink the ground truth the lab evaluates against.
        """
        stats.passages_seen += 1
        text = normalize(raw)
        if not (MIN_PASSAGE_CHARS <= len(text) <= MAX_PASSAGE_CHARS):
            stats.passages_filtered += 1
            return
        if repetition_ratio(text) > MAX_REPETITION_RATIO:
            stats.passages_degenerate += 1
            return

        pid = content_id(text)
        qrels.add(
            {
                "query_id": query_id,
                "lang": lang,
                "passage_id": pid,
                "is_selected": is_selected,
                "position": position,
            }
        )

        if pid in seen:
            stats.passages_deduped += 1
            return
        seen.add(pid)

        passages.add(
            {
                "passage_id": pid,
                "text": text,
                "lang": lang,
                "query_type": query_type,
                "source_query_id": query_id,
                "position": position,
                "n_chars": len(text),
                "n_words": text.count(" ") + 1,
            }
        )
        stats.passages_kept += 1
        stats.per_lang_kept[lang] = stats.per_lang_kept.get(lang, 0) + 1

    try:
        for lang in languages:
            shard = download_shard(lang, cache_dir)
            try:
                assert_disk_headroom(out_dir, MIN_FREE_BYTES, f"process {lang}")
                t0 = time.perf_counter()
                n_rows = 0

                for row in iter_rows(shard):
                    if limit_queries is not None and n_rows >= limit_queries:
                        break

                    qid = int(row["query_id"])
                    stats.rows_seen += 1
                    if not keep_query(qid, sample_1_in):
                        continue

                    n_rows += 1
                    stats.rows_read += 1
                    qtype = row.get("query_type") or ""
                    pas = row.get("passages") or {}
                    translated = pas.get("Translated_passages") or []
                    english = pas.get("English_passages") or []
                    selected = pas.get("is_selected") or []

                    query_text = normalize(row.get("query") or "")
                    is_degenerate = repetition_ratio(query_text) > MAX_REPETITION_RATIO
                    if is_degenerate:
                        stats.queries_degenerate += 1

                    queries.add(
                        {
                            "query_id": qid,
                            "lang": lang,
                            "query": query_text,
                            "answer": normalize(row.get("Answer") or ""),
                            "query_type": qtype,
                            "eng_query": normalize(row.get("Eng_Query") or ""),
                            "eng_answer": normalize(row.get("Eng_Answer") or ""),
                            "degenerate": is_degenerate,
                        }
                    )

                    for i, ptext in enumerate(translated):
                        if ptext:
                            emit_passage(
                                ptext,
                                lang,
                                qtype,
                                qid,
                                i,
                                int(selected[i]) if i < len(selected) else 0,
                            )
                    for i, ptext in enumerate(english):
                        if ptext:
                            emit_passage(
                                ptext,
                                "en",
                                qtype,
                                qid,
                                i,
                                int(selected[i]) if i < len(selected) else 0,
                            )

                log.info(
                    "%s: %d queries, %d passages kept so far (%.1fs)",
                    lang,
                    n_rows,
                    stats.passages_kept,
                    time.perf_counter() - t0,
                )
            finally:
                if not keep_shards and shard.exists():
                    shard.unlink()
                    log.info("deleted shard %s (free: %.1f GB)", shard.name, free_bytes(out_dir) / 1024**3)
    finally:
        passages.close()
        queries.close()
        qrels.close()

    manifest = {
        "source_dataset": REPO_ID,
        "split": "validation",
        "languages": languages,
        "indexed_languages": sorted(stats.per_lang_kept),
        "sample_1_in": sample_1_in,
        "stats": stats.as_dict(),
        "built_at_unix": int(time.time()),
        "files": {
            "passages": "passages.parquet",
            "queries": "queries.parquet",
            "qrels": "qrels.parquet",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", nargs="+", default=["hi", "gu", "bn", "ta"], choices=sorted(LANG_FILES))
    ap.add_argument("--out", type=Path, default=Path("data/corpus"))
    ap.add_argument("--cache", type=Path, default=Path(".cache/shards"))
    ap.add_argument("--limit-queries", type=int, default=None, help="Per-language cap, for smoke runs")
    ap.add_argument(
        "--sample-1-in",
        type=int,
        default=16,
        help="Keep 1 query in N, chosen by a hash of query_id so the same queries are selected in "
        "every language. Default 16 -> ~6.1k queries/lang -> ~290k passages. Use 1 for everything "
        "(~4.75M passages; will not fit in memory or the latency budget).",
    )
    ap.add_argument("--keep-shards", action="store_true", help="Do not delete shards after use")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("free disk: %.1f GB", free_bytes(Path.cwd()) / 1024**3)
    t0 = time.perf_counter()
    try:
        stats = build(
            args.languages,
            args.out,
            args.cache,
            args.limit_queries,
            args.keep_shards,
            sample_1_in=args.sample_1_in,
        )
    except DiskGuardError as e:
        log.error("%s", e)
        return 2

    dedupe_rate = stats.passages_deduped / max(1, stats.passages_seen)
    log.info("built in %.1fs", time.perf_counter() - t0)
    log.info("  queries in shards %d", stats.rows_seen)
    log.info("  queries sampled   %d (1 in %d)", stats.rows_read, args.sample_1_in)
    log.info("  passages seen     %d", stats.passages_seen)
    log.info("  passages kept     %d", stats.passages_kept)
    log.info("  deduped           %d (%.1f%%)", stats.passages_deduped, 100 * dedupe_rate)
    log.info("  filtered (length) %d", stats.passages_filtered)
    log.info("  filtered (degen)  %d", stats.passages_degenerate)
    log.info("  degenerate queries %d (flagged, not dropped)", stats.queries_degenerate)
    for lang, n in sorted(stats.per_lang_kept.items()):
        log.info("  %-3s %d", lang, n)
    log.info("output -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
