#!/usr/bin/env python3
"""Chunking Lab — six strategies, scored with real IR metrics.

The task asks for "vast" chunking with real thought behind it. Implementing six strategies and
asserting that one is best would satisfy the letter of that. This instead *measures* them, which is
possible only because MSMARCO-XI ships `is_selected` relevance labels — so every strategy can be
scored on MRR@10, Recall@5, and Recall@20 against ground truth rather than against an opinion.

The strategies
--------------
1. **passage-native** — the dataset's own passages. The baseline to beat, and a strong one:
   MS MARCO passages were curated as retrieval units at ~59 words.
2. **fixed-256-64** — 256 characters, 64 overlap. The naive baseline the task warns about,
   implemented precisely so the table shows *why* it loses rather than merely asserting it does.
3. **sentence-window** — embed each sentence, return the sentence plus a ±1 sentence window.
   Precise matching, wider context.
4. **semantic-boundary** — split where adjacent-sentence embedding similarity drops below a
   threshold, so chunks break at topic shifts instead of arbitrary offsets.
5. **parent-child** — embed small children, retrieve the full parent passage as context.
   Small-to-big: match precisely, answer with context.
6. **metadata-aware** — passage-native plus language-partitioned search and a `query_type` prior.

Evaluation protocol
-------------------
Sampling is by *query*, not by passage. Every passage belonging to a sampled query is indexed, so
ground truth is guaranteed present and Recall is meaningful — sampling passages directly would
silently delete the correct answer for some queries and score every strategy against an
unachievable ceiling.

A chunk is a hit when the passage it came from is labelled relevant. Chunk-level strategies are
therefore mapped back to parent passages before scoring, which is the only way to compare a
sentence-level index with a passage-level one on the same footing.

Usage
-----
    python lab/chunking_eval.py --corpus data/corpus --queries 500
    python lab/chunking_eval.py --corpus data/corpus --queries 500 --strategies passage-native fixed-256-64
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.stages.answer import split_sentences
from app.stages.embed import get_embedder

log = logging.getLogger("chunking_eval")

K_VALUES = (5, 10, 20)


@dataclass
class Chunk:
    """One indexed unit, with a pointer back to the passage it came from."""

    text: str
    parent_passage_id: str
    lang: str
    query_type: str | None = None
    # Text returned as context, which may be larger than the text that was embedded — this is the
    # entire point of sentence-window and parent-child.
    context_text: str | None = None


@dataclass
class StrategyResult:
    name: str
    n_chunks: int
    mean_chunk_chars: float
    build_seconds: float
    embed_seconds: float
    mrr_at_10: float = 0.0
    recall: dict[int, float] = field(default_factory=dict)
    query_latency_ms: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "chunks": self.n_chunks,
            "mean_chars": round(self.mean_chunk_chars, 1),
            "mrr@10": round(self.mrr_at_10, 4),
            **{f"recall@{k}": round(v, 4) for k, v in sorted(self.recall.items())},
            "latency_p50_ms": round(self.query_latency_ms.get("p50", 0), 3),
            "latency_p100_ms": round(self.query_latency_ms.get("p100", 0), 3),
            "build_s": round(self.build_seconds, 1),
        }


# ---------------------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------------------


def chunk_passage_native(passages: list[dict[str, Any]]) -> list[Chunk]:
    return [
        Chunk(p["text"], p["passage_id"], p["lang"], p.get("query_type"))
        for p in passages
    ]


def chunk_fixed(passages: list[dict[str, Any]], size: int = 256, overlap: int = 64) -> list[Chunk]:
    """Fixed-width character windows.

    Character-based rather than token-based deliberately: this is the naive strategy, and character
    windows are what naive implementations actually do. It is also actively hostile to Indic
    scripts, where a grapheme cluster spans several codepoints — a cut can land mid-cluster and
    produce a fragment that is not a word in any language. The table should show that cost.
    """
    out: list[Chunk] = []
    step = max(1, size - overlap)
    for p in passages:
        text = p["text"]
        for start in range(0, max(1, len(text)), step):
            piece = text[start : start + size]
            if len(piece) < 40:
                continue
            out.append(Chunk(piece, p["passage_id"], p["lang"], p.get("query_type")))
    return out


def chunk_sentence_window(passages: list[dict[str, Any]], window: int = 1) -> list[Chunk]:
    """Embed one sentence, return it with its neighbours as context."""
    out: list[Chunk] = []
    for p in passages:
        sents = split_sentences(p["text"])
        for i, sent in enumerate(sents):
            lo = max(0, i - window)
            hi = min(len(sents), i + window + 1)
            out.append(
                Chunk(
                    text=sent,
                    parent_passage_id=p["passage_id"],
                    lang=p["lang"],
                    query_type=p.get("query_type"),
                    context_text=" ".join(sents[lo:hi]),
                )
            )
    return out


def chunk_semantic(
    passages: list[dict[str, Any]], embedder: Any, threshold: float = 0.55, batch: int = 4096
) -> list[Chunk]:
    """Split where adjacent-sentence similarity drops — i.e. at topic boundaries.

    All sentences across the corpus are embedded in one batched pass rather than per passage.
    With a static embedder the per-call overhead dominates the arithmetic, so batching is worth
    roughly an order of magnitude here.
    """
    per_passage: list[list[str]] = [split_sentences(p["text"]) for p in passages]
    flat: list[str] = [s for sents in per_passage for s in sents]
    if not flat:
        return []

    vecs = embedder.encode_batch(flat, batch_size=batch)

    out: list[Chunk] = []
    cursor = 0
    for p, sents in zip(passages, per_passage, strict=True):
        if not sents:
            continue
        block = vecs[cursor : cursor + len(sents)]
        cursor += len(sents)

        current = [sents[0]]
        for i in range(1, len(sents)):
            similarity = float(block[i] @ block[i - 1])
            if similarity < threshold and current:
                out.append(Chunk(" ".join(current), p["passage_id"], p["lang"], p.get("query_type")))
                current = []
            current.append(sents[i])
        if current:
            out.append(Chunk(" ".join(current), p["passage_id"], p["lang"], p.get("query_type")))
    return out


def chunk_parent_child(passages: list[dict[str, Any]], child_chars: int = 120) -> list[Chunk]:
    """Embed small children; return the whole parent passage as context."""
    out: list[Chunk] = []
    for p in passages:
        sents = split_sentences(p["text"])
        for sent in sents:
            for start in range(0, max(1, len(sent)), child_chars):
                piece = sent[start : start + child_chars]
                if len(piece) < 30:
                    continue
                out.append(
                    Chunk(
                        text=piece,
                        parent_passage_id=p["passage_id"],
                        lang=p["lang"],
                        query_type=p.get("query_type"),
                        context_text=p["text"],
                    )
                )
    return out


# ---------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------


def evaluate(
    name: str,
    chunks: list[Chunk],
    queries: list[dict[str, Any]],
    relevant: dict[int, set[str]],
    embedder: Any,
    *,
    build_seconds: float,
    passage_lang: dict[str, str],
    lang_filter: bool = False,
    type_boost: bool = False,
) -> StrategyResult:
    """Index the chunks, run the queries, score against qrels."""
    t0 = time.perf_counter()
    vectors = embedder.encode_batch([c.text for c in chunks], batch_size=4096)
    embed_seconds = time.perf_counter() - t0
    log.info("  %s: embedded %d chunks in %.1fs", name, len(chunks), embed_seconds)

    parents = np.array([c.parent_passage_id for c in chunks])
    langs = np.array([c.lang for c in chunks])
    types = np.array([c.query_type or "" for c in chunks])

    result = StrategyResult(
        name=name,
        n_chunks=len(chunks),
        mean_chunk_chars=float(np.mean([len(c.text) for c in chunks])) if chunks else 0.0,
        build_seconds=build_seconds,
        embed_seconds=embed_seconds,
    )

    reciprocal_ranks: list[float] = []
    recall_hits: dict[int, list[float]] = {k: [] for k in K_VALUES}
    latencies: list[float] = []
    max_k = max(K_VALUES)

    for q in queries:
        gold = relevant.get(q["query_id"], set())
        # Restrict ground truth to the query's own language plus English.
        #
        # `query_id` is shared across every language shard, so the raw qrels for a Hindi query also
        # mark its Gujarati, Bengali, and Tamil translations relevant. Scoring against that set
        # rewards a system for returning a Tamil passage to a Hindi speaker, and punishes language
        # filtering for correctly excluding it — which is exactly backwards. The first run of this
        # lab showed the symptom clearly: metadata-aware led on MRR@10 while placing last on
        # recall@20, purely because it declined to retrieve translations nobody asked for.
        #
        # The restriction is applied identically to every strategy, so the comparison stays fair;
        # it changes what "relevant" means, not who is allowed to benefit from it.
        gold = {pid for pid in gold if passage_lang.get(pid) in (q["lang"], "en")}
        if not gold:
            continue

        started = time.perf_counter()
        qv = embedder.encode_query(q["query"])
        scores = vectors @ qv

        if lang_filter:
            # Metadata-aware: restrict to the query's language plus English, by masking rather
            # than slicing so no copy of the matrix is made.
            allowed = (langs == q["lang"]) | (langs == "en")
            scores = np.where(allowed, scores, -np.inf)
        if type_boost and q.get("query_type"):
            scores = scores + 0.02 * (types == q["query_type"])

        top = np.argpartition(-scores, min(max_k * 4, len(scores) - 1))[: max_k * 4]
        top = top[np.argsort(-scores[top])]

        # Collapse chunks to parent passages, preserving rank order. Without this, a strategy that
        # emits ten chunks per passage would fill its top-10 with one passage and score as if it
        # had retrieved ten distinct documents.
        seen: list[str] = []
        for row in top:
            pid = parents[row]
            if pid not in seen:
                seen.append(pid)
            if len(seen) >= max_k:
                break
        latencies.append((time.perf_counter() - started) * 1000)

        rank = next((i + 1 for i, pid in enumerate(seen[:10]) if pid in gold), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in K_VALUES:
            found = len(set(seen[:k]) & gold)
            recall_hits[k].append(found / len(gold))

    result.mrr_at_10 = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    result.recall = {k: float(np.mean(v)) if v else 0.0 for k, v in recall_hits.items()}
    if latencies:
        ordered = sorted(latencies)
        result.query_latency_ms = {
            "p50": ordered[len(ordered) // 2],
            "p100": ordered[-1],
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--queries", type=int, default=500, help="Eval queries per run")
    ap.add_argument("--out", type=Path, default=Path("bench/results/chunking_lab.json"))
    ap.add_argument("--strategies", nargs="*", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    qrels = pq.read_table(args.corpus / "qrels.parquet").to_pylist()
    queries_all = pq.read_table(args.corpus / "queries.parquet").to_pylist()
    passages_all = pq.read_table(args.corpus / "passages.parquet").to_pylist()

    relevant: dict[int, set[str]] = defaultdict(set)
    for r in qrels:
        if r["is_selected"] == 1:
            relevant[r["query_id"]].add(r["passage_id"])

    usable = [
        q
        for q in queries_all
        if q["query_id"] in relevant
        and q.get("query")
        and 3 <= len(q["query"]) <= 300
        and not q.get("degenerate", False)
    ]
    log.info("queries with >=1 relevant passage: %d of %d", len(usable), len(queries_all))

    # Round-robin across languages so every language is represented in the eval set.
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in usable:
        by_lang[q["lang"]].append(q)
    eval_queries: list[dict[str, Any]] = []
    i = 0
    while len(eval_queries) < args.queries and any(len(v) > i for v in by_lang.values()):
        for lang in sorted(by_lang):
            if i < len(by_lang[lang]) and len(eval_queries) < args.queries:
                eval_queries.append(by_lang[lang][i])
        i += 1
    log.info("eval set: %d queries across %s", len(eval_queries), sorted(by_lang))

    # Index every passage belonging to a sampled query, so ground truth is always present.
    wanted_qids = {q["query_id"] for q in eval_queries}
    wanted_pids = {r["passage_id"] for r in qrels if r["query_id"] in wanted_qids}
    passages = [p for p in passages_all if p["passage_id"] in wanted_pids]
    log.info("indexing %d passages for the eval subset", len(passages))

    passage_lang: dict[str, str] = {p["passage_id"]: p["lang"] for p in passages_all}

    embedder = get_embedder("fast")

    builders = {
        "passage-native": lambda: chunk_passage_native(passages),
        "fixed-256-64": lambda: chunk_fixed(passages, 256, 64),
        "sentence-window": lambda: chunk_sentence_window(passages, 1),
        "semantic-boundary": lambda: chunk_semantic(passages, embedder, 0.55),
        "parent-child": lambda: chunk_parent_child(passages, 120),
        "metadata-aware": lambda: chunk_passage_native(passages),
    }
    selected = args.strategies or list(builders)

    results: list[StrategyResult] = []
    for name in selected:
        if name not in builders:
            log.warning("unknown strategy %s, skipping", name)
            continue
        log.info("strategy: %s", name)
        t0 = time.perf_counter()
        chunks = builders[name]()
        build_s = time.perf_counter() - t0
        if not chunks:
            log.warning("  produced no chunks, skipping")
            continue
        results.append(
            evaluate(
                name,
                chunks,
                eval_queries,
                relevant,
                embedder,
                build_seconds=build_s,
                passage_lang=passage_lang,
                lang_filter=(name == "metadata-aware"),
                type_boost=(name == "metadata-aware"),
            )
        )

    rows = [r.as_row() for r in results]
    rows.sort(key=lambda r: r["mrr@10"], reverse=True)

    headers = list(rows[0]) if rows else []
    widths = {h: max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in headers}
    print()
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-|-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(widths[h]) for h in headers))
    print()
    if rows:
        print(f"winner by MRR@10: {rows[0]['strategy']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "eval_queries": len(eval_queries),
                "indexed_passages": len(passages),
                "languages": sorted(by_lang),
                "k_values": list(K_VALUES),
                "results": rows,
                "measured_at_unix": int(time.time()),
            },
            indent=2,
        )
    )
    print(f"artifact -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
