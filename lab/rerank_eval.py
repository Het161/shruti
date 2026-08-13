#!/usr/bin/env python3
"""Does the cross-encoder actually buy precision, and what does it cost?

Two numbers decide whether this ships, not one. A reranker that lifts MRR@10 while adding 600ms is
not an improvement to a system whose headline claim is a 200ms answer — so ΔMRR and Δlatency are
reported together at every depth, and the depth is a knob rather than a constant.

Protocol mirrors `lab/chunking_eval.py` so the numbers are comparable: the same qrels, the same
language-restricted ground truth (a Hindi query's gold set is its Hindi and English passages, not
its Tamil translation), and the same parent-passage collapse.

Baseline is the shipped pipeline: fast-lane dense + Indic BM25 + RRF. Reranking is applied to the
top-`depth` of that fused list.

Usage
-----
    python lab/rerank_eval.py --queries 150 --depths 10 20 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.corpus import Corpus
from app.pipeline import Pipeline
from app.settings import get_settings
from app.stages.dense import DenseIndex
from app.stages.lexical import LexicalIndex
from app.stages.rerank import get_reranker
from app.timing import RequestTimer

log = logging.getLogger("rerank_eval")


def metrics(ranked_ids: list[str], gold: set[str]) -> tuple[float, float]:
    rank = next((i + 1 for i, p in enumerate(ranked_ids[:10]) if p in gold), None)
    rr = 1.0 / rank if rank else 0.0
    recall = len(set(ranked_ids[:10]) & gold) / len(gold)
    return rr, recall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--queries", type=int, default=150)
    ap.add_argument("--depths", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--out", type=Path, default=Path("bench/results/rerank_eval.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    d = args.corpus
    settings = get_settings()
    corpus = Corpus.load(d)
    corpus.manifest["_artifact_dir"] = str(d)
    pipe = Pipeline(
        corpus,
        DenseIndex.load(d / "hnsw.usearch", corpus.dim, corpus.n_passages),
        LexicalIndex.load(str(d / "bm25"), corpus.n_passages),
        settings,
    )
    pipe.embedder_for("fast")
    # Fusion truncates to `max(context_top_n, 10)`. Left at the shipped value of 3 the harness
    # would hand the reranker 10 candidates while claiming depth 50, and every depth would quietly
    # measure the same thing.
    settings.context_top_n = max(args.depths)
    settings.dense_top_k = max(settings.dense_top_k, max(args.depths))
    settings.lexical_top_k = max(settings.lexical_top_k, max(args.depths))

    qrels = pq.read_table(d / "qrels.parquet").to_pylist()
    queries = pq.read_table(d / "queries.parquet").to_pylist()
    plang = {p["passage_id"]: p["lang"] for p in pq.read_table(d / "passages.parquet").to_pylist()}

    rel: dict[int, set[str]] = defaultdict(set)
    for r in qrels:
        if r["is_selected"] == 1:
            rel[r["query_id"]].add(r["passage_id"])

    usable = [
        q for q in queries
        if q["query_id"] in rel and q.get("query") and 3 <= len(q["query"]) <= 300
        and not q.get("degenerate", False)
    ]
    by = defaultdict(list)
    for q in usable:
        by[q["lang"]].append(q)
    ev: list[dict] = []
    i = 0
    while len(ev) < args.queries and any(len(v) > i for v in by.values()):
        for lg in sorted(by):
            if i < len(by[lg]) and len(ev) < args.queries:
                ev.append(by[lg][i])
        i += 1
    log.info("eval set: %d queries across %s", len(ev), sorted(by))

    # --- baseline: retrieve once, reuse for every depth -------------------------------
    log.info("running baseline retrieval")
    cases = []
    base_rr, base_rec, base_ms = [], [], []
    for q in ev:
        gold = {p for p in rel[q["query_id"]] if plang.get(p) in (q["lang"], "en")}
        if not gold:
            continue
        # `retrieve` rather than `ask`: the public AskRequest caps top_n at 20, which is right for
        # the API and wrong for this experiment — reranking depth 50 needs 50 candidates. This also
        # skips answer extraction, which the metric does not use.
        t0 = time.perf_counter()
        timer = RequestTimer()
        rr_res = pipe.retrieve(q["query"], timer, lane="fast", search_mode="hnsw")
        ms = (time.perf_counter() - t0) * 1000
        refs = [corpus.get(h.row) for h in rr_res.hits[:50]]
        ids = [r.passage_id for r in refs]
        texts = [r.text for r in refs]
        if not ids:
            continue
        rr, rc = metrics(ids, gold)
        base_rr.append(rr)
        base_rec.append(rc)
        base_ms.append(ms)
        cases.append({"q": q["query"], "gold": gold, "ids": ids, "texts": texts})

    log.info("baseline done on %d queries", len(cases))

    rows = [{
        "config": "baseline (dense+BM25+RRF)",
        "mrr@10": float(np.mean(base_rr)),
        "recall@10": float(np.mean(base_rec)),
        "delta_mrr_pct": 0.0,
        "rerank_p50_ms": 0.0,
        "rerank_p100_ms": 0.0,
        "total_p50_ms": float(np.percentile(base_ms, 50)),
    }]

    reranker = get_reranker()
    for depth in args.depths:
        log.info("reranking at depth %d", depth)
        rr_l, rec_l, rms = [], [], []
        for c in cases:
            out = reranker.rerank(c["q"], c["texts"], depth)
            ids = [c["ids"][i] for i in out.order]
            rr, rc = metrics(ids, c["gold"])
            rr_l.append(rr)
            rec_l.append(rc)
            rms.append(out.elapsed_ms)
        mrr = float(np.mean(rr_l))
        rows.append({
            "config": f"+ cross-encoder top-{depth}",
            "mrr@10": mrr,
            "recall@10": float(np.mean(rec_l)),
            "delta_mrr_pct": (mrr / rows[0]["mrr@10"] - 1) * 100 if rows[0]["mrr@10"] else 0.0,
            "rerank_p50_ms": float(np.percentile(rms, 50)),
            "rerank_p100_ms": float(np.max(rms)),
            "total_p50_ms": float(np.percentile(base_ms, 50)) + float(np.percentile(rms, 50)),
        })

    print()
    print("=" * 96)
    print(f"{'config':<28}{'MRR@10':>9}{'ΔMRR':>9}{'R@10':>8}{'rerank p50':>13}{'rerank p100':>13}{'total p50':>12}")
    print("-" * 96)
    for r in rows:
        print(f"{r['config']:<28}{r['mrr@10']:>9.4f}{r['delta_mrr_pct']:>8.1f}%{r['recall@10']:>8.4f}"
              f"{r['rerank_p50_ms']:>12.1f}ms{r['rerank_p100_ms']:>12.1f}ms{r['total_p50_ms']:>10.1f}ms")
    print("=" * 96)
    print("SLO reminder: Tier 1 answer budget is 200ms. Any row whose rerank cost approaches that")
    print("cannot sit on the Tier 1 path — which is why the reranker ships in the Tier 2 lane.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_queries": len(cases), "rows": rows}, indent=2))
    print(f"\nartifact -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
