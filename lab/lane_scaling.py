#!/usr/bin/env python3
"""Does a contextual encoder degrade more gracefully with corpus size than a static one?

The measured precision-vs-size curve for the shipped fast lane (potion, static Model2Vec) is steep:
MRR@10 falls ~60% going from 6k to 310k passages, because added passages are distractors rather
than answers and a static bag-of-subwords centroid cannot separate them.

The open question is whether that steepness is a property of *static* embeddings specifically or of
the task. If e5-small — a real transformer with contextual attention — holds up better as
distractors accumulate, then the quality lane is worth its 477 MB and ~26 minutes of offline
compute. If it degrades at the same rate, the lane can be dropped with evidence instead of left as
an unexplained toggle.

Controlled comparison: identical eval queries, identical passage sets, identical metric. Only the
encoder changes. Both are evaluated at the same sizes so the *slope* is comparable even though the
absolute numbers are not directly meaningful across models.

Usage
-----
    python lab/lane_scaling.py --queries 120 --sizes 6000 20000 40000
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

from app.stages.embed import get_embedder
from app.stages.lang import retrieval_langs

log = logging.getLogger("lane_scaling")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--queries", type=int, default=120)
    ap.add_argument("--sizes", type=int, nargs="+", default=[6000, 20000, 40000])
    ap.add_argument("--out", type=Path, default=Path("bench/results/lane_scaling.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    d = args.corpus
    qrels = pq.read_table(d / "qrels.parquet").to_pylist()
    queries = pq.read_table(d / "queries.parquet").to_pylist()
    passages = pq.read_table(d / "passages.parquet").to_pylist()
    plang = {p["passage_id"]: p["lang"] for p in passages}

    rel: dict[int, set[str]] = defaultdict(set)
    for r in qrels:
        if r["is_selected"] == 1:
            rel[r["query_id"]].add(r["passage_id"])

    usable = [q for q in queries if q["query_id"] in rel and q.get("query")
              and 3 <= len(q["query"]) <= 300 and not q.get("degenerate", False)]
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

    # Core = every passage belonging to an eval query, so ground truth is always present.
    qids = {q["query_id"] for q in ev}
    core_pids = {r["passage_id"] for r in qrels if r["query_id"] in qids}
    core = [i for i, p in enumerate(passages) if p["passage_id"] in core_pids]
    rest = [i for i in range(len(passages)) if i not in set(core)]
    rng = np.random.default_rng(0)
    rest = rng.permutation(rest).tolist()
    log.info("eval %d queries · core %d passages", len(ev), len(core))

    pid = np.array([p["passage_id"] for p in passages])
    text = [p["text"] for p in passages]

    results: dict[str, list[dict]] = {}
    for lane in ("fast", "quality"):
        emb = get_embedder(lane)
        log.info("lane=%s dim=%d — embedding passages", lane, emb.dim)
        biggest = max(args.sizes)
        rows_needed = (core + rest)[:max(biggest, len(core))]
        t0 = time.perf_counter()
        vecs = emb.encode_batch([text[i] for i in rows_needed], batch_size=256)
        log.info("  embedded %d in %.1fs (%.0f/s)", len(rows_needed),
                 time.perf_counter() - t0, len(rows_needed) / max(1e-9, time.perf_counter() - t0))
        qv = {q["query_id"]: emb.encode_query(q["query"]) for q in ev}

        lane_rows = []
        for size in args.sizes:
            n = max(size, len(core))
            sub = rows_needed[:n]
            V = vecs[:n]
            rr, rec = [], []
            for q in ev:
                gold = {p for p in rel[q["query_id"]] if plang.get(p) in (q["lang"], "en")}
                if not gold:
                    continue
                sc = V @ qv[q["query_id"]]
                langs = set(retrieval_langs(q["lang"]))
                k = min(20, len(sc))
                top = np.argpartition(-sc, k - 1)[:k]
                top = top[np.argsort(-sc[top])]
                got: list[str] = []
                for t in top:
                    p_id = pid[sub[t]]
                    if plang.get(p_id) in langs and p_id not in got:
                        got.append(p_id)
                r = next((j + 1 for j, p in enumerate(got[:10]) if p in gold), None)
                rr.append(1 / r if r else 0.0)
                rec.append(len(set(got[:20]) & gold) / len(gold))
            lane_rows.append({"size": n, "mrr@10": float(np.mean(rr)), "recall@20": float(np.mean(rec))})
            log.info("  size %6d  MRR@10 %.4f  R@20 %.4f", n, lane_rows[-1]["mrr@10"], lane_rows[-1]["recall@20"])
        results[lane] = lane_rows

    print()
    print("=" * 78)
    print("PRECISION vs CORPUS SIZE — static (potion 256d) vs contextual (e5-small 384d)")
    print("=" * 78)
    print(f"{'size':>10}{'potion MRR':>14}{'e5 MRR':>12}{'potion drop':>14}{'e5 drop':>12}")
    print("-" * 78)
    for i in range(len(args.sizes)):
        f = results["fast"][i]
        q = results["quality"][i]
        fd = (f["mrr@10"] / results["fast"][0]["mrr@10"] - 1) * 100
        qd = (q["mrr@10"] / results["quality"][0]["mrr@10"] - 1) * 100
        print(f"{f['size']:>10,}{f['mrr@10']:>14.4f}{q['mrr@10']:>12.4f}{fd:>13.1f}%{qd:>11.1f}%")
    print("=" * 78)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_queries": len(ev), "results": results}, indent=2))
    print(f"artifact -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
