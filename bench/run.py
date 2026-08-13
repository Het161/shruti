#!/usr/bin/env python3
"""Benchmark harness — this produces the numbers we publish.

Design rules, each of which exists because the alternative would have produced a prettier but
dishonest number:

**Run against the deployed URL, not localhost.** Localhost measures a laptop, and the claim is
about a deployed system.

**Report two clocks, and their difference.** `server_ms` comes from the deployed process's own
monotonic timer via the `X-Server-Time-Ms` header; `client_ms` is this harness's wall clock. Their
difference is network, and it is published as its own column rather than folded into either. A run
from Gujarat to a US host is dominated by ~250ms of Pacific; the same binary measured from a US
runner is not. Both are real, and they answer different questions.

**Warmup is excluded and stated.** The first queries after a cold start pay model load and page
faults. Including them would inflate the tail; hiding that they were dropped would be dishonest. So
they are run, excluded, and counted in the artifact.

**Percentiles include P100.** P100 is the max — a single bad request cannot hide behind an average.
The task asks for P50/P70/P100 and that is exactly what is reported, per stage and per language.

**Tier 2 is off.** The SLO is about the guaranteed answer path. Including generation would measure
Groq's queue depth, not this system.

Usage
-----
    python bench/run.py --url https://shruti-xxx.run.app --n 300
    python bench/run.py --url http://127.0.0.1:7860 --n 300 --out bench/results/local.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. P100 is the maximum, by construction."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if p >= 100:
        return ordered[-1]
    idx = min(len(ordered) - 1, int(len(ordered) * p / 100))
    return ordered[idx]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "n": len(values),
        "p50": round(percentile(values, 50), 3),
        "p70": round(percentile(values, 70), 3),
        "p90": round(percentile(values, 90), 3),
        "p99": round(percentile(values, 99), 3),
        "p100": round(percentile(values, 100), 3),
        "mean": round(statistics.mean(values), 3),
    }


def load_queries(corpus_dir: Path, n: int, seed_skip: int) -> list[dict[str, str]]:
    """Sample benchmark queries across all indexed languages.

    Degenerate queries are excluded via the `degenerate` column: a 7,783-character translation loop
    is not a realistic voice query, and letting a handful of them into the tail would make P100
    describe a dataset artifact rather than the system.
    """
    table = pq.read_table(corpus_dir / "queries.parquet")
    cols = table.column_names
    rows = table.to_pylist()

    usable = [
        r
        for r in rows
        if r.get("query")
        and 3 <= len(r["query"]) <= 300
        and not (r.get("degenerate") if "degenerate" in cols else False)
    ]

    # Round-robin across languages so no single language dominates the sample and the per-language
    # breakdown has enough support to mean anything.
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in usable:
        by_lang[r["lang"]].append(r)

    picked: list[dict[str, str]] = []
    idx = seed_skip
    while len(picked) < n and any(by_lang.values()):
        for lang in sorted(by_lang):
            pool = by_lang[lang]
            if not pool:
                continue
            picked.append(pool[idx % len(pool)])
            if len(picked) >= n:
                break
        idx += 1
    return picked[:n]


def run(
    url: str, queries: list[dict[str, str]], warmup: int, search_mode: str, lane: str
) -> dict[str, Any]:
    client = httpx.Client(timeout=30.0)
    endpoint = f"{url.rstrip('/')}/api/ask"

    # --- cold start, measured separately and never folded into query latency ----------
    cold_ms = None
    t0 = time.perf_counter()
    try:
        health = client.get(f"{url.rstrip('/')}/api/health", timeout=120.0)
        cold_ms = (time.perf_counter() - t0) * 1000
        health_json = health.json()
    except Exception as e:
        print(f"health check failed: {e}", file=sys.stderr)
        health_json = {}

    print(f"health: ready={health_json.get('ready')} passages={health_json.get('corpus_passages')}")
    print(f"first-contact (incl. any cold start): {cold_ms:.0f}ms" if cold_ms else "")

    def ask(text: str) -> tuple[dict[str, Any] | None, float, float]:
        started = time.perf_counter()
        try:
            r = client.post(
                endpoint,
                json={"text": text, "search_mode": search_mode, "lane": lane, "generate": False},
            )
            client_ms = (time.perf_counter() - started) * 1000
            if r.status_code != 200:
                return None, client_ms, 0.0
            server_ms = float(r.headers.get("X-Server-Time-Ms", "0"))
            return r.json(), client_ms, server_ms
        except Exception:
            return None, (time.perf_counter() - started) * 1000, 0.0

    print(f"warmup: {warmup} queries (excluded from all statistics)")
    for q in queries[:warmup]:
        ask(q["query"])

    measured = queries[warmup:]
    print(f"measuring: {len(measured)} queries, mode={search_mode} lane={lane}")

    client_times: list[float] = []
    server_times: list[float] = []
    pipeline_times: list[float] = []
    stage_times: dict[str, list[float]] = defaultdict(list)
    per_lang: dict[str, list[float]] = defaultdict(list)
    refusals = 0
    errors = 0
    answered = 0

    for i, q in enumerate(measured):
        data, client_ms, server_ms = ask(q["query"])
        if data is None:
            errors += 1
            continue

        client_times.append(client_ms)
        server_times.append(server_ms)
        pipeline_times.append(data["timings"]["total_ms"])
        per_lang[q["lang"]].append(data["timings"]["total_ms"])
        for s in data["timings"]["stages"]:
            stage_times[s["name"]].append(s["duration_ms"])

        if data["answer"] is None:
            refusals += 1
        else:
            answered += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(measured)}  pipeline p50={percentile(pipeline_times, 50):.2f}ms")

    client.close()

    return {
        "target_url": url,
        "config": {"search_mode": search_mode, "lane": lane},
        "health": health_json,
        "counts": {
            "queries_total": len(queries),
            "warmup_excluded": warmup,
            "measured": len(measured),
            "answered": answered,
            "refused": refusals,
            "errors": errors,
        },
        "cold_start_first_contact_ms": round(cold_ms, 1) if cold_ms else None,
        "slo": {
            # The headline number: the deployed server's own measurement of its full request
            # handling, network excluded.
            "server_side_ms": summarize(server_times),
            # The pipeline's internal total, excluding FastAPI/uvicorn overhead.
            "pipeline_ms": summarize(pipeline_times),
            # What a client at this location actually experienced, network included.
            "client_observed_ms": summarize(client_times),
            # The difference, stated rather than hidden.
            "network_overhead_ms": summarize(
                [c - s for c, s in zip(client_times, server_times, strict=True) if c >= s]
            ),
        },
        "stages_ms": {k: summarize(v) for k, v in sorted(stage_times.items())},
        "per_language_pipeline_ms": {k: summarize(v) for k, v in sorted(per_lang.items())},
        "measured_at_unix": int(time.time()),
    }


def print_report(result: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print(f"SHRUTI benchmark — {result['target_url']}")
    print("=" * 72)
    c = result["counts"]
    print(
        f"measured {c['measured']}  answered {c['answered']}  refused {c['refused']}  errors {c['errors']}"
        f"  (warmup {c['warmup_excluded']} excluded)"
    )
    print()
    print(f"{'SLO':<26}{'P50':>9}{'P70':>9}{'P90':>9}{'P100':>9}")
    print("-" * 72)
    for name, s in result["slo"].items():
        if s:
            print(f"{name:<26}{s['p50']:>9.2f}{s['p70']:>9.2f}{s['p90']:>9.2f}{s['p100']:>9.2f}")
    print()
    print(f"{'STAGE':<26}{'P50':>9}{'P70':>9}{'P90':>9}{'P100':>9}")
    print("-" * 72)
    for name, s in result["stages_ms"].items():
        print(f"{name:<26}{s['p50']:>9.3f}{s['p70']:>9.3f}{s['p90']:>9.3f}{s['p100']:>9.3f}")
    print()
    print(f"{'LANGUAGE':<26}{'N':>9}{'P50':>9}{'P70':>9}{'P100':>9}")
    print("-" * 72)
    for name, s in result["per_language_pipeline_ms"].items():
        print(f"{name:<26}{s['n']:>9}{s['p50']:>9.2f}{s['p70']:>9.2f}{s['p100']:>9.2f}")

    target = result["slo"].get("server_side_ms", {})
    if target:
        verdict = "MET" if target["p100"] < 200 else "MISSED"
        print()
        print(f"200ms P100 target (server-side): {verdict}  — P100 = {target['p100']:.2f}ms")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="Deployed base URL (production is the source of truth)")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--n", type=int, default=320, help="Total queries incl. warmup")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--search-mode", default="hnsw", choices=["hnsw", "exact"])
    ap.add_argument("--lane", default="fast", choices=["fast", "quality"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.n - args.warmup < 300:
        print(
            f"note: {args.n - args.warmup} measured queries is below the 300 the task asks for",
            file=sys.stderr,
        )

    queries = load_queries(args.corpus, args.n, seed_skip=0)
    if len(queries) < args.n:
        print(f"note: only {len(queries)} usable queries available", file=sys.stderr)

    result = run(args.url, queries, args.warmup, args.search_mode, args.lane)
    print_report(result)

    out = args.out or Path("bench/results") / f"bench-{result['measured_at_unix']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nartifact -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
