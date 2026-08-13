#!/usr/bin/env python3
"""Calibrate the scope gate's abstention threshold τ.

The system must know when *not* to answer. Retrieval always returns something — a top-k list is
never empty, only sometimes meaningless — so abstention needs a threshold, and a threshold needs
calibration rather than a guess. Until this script runs, `SHRUTI_SCOPE_TAU` is unset and the gate
reports itself uncalibrated in every response instead of pretending.

Method
------
Two distributions of the **top dense cosine score**:

- **in-domain**: real queries sampled from the corpus's own query set.
- **out-of-domain**: an authored probe set of questions the corpus provably cannot answer —
  conversational ("what is your name"), assistant-meta ("who created you"), personal ("what is my
  bank balance"), and actionable ("order me a pizza"), across all five indexed languages.

τ is chosen as the smallest threshold achieving the target out-of-domain rejection rate, and the
in-domain cost of that choice — the fraction of answerable queries it wrongly refuses — is reported
alongside. A gate that rejects 100% of out-of-domain traffic by refusing everything is not a
guardrail, so both numbers ship.

Why the dense cosine and not the fused RRF score: RRF is rank-derived and has no stable meaning
across queries, so a fixed threshold on it drifts with how many retrievers returned hits. Cosine
similarity is comparable across queries by construction. See `app/stages/fuse.py`.

Usage
-----
    python lab/calibrate_scope.py --corpus data/corpus --target-rejection 0.95
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.corpus import Corpus
from app.stages.embed import get_embedder
from app.stages.lang import retrieval_langs

log = logging.getLogger("calibrate_scope")

# Out-of-domain probe set. Authored, not sampled — the point is questions whose answers are
# provably absent from an MS MARCO passage corpus, which is a property no filter over the dataset
# itself can produce. Four families, deliberately: conversational, assistant-meta, personal-private,
# and actionable-command. Each is a realistic thing a user will say to a voice interface and none
# has an answer in the corpus.
OOD_PROBES: dict[str, list[str]] = {
    "en": [
        "what is your name", "how are you today", "who created you", "tell me a joke",
        "what model are you running on", "sing me a song", "what did I have for breakfast",
        "book me a flight to Paris", "what is my password", "shut down the system",
        "what time is it right now", "who won the 2026 election", "write me a python function",
        "what is the weather in Ahmedabad today", "order me a pizza", "what is my bank balance",
        "call my mother", "set an alarm for 7am", "what are you thinking about",
        "are you conscious", "delete all my files", "what is my location right now",
    ],
    "hi": [
        "तुम्हारा नाम क्या है", "तुम कैसे हो", "मुझे एक चुटकुला सुनाओ", "तुम्हें किसने बनाया",
        "मेरा पासवर्ड क्या है", "आज मौसम कैसा है", "मेरे लिए पिज़्ज़ा ऑर्डर करो",
        "अभी समय क्या है", "मुझे एक गाना गाओ", "मेरी माँ को फोन करो",
        "क्या तुम इंसान हो", "मेरे बैंक खाते में कितने पैसे हैं",
    ],
    "gu": [
        "તમારું નામ શું છે", "તમે કેમ છો", "મને એક જોક કહો", "તમને કોણે બનાવ્યા",
        "મારો પાસવર્ડ શું છે", "આજે હવામાન કેવું છે", "મારા માટે પિઝા ઓર્ડર કરો",
        "અત્યારે સમય શું છે", "મને એક ગીત ગાઓ", "મારી માતાને ફોન કરો",
        "શું તમે માણસ છો", "મારા બેંક ખાતામાં કેટલા પૈસા છે",
    ],
    "bn": [
        "তোমার নাম কি", "তুমি কেমন আছো", "আমাকে একটা কৌতুক বলো", "তোমাকে কে বানিয়েছে",
        "আমার পাসওয়ার্ড কি", "আজ আবহাওয়া কেমন", "আমার জন্য পিৎজা অর্ডার করো",
        "এখন কয়টা বাজে", "আমাকে একটা গান গাও", "আমার মাকে ফোন করো",
        "তুমি কি মানুষ", "আমার ব্যাংক অ্যাকাউন্টে কত টাকা আছে",
    ],
    "ta": [
        "உங்கள் பெயர் என்ன", "நீங்கள் எப்படி இருக்கிறீர்கள்", "எனக்கு ஒரு நகைச்சுவை சொல்லுங்கள்",
        "உங்களை யார் உருவாக்கியது", "என் கடவுச்சொல் என்ன", "இன்று வானிலை எப்படி",
        "எனக்கு பீட்சா ஆர்டர் செய்யுங்கள்", "இப்போது மணி என்ன", "எனக்கு ஒரு பாடல் பாடுங்கள்",
        "என் அம்மாவை அழைக்கவும்", "நீங்கள் மனிதரா", "என் வங்கிக் கணக்கில் எவ்வளவு பணம் உள்ளது",
    ],
}


def signals(corpus: Corpus, embedder: object, text: str, lang: str, k: int = 20) -> dict[str, float]:
    """Compute several candidate abstention signals for one query.

    Top-1 cosine is the obvious signal and it turned out to separate badly: measured on this
    corpus, in-domain p50 was 0.7036 against out-of-domain p50 of 0.6132, with p05/p95 ranges that
    almost entirely overlap. Rejecting 95% of out-of-domain traffic required a threshold that also
    refused 78.8% of answerable queries — a gate that says no to everything is not a guardrail.

    The reason is structural, not a tuning failure. A static embedder maps any text to a
    bag-of-subwords centroid, so *every* query has some passage at moderate cosine. The score
    measures topical proximity, which is not the same question as "does the corpus answer this".

    So this evaluates alternatives with a clear hypothesis behind each:

    - `top1`        the baseline.
    - `margin`      top1 minus the mean of the rest of the top-k. An answerable query should have
                    one distinctly best passage; an unanswerable one has a flat field of
                    mediocre matches. This measures peakedness rather than height.
    - `margin_top5` the same idea, tightened to the immediate neighbourhood.
    - `top1_x_margin` height and peakedness together, for queries that need both.

    Which one ships is decided by separation on the two distributions, not by which story is
    prettiest.
    """
    vec = embedder.encode_query(text)  # type: ignore[attr-defined]
    mask = corpus.mask_for_langs(retrieval_langs(lang))
    _rows, scores = corpus.exact_search(vec, k, mask=mask)
    if scores.size == 0:
        return dict.fromkeys(("top1", "margin", "margin_top5", "top1_x_margin"), -1.0)

    top1 = float(scores[0])
    rest = scores[1:]
    margin = top1 - float(rest.mean()) if rest.size else 0.0
    rest5 = scores[1:6]
    margin_top5 = top1 - float(rest5.mean()) if rest5.size else 0.0
    return {
        "top1": top1,
        "margin": margin,
        "margin_top5": margin_top5,
        "top1_x_margin": top1 * margin,
    }


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC via the rank-sum identity — the probability a random in-domain query outscores a random
    out-of-domain one. 0.5 is coin-flip; 1.0 is perfect separation."""
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    values = np.concatenate([pos, neg])
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pick_tau(pos: np.ndarray, neg: np.ndarray, target_rejection: float) -> tuple[float, float, float]:
    """Smallest threshold meeting the target out-of-domain rejection rate.

    Smallest, not largest: among thresholds that satisfy the requirement, the lowest one refuses
    the fewest answerable queries.
    """
    for tau in np.sort(np.unique(np.concatenate([pos, neg]))):
        if float((neg < tau).mean()) >= target_rejection:
            return float(tau), float((neg < tau).mean()), float((pos < tau).mean())
    tau = float(neg.max())
    return tau, float((neg < tau).mean()), float((pos < tau).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--in-domain", type=int, default=500)
    ap.add_argument("--target-rejection", type=float, default=0.95)
    ap.add_argument("--out", type=Path, default=Path("bench/results/scope_calibration.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    corpus = Corpus.load(args.corpus)
    embedder = get_embedder("fast")

    # --- in-domain -------------------------------------------------------------------
    rows = [
        r
        for r in pq.read_table(args.corpus / "queries.parquet").to_pylist()
        if r.get("query") and 3 <= len(r["query"]) <= 300 and not r.get("degenerate", False)
    ]
    step = max(1, len(rows) // args.in_domain)
    in_domain = rows[::step][: args.in_domain]
    log.info("scoring %d in-domain queries", len(in_domain))
    in_sig = [signals(corpus, embedder, r["query"], r["lang"]) for r in in_domain]

    probes = [(t, lg) for lg, texts in OOD_PROBES.items() for t in texts]
    log.info("scoring %d out-of-domain probes", len(probes))
    ood_sig = [signals(corpus, embedder, t, lg) for t, lg in probes]

    names = list(in_sig[0])
    results: dict[str, dict[str, float]] = {}
    for name in names:
        pos = np.array([s[name] for s in in_sig])
        neg = np.array([s[name] for s in ood_sig])
        tau, rejected, refused = pick_tau(pos, neg, args.target_rejection)
        results[name] = {
            "auc": roc_auc(pos, neg),
            "tau_at_target": tau,
            "ood_rejected": rejected,
            "in_domain_refused": refused,
            "in_p50": float(np.percentile(pos, 50)),
            "ood_p50": float(np.percentile(neg, 50)),
        }

    # Rank by AUC: the probability a random answerable query outscores a random unanswerable one.
    # It summarises separation across every threshold instead of privileging one operating point.
    ranked = sorted(results.items(), key=lambda kv: kv[1]["auc"], reverse=True)
    best_name, best = ranked[0]

    print()
    print("=" * 78)
    print("SCOPE GATE CALIBRATION — signal comparison")
    print("=" * 78)
    print(f"in-domain n={len(in_sig)}   out-of-domain n={len(ood_sig)}")
    print()
    print(f"{'signal':<16}{'AUC':>7}{'in p50':>9}{'ood p50':>9}{'tau':>9}{'OOD rej':>9}{'ID refused':>12}")
    print("-" * 78)
    for name, r in ranked:
        print(f"{name:<16}{r['auc']:>7.3f}{r['in_p50']:>9.4f}{r['ood_p50']:>9.4f}"
              f"{r['tau_at_target']:>9.4f}{r['ood_rejected']:>8.1%}{r['in_domain_refused']:>12.1%}")
    print()
    print(f"best separating signal: {best_name}  (AUC {best['auc']:.3f})")

    # An operating point is only worth shipping if it refuses few answerable queries. A gate that
    # blocks most real traffic to catch out-of-domain probes has made the product worse.
    pos = np.array([s[best_name] for s in in_sig])
    neg = np.array([s[best_name] for s in ood_sig])
    print()
    print(f"{'target OOD rejection':<24}{'tau':>10}{'OOD rej':>10}{'ID refused':>13}")
    print("-" * 78)
    operating: list[dict[str, float]] = []
    for target in (0.50, 0.70, 0.80, 0.90, 0.95):
        tau, rej, ref = pick_tau(pos, neg, target)
        operating.append(
            {"target": target, "tau": tau, "ood_rejected": rej, "in_domain_refused": ref}
        )
        print(f"{target:<24.0%}{tau:>10.4f}{rej:>10.1%}{ref:>13.1%}")

    # Ship the strictest threshold whose collateral damage stays under 10% of real queries.
    shippable = [o for o in operating if o["in_domain_refused"] <= 0.10]
    chosen = max(shippable, key=lambda o: o["ood_rejected"]) if shippable else None

    print()
    if chosen:
        print(f"SHIPPING: {best_name} tau={chosen['tau']:.4f} — rejects "
              f"{chosen['ood_rejected']:.1%} of out-of-domain, refuses "
              f"{chosen['in_domain_refused']:.1%} of answerable")
        print(f"  set SHRUTI_SCOPE_SIGNAL={best_name}  SHRUTI_SCOPE_TAU={chosen['tau']:.4f}")
    else:
        print("NO SHIPPABLE THRESHOLD: every operating point refuses >10% of answerable queries.")
        print("  The gate stays uncalibrated and reports itself as such, which is the honest state.")
    print("=" * 78)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "best_signal": best_name,
                "target_rejection": args.target_rejection,
                "signals": results,
                "operating_points": operating,
                "shipped": chosen,
                "n_in_domain": len(in_sig),
                "n_out_of_domain": len(ood_sig),
                "in_domain_hist": np.histogram(pos, bins=40)[0].tolist(),
                "ood_hist": np.histogram(neg, bins=40, range=(float(pos.min()), float(pos.max())))[0].tolist(),
                "hist_range": [float(pos.min()), float(pos.max())],
            },
            indent=2,
        )
    )
    print(f"artifact -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
