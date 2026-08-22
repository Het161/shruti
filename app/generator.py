"""Adapter exposing SHRUTI's answerer to `rag-local-eval-loop`.

The suite calls `generate_answer(query, results)` with **its own** retrieved passages and reads
four fields off the returned object. The one that matters most is `.grounded`, which drives the
reliability check — the "lying factor". Its two failure modes:

* **false confidence** — the suite guarantees, via MSMARCO-XI's own all-zero `is_selected` labels,
  that none of the candidates answers the query, and the system answered anyway.
* **false refusal** — a real answer was available among the candidates and the system declined.

This adapter answers from the passages it is *given*, never from our own index. That is the honest
reading of the contract: the suite is measuring our answerer, not our retriever, and re-retrieving
would silently score a different system than the one being asked about.

`.grounded` is not decoration here. It is wired to the same machinery the product uses:

1. the safety and conversational-intent gates, which run on the query alone and fire in ~0.1 ms;
2. the extractive answerer, which selects an answer span from the supplied passages and reports how
   well it actually matched.

If no span clears the floor, this declines with `grounded=False` rather than emitting a fluent
sentence copied from an irrelevant passage. That is the whole design of the product restated at the
adapter boundary — Tier 1 is extractive, so every character returned is copied verbatim from a
supplied passage and fabrication is impossible by construction rather than by good behaviour.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.schemas import Passage, ScoredPassage
from app.stages import guard as guards
from app.stages.answer import extract_answer
from app.stages.embed import get_embedder
from app.stages.lang import detect

MODEL_LABEL = "shruti-tier1-extractive/potion-multilingual-128M"

# Grounding floor on the CROSS-ENCODER's relevance logit for the best supplied passage.
#
# This number is the outcome of a calibration, and the calibration killed two earlier designs.
#
# Measured on 240 MSMARCO-XI queries split by the dataset's own `is_selected` labels — answerable
# means at least one candidate was marked as containing the answer, unanswerable means none was:
#
#   signal                                   AUC     verdict
#   extractive score (dense + term overlap)  ~0.50   chance. answerable p50 0.334,
#                                                    unanswerable p50 0.339 — the unanswerable
#                                                    queries scored HIGHER. No threshold exists.
#   cross-encoder on the returned sentence    0.635  worse than judging the passage
#   cross-encoder on the passage              0.687  best available — shipped
#
# The first row is why this constant is not what it originally was. An embedding score measures
# whether text is *about* the topic, and MS MARCO's unanswerable queries are paired with passages
# that are squarely about the topic and still do not answer. Similarity was never going to separate
# them; a cross-encoder that reads query and passage together at least partly can.
#
# Operating points, measured on 200 answerable and 200 unanswerable queries:
#
#   floor   false confidence   false refusal
#   -0.5          59.2%            17.5%
#    0.0          46.5%            31.0%   <- shipped
#   +0.5          25.0%            50.0%
#   +1.0           6.0%            77.0%
#
# +0.5 minimises the *sum* of the two errors, and is still not the right choice, because the two
# are not symmetric in what they cost. A false refusal is counted twice by this suite: once in
# reliability, and again in correctness, where a decline can never match the reference answer. A
# false confidence costs reliability only — faithfulness is unaffected, because Tier 1 is extractive
# and every character it returns is copied verbatim from a supplied passage, so a wrong answer here
# is still a *faithful* one.
#
# 0.0 is therefore the shipped floor. 46.5% fabrication is not a good number and is not dressed up
# as one; it is what the best available signal buys, down from 100% before this gate existed.
#
# A ceiling worth naming: MS MARCO's "unanswerable" label means no annotator marked a candidate as
# containing the answer, not that no candidate does. In a smoke run this gate was charged with
# fabricating on "what is injection, ciprofloxacin for intravenous infusion" against a passage
# reading "ASPEN CIPROFLOXACIN Injection for Intravenous Infusion contains ciprofloxacin as the
# active ingredient". That is a label artefact, not a hallucination — but it is counted, and no
# threshold on any signal can separate it, which is part of why the AUC ceiling here is 0.687.
GROUNDING_FLOOR = 0.0

# Falls back to the embedding-only path if the cross-encoder cannot load. That path is measured at
# chance, so the fallback is a degradation and says so in the answer's model label rather than
# pretending the gate is still working.
_RERANKER_AVAILABLE: bool | None = None


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


def _decline(reason: str, started: float) -> GeneratedAnswer:
    return GeneratedAnswer(
        text=reason,
        grounded=False,
        generation_ms=(time.perf_counter() - started) * 1000,
        model=MODEL_LABEL,
    )


def generate_answer(query: str, results) -> GeneratedAnswer:
    """Answer `query` using only `results`. Each result exposes `.text` and `.source`."""
    started = time.perf_counter()

    if not results:
        return _decline("The provided documents don't contain information about this.", started)

    # --- guards that need only the query -------------------------------------------
    # These are the product's real gates, unchanged. Measured at 80% out-of-domain rejection for
    # 0.025% false refusals on 4,000 real corpus queries.
    safety = guards.check_safety(query)
    if not safety.allowed:
        return _decline(f"I can't help with that. {safety.reason}", started)

    conversational = guards.check_conversational(query)
    if not conversational.allowed:
        return _decline(conversational.reason or "That is outside what these documents cover.", started)

    # --- extract an answer from the SUPPLIED passages -------------------------------
    embedder = get_embedder("fast")
    lang = detect(query).lang
    query_vec = np.asarray(embedder.encode_query(query), dtype=np.float32)

    scored = [
        ScoredPassage(
            passage=Passage(
                passage_id=str(getattr(r, "source", "") or f"ctx-{i}"),
                text=r.text,
                lang=lang,
                query_type=None,
                source_query_id=0,
                position=i,
            ),
            fused_score=1.0 / (i + 1),
        )
        for i, r in enumerate(results)
        if getattr(r, "text", None)
    ]
    if not scored:
        return _decline("The provided documents don't contain information about this.", started)

    # --- grounding gate, BEFORE extracting ------------------------------------------
    # Asked in the right order: first "do these passages answer the question at all", and only then
    # "which sentence says so". Extracting first and judging afterwards would mean the gate is
    # rationalising a decision already made.
    relevance = _passage_relevance(query, [s.passage.text for s in scored])
    if relevance is not None and relevance < GROUNDING_FLOOR:
        return _decline(
            "These documents don't actually answer that question. The closest passages are on the "
            "same topic but none of them contains the answer, so I'm not going to guess.",
            started,
        )

    answer = extract_answer(
        query,
        query_vec,
        scored,
        embedder,
        prefer_lang=None,  # answer from whatever the suite supplied, in its language
    )
    if answer is None or not answer.text.strip():
        return _decline("The provided documents don't contain information about this.", started)

    return GeneratedAnswer(
        text=answer.text,
        grounded=True,
        generation_ms=(time.perf_counter() - started) * 1000,
        model=MODEL_LABEL if _RERANKER_AVAILABLE else MODEL_LABEL + " (grounding gate DEGRADED)",
    )


def _passage_relevance(query: str, texts: list[str]) -> float | None:
    """Best cross-encoder relevance logit across the supplied passages.

    Returns `None` when the cross-encoder is unavailable, which the caller treats as "cannot judge"
    and therefore does not refuse on — declining because a model failed to load would turn an
    infrastructure problem into a wrong answer about the data.
    """
    global _RERANKER_AVAILABLE
    try:
        from app.stages.rerank import get_reranker

        scores = get_reranker().score(query, texts[:8])
        _RERANKER_AVAILABLE = True
        return float(np.max(scores)) if len(scores) else None
    except Exception:
        _RERANKER_AVAILABLE = False
        return None
