"""Tier 1 — the extractive answerer.

This stage is why the sub-200ms claim can be made honestly.

The usual shape of a RAG system is retrieve-then-generate, which means the answer latency is
whatever the LLM provider's time-to-first-token happens to be that second — typically 200–800ms
before a single word appears, and entirely outside our control. No amount of retrieval optimisation
recovers that.

So the answer is produced in two tiers. Tier 1 selects the best answer-bearing sentences from the
retrieved passages and returns them immediately, with citations. It is:

- **fast** — sentence scoring is a handful of static embeddings and a token-overlap count, single-
  digit milliseconds;
- **grounded by construction** — every character is copied verbatim from a retrieved passage, so
  hallucination is not mitigated here, it is impossible;
- **a real answer** — not a placeholder. MS MARCO passages are answer-bearing by design, and the
  dataset's own `Answer` field is usually a light rewrite of one passage sentence. Extraction is
  well matched to this corpus rather than a fallback we settled for.

Tier 2 then streams in and expands it. If Tier 2 is slow, fails, or flunks the grounding check,
Tier 1 stands alone and the user still has a complete cited answer.

Sentence selection combines two signals because each fails differently: embedding similarity
handles paraphrase but will happily rank a topically-related sentence that answers nothing, while
query-term overlap catches the literal answer span but is blind to restatement. Their product-free
weighted sum, plus a `query_type` prior, is what selects the span.
"""

from __future__ import annotations

import re

import numpy as np

from app.schemas import Citation, ExtractiveAnswer, ScoredPassage
from app.stages.lexical import tokenize

# Sentence boundaries: Indic danda and double danda, ASCII terminators, and newlines.
# The lookahead on `.` avoids splitting decimals ("3.5") and common abbreviations ("U.S."), which
# otherwise shatter NUMERIC answers — the exact query type where the number IS the answer.
_SENT_BOUNDARY = re.compile(
    r"(?<=[।॥!?])\s+"
    r"|(?<=[.])\s+(?=[A-Zऀ-ൿ])"
    r"|\n+"
)

MIN_SENT_CHARS = 25
MAX_ANSWER_SENTS = 2

# Weights for the two selection signals. Dense similarity leads because it survives translation and
# paraphrase, which is the common case in a translated corpus; lexical overlap is the corrective
# term that keeps literal answer spans from being outranked by fluent-but-empty sentences.
W_DENSE = 0.65
W_LEXICAL = 0.35

# `query_type` priors. MS MARCO labels the expected answer shape, which is free supervision: a
# NUMERIC query wants the sentence containing a number, and rewarding that directly is both cheaper
# and more reliable than hoping the embedding encodes it.
_NUMERIC = re.compile(r"\d")
_TYPE_BONUS = 0.12


def split_sentences(text: str) -> list[str]:
    """Split into sentences, merging fragments below the minimum length.

    Merging matters: an over-eager split produces a 12-character 'sentence' that scores well on
    lexical overlap purely because it is short, and returning it as the answer looks broken.
    """
    raw = [s.strip() for s in _SENT_BOUNDARY.split(text) if s and s.strip()]
    if not raw:
        return []

    merged: list[str] = []
    for sent in raw:
        if merged and len(merged[-1]) < MIN_SENT_CHARS:
            merged[-1] = f"{merged[-1]} {sent}"
        else:
            merged.append(sent)
    # A trailing fragment has nothing after it to merge into, so fold it backwards. Pop into a
    # local first: doing it inline would resolve the assignment index against the already-shortened
    # list, which is off by one and raises outright at length 2.
    if len(merged) > 1 and len(merged[-1]) < MIN_SENT_CHARS:
        tail = merged.pop()
        merged[-1] = f"{merged[-1]} {tail}"
    return merged


def _lexical_overlap(query_tokens: set[str], sentence: str) -> float:
    """Containment of query terms in the sentence, in [0, 1].

    Containment rather than Jaccard: the sentence is typically far longer than the query, and
    Jaccard would penalise a sentence for carrying the surrounding context that makes it a good
    answer.
    """
    if not query_tokens:
        return 0.0
    sent_tokens = set(tokenize(sentence))
    return len(query_tokens & sent_tokens) / len(query_tokens)


def extract_answer(
    query: str,
    query_vec: np.ndarray,
    passages: list[ScoredPassage],
    embedder: object,
    *,
    query_type: str | None = None,
    prefer_lang: str | None = None,
    max_sentences: int = MAX_ANSWER_SENTS,
) -> ExtractiveAnswer | None:
    """Select the best answer span from retrieved passages.

    Returns `None` when nothing clears the floor — an empty answer is a valid, honest outcome, and
    the caller surfaces it through the guard rather than inventing text.
    """
    if not passages:
        return None

    # Restrict to the user's language when that language is present among the results. Answering a
    # Gujarati question with an English sentence is a worse experience than a slightly weaker
    # Gujarati one, and the corpus almost always carries both.
    pool = passages
    if prefer_lang:
        same_lang = [p for p in passages if p.passage.lang == prefer_lang]
        if same_lang:
            pool = same_lang

    query_tokens = set(tokenize(query))
    wants_number = (query_type or "").upper() == "NUMERIC"

    candidates: list[tuple[float, str, str]] = []  # (score, sentence, passage_id)
    for scored in pool:
        sentences = split_sentences(scored.passage.text)
        if not sentences:
            continue

        # One batched embed call per passage rather than per sentence: with a static model the
        # per-call overhead dominates the arithmetic, so batching is most of the speed here.
        vecs = embedder.encode_batch(sentences)  # type: ignore[attr-defined]
        sims = np.asarray(vecs, dtype=np.float32) @ np.asarray(query_vec, dtype=np.float32)

        for sent, sim in zip(sentences, sims.tolist(), strict=True):
            score = W_DENSE * float(sim) + W_LEXICAL * _lexical_overlap(query_tokens, sent)
            if wants_number and _NUMERIC.search(sent):
                score += _TYPE_BONUS
            candidates.append((score, sent, scored.passage.passage_id))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0], reverse=True)
    chosen = candidates[:max_sentences]

    # Keep a second sentence only if it is close in quality to the first. A weak second sentence
    # dilutes a good answer, and terseness is the correct default for a spoken interface.
    if len(chosen) > 1 and chosen[1][0] < chosen[0][0] * 0.8:
        chosen = chosen[:1]

    # Restore reading order so multi-sentence answers are not scrambled by score order.
    order = {id(c): i for i, c in enumerate(candidates)}
    chosen.sort(key=lambda c: order[id(c)])

    marker_of: dict[str, int] = {}
    parts: list[str] = []
    citations: list[Citation] = []
    for _score, sent, pid in chosen:
        if pid not in marker_of:
            marker_of[pid] = len(marker_of) + 1
        marker = marker_of[pid]
        parts.append(f"{sent} [{marker}]")
        citations.append(Citation(marker=marker, passage_id=pid))

    text = " ".join(parts)
    lang = next((p.passage.lang for p in pool if p.passage.passage_id in marker_of), "en")

    return ExtractiveAnswer(
        text=text,
        lang=lang,
        citations=citations,
        source_passage_ids=list(marker_of),
    )
