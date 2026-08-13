"""BM25 lexical retrieval with Indic-aware tokenisation.

Why lexical retrieval earns its place next to a dense retriever
--------------------------------------------------------------
Dense embeddings are strong on paraphrase and weak on exact tokens — names, numbers, dates, rare
entities. MS MARCO has a `query_type` field with NUMERIC, PERSON, and LOCATION among its values,
which is close to a list of the cases where dense retrieval alone underperforms. BM25 covers
exactly that gap, costs a few milliseconds, and needs no model.

Tokenisation is the whole problem for Indic text
------------------------------------------------
Off-the-shelf BM25 tokenisers assume whitespace-and-ASCII. Applied to Devanagari or Gujarati they
degrade badly, for three specific reasons this module handles:

1. **Danda punctuation.** `।` (U+0964) and `॥` (U+0965) end sentences in Indic scripts. ASCII
   punctuation strippers leave them glued to the preceding word, so `दिल्ली।` and `दिल्ली` become
   different terms.
2. **Combining marks.** Vowel signs and nuqta are separate codepoints. Without NFC normalisation
   the same word tokenises two ways depending on how the translator emitted it.
3. **Zero-width joiners.** ZWJ/ZWNJ (U+200C/U+200D) control ligature formation and are invisible,
   so they silently fragment terms.

English stemming is applied only to Latin-script tokens. There is no reliable light stemmer here
for most of these scripts, and a wrong stemmer is worse than none — it conflates distinct words.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np

log = logging.getLogger("shruti.lexical")

# Indic danda and double danda, plus the usual ASCII punctuation.
_DANDA = "।॥"
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")
_PUNCT = re.compile(rf"[{re.escape(_DANDA)}!-/:-@\[-`{{-~“”‘’—–…]")
_WS = re.compile(r"\s+")
_LATIN = re.compile(r"^[a-z0-9]+$")

# Minimal English stopword list. Deliberately short: aggressive stopword removal hurts MS MARCO,
# where queries are natural-language questions and words like "how" and "what" carry intent that
# query_type classification depends on.
_STOP = frozenset(
    ["a", "an", "the", "of", "to", "in", "is", "are", "was", "were", "and", "or", "for", "on", "at", "by", "with", "as", "from", "that", "this", "it", "be", "been"]
)


def tokenize(text: str) -> list[str]:
    """Script-aware tokenisation shared by index build and query time.

    Index and query MUST tokenise identically or BM25 silently retrieves nothing; keeping this in
    one function is the enforcement mechanism.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _PUNCT.sub(" ", text.lower())
    toks = [t for t in _WS.split(text) if t]
    return [t for t in toks if t not in _STOP and len(t) > 1]


class LexicalIndex:
    """BM25 over the corpus. Built offline, memory-mapped at startup."""

    def __init__(self, retriever: object, n_docs: int) -> None:
        self._retriever = retriever
        self.n_docs = n_docs

    @classmethod
    def build(cls, texts: list[str]) -> LexicalIndex:
        import bm25s

        log.info("tokenising %d passages for BM25", len(texts))
        corpus_tokens = [tokenize(t) for t in texts]
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        log.info("BM25 index built")
        return cls(retriever, len(texts))

    def save(self, path: str) -> None:
        self._retriever.save(path)  # type: ignore[attr-defined]

    @classmethod
    def load(cls, path: str, n_docs: int) -> LexicalIndex:
        import bm25s

        retriever = bm25s.BM25.load(path, mmap=True)
        return cls(retriever, n_docs)

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return `(rows, scores)` for the top-k BM25 matches, descending.

        An empty result is a normal outcome, not an error: a query whose every token is unseen has
        no lexical evidence, and the scope gate should hear that as silence rather than have it
        papered over with an exception.
        """
        tokens = tokenize(query)
        if not tokens:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        k = min(k, self.n_docs)
        rows, scores = self._retriever.retrieve(  # type: ignore[attr-defined]
            [tokens], k=k, show_progress=False
        )
        return (
            np.asarray(rows[0], dtype=np.int64),
            np.asarray(scores[0], dtype=np.float32),
        )
