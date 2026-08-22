"""Adapter exposing SHRUTI's embedder to `rag-local-eval-loop`.

The evaluation suite imports `embed`, `embed_one` and `get_model` from this module, builds its own
throwaway index over MSMARCO-XI, and scores retrieval against the dataset's own `is_selected`
labels. It never touches our production index — so this file deliberately loads **only the
embedding model**, not the 310k-passage corpus, the HNSW graph, or the BM25 index. Pulling those in
would cost ~500 MB and several seconds for data the suite does not use.

The model is the real shipped one: `potion-multilingual-128M`, the same static Model2Vec lookup
that serves the live deployment. Whatever the suite measures here is what production does.

Vectors are L2-normalised, so inner product and cosine coincide — the suite is free to use either.
"""

from __future__ import annotations

import numpy as np

from app.stages.embed import get_embedder

# 256, from `potion-multilingual-128M`. Read off the model rather than hardcoded, and the suite
# infers it empirically from a real `embed_one` call anyway.
_DIM: int | None = None


def get_model():
    """Load the embedding model. Called once by the suite; the side effect is the point."""
    global _DIM
    embedder = get_embedder("fast")
    _DIM = embedder.dim
    return embedder


def embed_one(text: str) -> np.ndarray:
    """Embed a single string. Returns float32, shape (dim,), L2-normalised."""
    return np.asarray(get_model().encode_query(text), dtype=np.float32)


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch. Returns float32, shape (len(texts), dim), rows L2-normalised.

    An empty input returns a correctly-shaped empty array rather than raising — the suite may probe
    with one, and a shape-correct empty result keeps `np.vstack` downstream happy.
    """
    embedder = get_model()
    if not texts:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    return np.asarray(embedder.encode_batch(texts), dtype=np.float32)
