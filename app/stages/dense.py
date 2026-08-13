"""Dense vector retrieval — approximate (HNSW) and exact, both kept and both measured.

Keeping an exact path in production is unusual and deliberate. At ~500k passages of 256 dimensions
a brute-force scan is a single BLAS matmul over ~0.5 GB — a few milliseconds, well inside budget.
That makes exact search a real runtime option rather than a debugging aid, and it means the recall
cost of approximate search is something we *measure* on the deployed system instead of inheriting
from a library's README.

Language filtering
------------------
Metadata-aware retrieval routes a query to its own language partition plus English. Two ways to do
that, with an honest trade-off:

- **exact mode** restricts the scan to the partition's rows. Exact by construction, no recall loss.
- **hnsw mode** oversamples the global search and filters afterwards. Faster, but a passage in the
  right language can fall outside the oversampled window, so recall is bounded by the oversample
  factor. `HNSW_OVERSAMPLE` is the knob, and the lab quantifies what it costs.

usearch's own predicate-filtered search is not used: the predicate is invoked from Python for every
candidate visited, and the interpreter overhead dwarfs the search it is filtering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.stages.base import ErrorKind, StageError

log = logging.getLogger("shruti.dense")

# Multiplier applied to k before post-filtering by language in HNSW mode.
HNSW_OVERSAMPLE = 4
# HNSW build parameters. connectivity=16 / expansion_add=128 are the usual quality-side defaults;
# expansion_search is the recall/latency dial and is swept in the lab rather than guessed.
HNSW_CONNECTIVITY = 16
HNSW_EXPANSION_ADD = 128
HNSW_EXPANSION_SEARCH = 64


class DenseIndex:
    """usearch HNSW over the corpus embedding matrix."""

    def __init__(self, index: object, dim: int, n_vectors: int) -> None:
        self._index = index
        self.dim = dim
        self.n_vectors = n_vectors

    @classmethod
    def build(cls, embeddings: np.ndarray, *, expansion_add: int = HNSW_EXPANSION_ADD) -> DenseIndex:
        from usearch.index import Index

        n, dim = embeddings.shape
        log.info("building HNSW over %d vectors (dim=%d)", n, dim)
        index = Index(
            ndim=dim,
            # Vectors are L2-normalised upstream, so inner product ranks identically to cosine
            # while skipping a norm computation per comparison.
            metric="ip",
            dtype="f32",
            connectivity=HNSW_CONNECTIVITY,
            expansion_add=expansion_add,
            expansion_search=HNSW_EXPANSION_SEARCH,
        )
        index.add(np.arange(n, dtype=np.int64), np.asarray(embeddings, dtype=np.float32))
        log.info("HNSW built: %d vectors", len(index))
        return cls(index, dim, n)

    def save(self, path: Path) -> None:
        self._index.save(str(path))  # type: ignore[attr-defined]

    @classmethod
    def load(cls, path: Path, dim: int, n_vectors: int) -> DenseIndex:
        from usearch.index import Index

        if not path.exists():
            raise StageError("dense", ErrorKind.INDEX_ERROR, f"missing HNSW index at {path}")
        index = Index(ndim=dim, metric="ip", dtype="f32")
        # load() reads the graph into RAM; view() would memory-map it. Same reasoning as the
        # embedding matrix in corpus.py: HNSW traversal jumps across the graph, so a mapped index
        # takes page faults at unpredictable points and produces exactly the tail outliers this
        # system exists to avoid. Startup pays a few hundred milliseconds once; every query after
        # it is deterministic.
        index.load(str(path))
        return cls(index, dim, n_vectors)

    def search(
        self, query_vec: np.ndarray, k: int, allowed_rows: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(rows, scores)` descending by similarity.

        Scores are converted back to inner product (`1 - distance`) so they are directly comparable
        with the exact path's output. Mixing a distance and a similarity across the two modes would
        silently invert the scope gate's threshold comparison.
        """
        want = k * HNSW_OVERSAMPLE if allowed_rows is not None else k
        want = min(want, self.n_vectors)
        if want == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        matches = self._index.search(  # type: ignore[attr-defined]
            np.asarray(query_vec, dtype=np.float32), want
        )
        rows = np.asarray(matches.keys, dtype=np.int64)
        scores = 1.0 - np.asarray(matches.distances, dtype=np.float32)

        if allowed_rows is not None:
            mask = np.isin(rows, allowed_rows)
            rows, scores = rows[mask], scores[mask]

        return rows[:k], scores[:k]
