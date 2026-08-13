"""Reciprocal Rank Fusion of the dense and lexical result lists.

Why RRF rather than a weighted score blend
------------------------------------------
BM25 scores are unbounded and corpus-dependent; cosine similarities live in [-1, 1]. Blending them
requires normalising two distributions whose shapes shift with the query, and any fixed weight is
really a hidden hyperparameter fitted to whichever queries were looked at last.

RRF ignores scores entirely and fuses *ranks*: each list contributes `1/(k + rank)`. It is
scale-free, needs no per-retriever calibration, and is consistently competitive with tuned blends.
The single constant `k` damps the influence of top ranks — at k=60 the gap between rank 1 and rank
2 is small enough that one retriever cannot unilaterally dominate a fused list.

A consequence worth stating plainly: **fused scores are not similarities.** An RRF score of 0.03 is
meaningful only relative to other RRF scores for the same query. The scope gate therefore
thresholds the top *dense cosine* score, not the fused score, because only the former has a stable
interpretation across queries. Thresholding RRF would produce a gate whose behaviour drifts with
how many retrievers happened to return results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FusedHit:
    row: int
    fused_score: float
    dense_score: float | None
    lexical_score: float | None
    dense_rank: int | None
    lexical_rank: int | None


def reciprocal_rank_fusion(
    dense_rows: np.ndarray,
    dense_scores: np.ndarray,
    lexical_rows: np.ndarray,
    lexical_scores: np.ndarray,
    *,
    k: int = 60,
    top_n: int | None = None,
) -> list[FusedHit]:
    """Fuse two ranked lists. Returns hits sorted by fused score, descending.

    Component scores and ranks are preserved on every hit rather than discarded, because the UI
    shows why a passage surfaced and the lab needs per-retriever attribution to explain results.
    """
    contrib: dict[int, float] = {}
    d_score: dict[int, float] = {}
    l_score: dict[int, float] = {}
    d_rank: dict[int, int] = {}
    l_rank: dict[int, int] = {}

    for rank, (row, score) in enumerate(zip(dense_rows.tolist(), dense_scores.tolist(), strict=True), start=1):
        contrib[row] = contrib.get(row, 0.0) + 1.0 / (k + rank)
        d_score[row] = score
        d_rank[row] = rank

    for rank, (row, score) in enumerate(zip(lexical_rows.tolist(), lexical_scores.tolist(), strict=True), start=1):
        contrib[row] = contrib.get(row, 0.0) + 1.0 / (k + rank)
        l_score[row] = score
        l_rank[row] = rank

    hits = [
        FusedHit(
            row=row,
            fused_score=score,
            dense_score=d_score.get(row),
            lexical_score=l_score.get(row),
            dense_rank=d_rank.get(row),
            lexical_rank=l_rank.get(row),
        )
        for row, score in contrib.items()
    ]
    # Tie-break on dense score: RRF produces exact ties whenever two passages appear at the same
    # rank in one list and are absent from the other, which at small k is common rather than rare.
    hits.sort(key=lambda h: (h.fused_score, h.dense_score or -1.0), reverse=True)
    return hits[:top_n] if top_n is not None else hits
