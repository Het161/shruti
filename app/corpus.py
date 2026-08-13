"""In-process corpus: passages, embeddings, and the vector indexes over them.

Loaded once at startup, then read-only for the life of the process. Nothing here touches the disk
or the network during a request — after warmup, every byte a query needs is resident.

Two decisions worth stating
---------------------------

**Passage text is held as an Arrow array, not a Python list.** 500k Python `str` objects carry
roughly 50 bytes of object overhead each on top of their UTF-8 payload, and for Indic scripts
CPython's internal representation widens to 2 or 4 bytes per character. Arrow keeps the same text
as one contiguous UTF-8 buffer with an offsets array — several times smaller, and conversion to
Python `str` happens only for the handful of passages that actually reach an answer.

**Embeddings default to float32 rather than int8.** At 256 dimensions the whole matrix is ~0.5 GB,
which fits the Space comfortably, and a float32 matrix means exact search is a single BLAS call
instead of a chunked dequantise-and-multiply loop. int8 is supported for when size genuinely
demands it, and the loader handles either transparently — but paying 4× the memory to keep the
verification path a one-liner is the right trade at this corpus scale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from app.stages.base import ErrorKind, StageError

log = logging.getLogger("shruti.corpus")

# Chunk size for exact search when embeddings are int8 and must be widened block by block.
_EXACT_CHUNK = 65_536


@dataclass(frozen=True, slots=True)
class PassageRef:
    """A passage resolved out of the columnar store, ready to hand to a pydantic model."""

    passage_id: str
    text: str
    lang: str
    query_type: str | None
    source_query_id: int
    position: int


class Corpus:
    """Columnar passage store plus the embedding matrix aligned to it.

    Row order is the contract: index `i` in `embeddings` is the passage at index `i` in every
    column here. The loader verifies that alignment rather than trusting it, because a silent
    off-by-one would produce confidently wrong citations — the single worst failure mode this
    system has.
    """

    def __init__(
        self,
        *,
        passage_ids: Any,
        texts: Any,
        langs: Any,
        query_types: Any,
        source_query_ids: np.ndarray,
        positions: np.ndarray,
        embeddings: np.ndarray,
        quant_scale: float,
        manifest: dict[str, Any],
    ) -> None:
        self._passage_ids = passage_ids
        self._texts = texts
        self._langs = langs
        self._query_types = query_types
        self._source_query_ids = source_query_ids
        self._positions = positions
        self.embeddings = embeddings
        self.quant_scale = quant_scale
        self.manifest = manifest

        self.n_passages = len(passage_ids)
        self.dim = int(embeddings.shape[1]) if embeddings.size else 0
        self._id_to_row: dict[str, int] = {
            pid: i for i, pid in enumerate(passage_ids.to_pylist())
        }
        # Language partitions, precomputed. Metadata-aware retrieval filters by language on every
        # query, so building these once at startup keeps that filter free at request time.
        by_lang: dict[str, list[int]] = {}
        for i, lg in enumerate(langs.to_pylist()):
            by_lang.setdefault(lg, []).append(i)
        self.lang_rows: dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.int64) for k, v in by_lang.items()
        }
        # Boolean masks alongside the row-index arrays. Exact search filters by masking a full
        # score vector rather than by fancy-indexing the embedding matrix: `emb[rows]` would
        # materialise a fresh ~200 MB copy on every query, which costs far more than the search
        # itself. A bool mask is one byte per passage and the matmul happens anyway.
        self.lang_mask: dict[str, np.ndarray] = {}
        for k, rows in self.lang_rows.items():
            mask = np.zeros(self.n_passages, dtype=bool)
            mask[rows] = True
            self.lang_mask[k] = mask

    def rows_for_langs(self, langs: list[str]) -> np.ndarray | None:
        """Row indices for the union of `langs`. `None` means no restriction."""
        present = [self.lang_rows[lg] for lg in langs if lg in self.lang_rows]
        if not present or len(present) == len(self.lang_rows):
            return None
        return np.concatenate(present)

    def mask_for_langs(self, langs: list[str]) -> np.ndarray | None:
        """Boolean mask for the union of `langs`. `None` means no restriction."""
        present = [self.lang_mask[lg] for lg in langs if lg in self.lang_mask]
        if not present or len(present) == len(self.lang_mask):
            return None
        out = present[0].copy()
        for m in present[1:]:
            out |= m
        return out

    # -- loading ------------------------------------------------------------------------

    @classmethod
    def load(cls, artifact_dir: Path) -> Corpus:
        artifact_dir = Path(artifact_dir)
        manifest_path = artifact_dir / "manifest.json"
        passages_path = artifact_dir / "passages.parquet"
        emb_path = artifact_dir / "embeddings.npy"

        for p in (manifest_path, passages_path, emb_path):
            if not p.exists():
                raise StageError(
                    "corpus",
                    ErrorKind.INDEX_ERROR,
                    f"missing artifact {p}. Run scripts/build_corpus.py then scripts/embed_corpus.py",
                )

        manifest = json.loads(manifest_path.read_text())
        table = pq.read_table(passages_path)

        # mmap_mode keeps the matrix backed by the page cache instead of copying it into the heap;
        # warmup then faults in the pages that queries actually touch.
        embeddings = np.load(emb_path, mmap_mode="r")

        n_rows = table.num_rows
        if embeddings.shape[0] != n_rows:
            raise StageError(
                "corpus",
                ErrorKind.INDEX_ERROR,
                f"row misalignment: {n_rows} passages but {embeddings.shape[0]} embeddings. "
                "Rebuild embeddings against this exact passages.parquet.",
            )

        quant_scale = float(manifest.get("embeddings", {}).get("quant_scale", 1.0))
        log.info(
            "corpus loaded: %d passages, dim=%d, dtype=%s",
            n_rows,
            embeddings.shape[1],
            embeddings.dtype,
        )

        return cls(
            passage_ids=table.column("passage_id"),
            texts=table.column("text"),
            langs=table.column("lang"),
            query_types=table.column("query_type"),
            source_query_ids=table.column("source_query_id").to_numpy(zero_copy_only=False),
            positions=table.column("position").to_numpy(zero_copy_only=False),
            embeddings=embeddings,
            quant_scale=quant_scale,
            manifest=manifest,
        )

    # -- access -------------------------------------------------------------------------

    def get(self, row: int) -> PassageRef:
        return PassageRef(
            passage_id=self._passage_ids[row].as_py(),
            text=self._texts[row].as_py(),
            lang=self._langs[row].as_py(),
            query_type=self._query_types[row].as_py(),
            source_query_id=int(self._source_query_ids[row]),
            position=int(self._positions[row]),
        )

    def text_at(self, row: int) -> str:
        return self._texts[row].as_py()  # type: ignore[no-any-return]

    def row_of(self, passage_id: str) -> int | None:
        return self._id_to_row.get(passage_id)

    def iter_texts(self) -> list[str]:
        """Materialise all text. Used only by offline index builders, never at request time."""
        return self._texts.to_pylist()  # type: ignore[no-any-return]

    @property
    def languages(self) -> list[str]:
        return sorted(self.lang_rows)

    # -- exact search -------------------------------------------------------------------

    def exact_search(
        self, query_vec: np.ndarray, k: int, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Brute-force cosine search. Returns `(rows, scores)` sorted descending.

        This is the ground truth the HNSW index is checked against. At this corpus size it is only
        a few milliseconds, so it is a genuine runtime option rather than a debug-only path — and
        having both means the recall cost of approximate search is a measured number, not an
        assumption.

        `mask` restricts the search to selected rows. It is applied to the *scores* rather than to
        the embedding matrix: slicing `emb[rows]` would copy hundreds of megabytes per query, while
        masking a float32 score vector costs a few hundred microseconds over memory we had to touch
        regardless.
        """
        emb = self.embeddings
        q = np.ascontiguousarray(query_vec, dtype=np.float32)

        if emb.dtype == np.float32:
            scores = np.asarray(emb @ q, dtype=np.float32)
        else:
            # int8 path: widen block by block so the temporary never exceeds one chunk.
            scores = np.empty(emb.shape[0], dtype=np.float32)
            for start in range(0, emb.shape[0], _EXACT_CHUNK):
                block = np.asarray(emb[start : start + _EXACT_CHUNK], dtype=np.float32)
                block *= self.quant_scale
                scores[start : start + block.shape[0]] = block @ q

        if mask is not None:
            # -inf rather than 0: cosine similarity is legitimately negative, and zeroing would
            # rank an excluded passage above a genuinely dissimilar included one.
            scores = np.where(mask, scores, -np.inf)
            available = int(mask.sum())
        else:
            available = scores.shape[0]

        k = min(k, available)
        if k == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        # argpartition is O(n) against argsort's O(n log n); only the top-k slice is then sorted.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return top.astype(np.int64), scores[top].astype(np.float32)
