"""int8 static embedder — the fast lane at a quarter of the memory.

Why this exists
---------------
`potion-multilingual-128M` holds a 500,353 x 256 float32 lookup table: **512 MB**, and loading it
peaks around 1.19 GB. Measured, that is the entire memory cost of this service — the 58k-passage
corpus and the BM25 index together add under 40 MB. On a 512 MB free tier the model alone is the
blocker, and no amount of shrinking the corpus would have fixed it.

Quantising the table to int8 takes it to **128 MB**, which is what makes card-free hosting possible
at all.

Why the quality loss is smaller than the numbers suggest
--------------------------------------------------------
Mean absolute quantisation error on the table is 0.0446 against a standard deviation of 0.857 —
about 5% relative. But a Model2Vec embedding is the **mean** of its tokens' rows, and quantisation
error is zero-mean and independent per element, so averaging over k tokens shrinks it by roughly
sqrt(k). L2 normalisation afterwards removes any residual scale drift. The measured effect on
MRR@10 is reported in docs/BUILD_LOG.md rather than argued for here.

The arithmetic is reimplemented rather than reused because `StaticModel.encode` requires the
float32 table to exist. This class reads the model's own config to confirm the pooling it assumes
is the pooling the model was distilled with — plain mean, no SIF or Zipf weighting — and refuses to
load if that ever changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.stages.base import ErrorKind, StageError

log = logging.getLogger("shruti.embed_int8")


class Int8StaticEmbedder:
    """Model2Vec static embeddings from an int8 lookup table."""

    lane = "fast"

    def __init__(self, model_dir: str | Path) -> None:
        from tokenizers import Tokenizer

        d = Path(model_dir)
        meta_path = d / "int8_meta.json"
        if not meta_path.exists():
            raise StageError(
                "embed", ErrorKind.INDEX_ERROR,
                f"no int8 model at {d}; build it with scripts/quantize_model.py",
            )
        meta = json.loads(meta_path.read_text())

        # Guard the assumption rather than trusting it. If a future model is distilled with SIF or
        # Zipf weighting, plain mean pooling would be silently wrong — producing embeddings that
        # look fine and retrieve badly, which is the worst kind of failure.
        if meta.get("apply_zipf") or meta.get("sif_coefficient"):
            raise StageError(
                "embed", ErrorKind.INTERNAL,
                "model uses SIF/Zipf weighting; this loader implements plain mean pooling only",
            )

        self._table = np.load(d / "embedding_int8.npy")
        self.scale = float(meta["scale"])
        self.dim = int(meta["dim"])
        self.normalize = bool(meta.get("normalize", True))
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        log.info(
            "int8 embedder ready: %s table, dim=%d, %.0f MB",
            self._table.shape, self.dim, self._table.nbytes / 1e6,
        )

    def _pool(self, ids: list[int]) -> np.ndarray:
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        # Only the handful of rows this text actually uses are widened to float32 — typically ten
        # or so out of half a million. Widening the whole table would defeat the entire point.
        rows = self._table[ids].astype(np.float32)
        vec = rows.mean(axis=0) * self.scale
        if self.normalize:
            n = float(np.linalg.norm(vec))
            if n > 0:
                vec /= n
        return vec.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._pool(self._tok.encode(text, add_special_tokens=False).ids)

    def encode_batch(self, texts: list[str], batch_size: int = 1024) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for j, enc in enumerate(self._tok.encode_batch(chunk, add_special_tokens=False)):
                out[i + j] = self._pool(enc.ids)
        return out
