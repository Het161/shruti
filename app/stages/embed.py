"""Query embedding — the fast lane and the quality lane.

Why this stage never makes a network call
-----------------------------------------
An embedding API round-trip is 50–300ms depending on where the provider is. The entire answer
budget is 200ms. So a hosted embedding endpoint does not merely make the target harder — it makes
it unreachable, regardless of how fast everything else is. Both lanes therefore run in-process,
loaded once at startup.

The two lanes
-------------
**Fast (`potion-multilingual-128M`)** — a Model2Vec static model: token embeddings distilled from
bge-m3, pooled by lookup and averaging. There is no transformer forward pass, so a query embeds in
tens of microseconds on CPU. 256 dimensions, 101 languages, unit-normalised at source.

**Quality (`multilingual-e5-small` via ONNX int8)** — a real forward pass, tens of milliseconds on
CPU. Loaded lazily: most deployments never select it, and paying its load cost at startup would
slow every cold start for a lane that is off by default.

Which one ships is decided by the chunking lab's quality/latency table, not by preference. The
toggle stays in the UI either way, because a measured trade-off the judges can flip is worth more
than an assertion about which is better.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import numpy as np

from app.stages.base import ErrorKind, StageError

log = logging.getLogger("shruti.embed")

FAST_MODEL_ID = "minishlab/potion-multilingual-128M"
QUALITY_MODEL_ID = "intfloat/multilingual-e5-small"


class Embedder(Protocol):
    """What every lane must provide."""

    lane: str
    dim: int

    def encode_query(self, text: str) -> np.ndarray: ...
    def encode_batch(self, texts: list[str], batch_size: int = 1024) -> np.ndarray: ...


class FastEmbedder:
    """Model2Vec static embeddings. The default lane."""

    lane = "fast"

    def __init__(self, model_id: str = FAST_MODEL_ID) -> None:
        from model2vec import StaticModel

        log.info("loading fast embedder %s", model_id)
        self._model = StaticModel.from_pretrained(model_id)
        probe = self._model.encode(["warm"])
        self.dim = int(np.asarray(probe).shape[-1])
        log.info("fast embedder ready, dim=%d", self.dim)

    def encode_query(self, text: str) -> np.ndarray:
        """Embed a single query.

        Returned as float32 and L2-normalised so that downstream cosine similarity is a plain dot
        product — normalising here once is cheaper than normalising inside every scoring loop.
        """
        vec = np.asarray(self._model.encode([text]), dtype=np.float32)[0]
        return _l2_normalize(vec)

    def encode_batch(self, texts: list[str], batch_size: int = 1024) -> np.ndarray:
        """Encode many texts.

        `use_multiprocessing=False` is not a default worth inheriting here. model2vec spawns
        workers above 10,000 inputs, and on macOS (spawn, not fork) each worker re-imports the
        module and materialises its own slice of the input list. Embedding 310k passages on an 8 GB
        machine this way drove swap to 8.8 GB, load average to 28, and left the workers at ~16% CPU
        thrashing on page-ins — it did not finish in 13 minutes.

        Single-process throughput is ~16,000 passages/s, so the same job takes about 20 seconds.
        The parallelism was a pessimisation, not an optimisation: this is a memory-bandwidth-bound
        lookup, and handing it more processes only multiplies the working set.
        """
        vecs = np.asarray(
            self._model.encode(texts, batch_size=batch_size, use_multiprocessing=False),
            dtype=np.float32,
        )
        return _l2_normalize_rows(vecs)


class QualityEmbedder:
    """multilingual-e5-small through ONNX Runtime, int8-quantised.

    e5 was trained with asymmetric prefixes: queries are prefixed `query: ` and documents
    `passage: `. Omitting them measurably degrades retrieval, so the prefix is applied here rather
    than left to callers to remember.
    """

    lane = "quality"

    def __init__(self, model_id: str = QUALITY_MODEL_ID) -> None:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover - optional extra
            raise StageError(
                "embed",
                ErrorKind.INTERNAL,
                "quality lane requires the 'quality' extra: pip install -e '.[quality]'",
                cause=e,
            ) from e

        log.info("loading quality embedder %s", model_id)

        # The repo's int8 export is `model_qint8_avx512_vnni.onnx` — quantised for AVX512-VNNI
        # specifically, which is not present on every CPU (and not on ARM at all). So it is tried
        # first and fp32 is the fallback, rather than hard-coding either.
        #
        # An earlier version of this file requested `onnx/model_int8.onnx`, which does not exist in
        # the repo at all. That bug was invisible until the lane was actually exercised, because
        # nothing constructs a QualityEmbedder until someone flips the toggle.
        onnx_path = None
        for candidate in ("onnx/model_qint8_avx512_vnni.onnx", "onnx/model.onnx"):
            try:
                onnx_path = hf_hub_download(model_id, candidate)
                log.info("quality lane using %s", candidate)
                break
            except Exception as e:
                log.warning("quality lane: %s unavailable (%s)", candidate, type(e).__name__)
        if onnx_path is None:
            raise StageError(
                "embed", ErrorKind.INTERNAL, f"no usable ONNX export found for {model_id}"
            )
        tok_path = hf_hub_download(model_id, "tokenizer.json")

        opts = ort.SessionOptions()
        # The Space runs on a small shared CPU. Letting ORT spawn a thread per core on a
        # contended box adds scheduling latency rather than removing it; one thread is both
        # faster and far more predictable at the tail, which is what we are actually optimising.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(tok_path)
        self._tokenizer.enable_truncation(max_length=512)
        self._input_names = {i.name for i in self._session.get_inputs()}

        probe = self.encode_query("warm")
        self.dim = int(probe.shape[-1])
        log.info("quality embedder ready, dim=%d", self.dim)

    def _run(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        max_len = max(len(e.ids) for e in encodings)
        n = len(encodings)

        ids = np.zeros((n, max_len), dtype=np.int64)
        mask = np.zeros((n, max_len), dtype=np.int64)
        for i, enc in enumerate(encodings):
            ln = len(enc.ids)
            ids[i, :ln] = enc.ids
            mask[i, :ln] = enc.attention_mask

        feed: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros((n, max_len), dtype=np.int64)

        hidden = self._session.run(None, feed)[0]
        # e5 uses mean pooling over non-padding tokens, not the CLS vector.
        m = mask[:, :, None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        return _l2_normalize_rows(pooled.astype(np.float32))

    def encode_query(self, text: str) -> np.ndarray:
        return self._run([f"query: {text}"])[0]

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = [self._run([f"passage: {t}" for t in texts[i : i + batch_size]])
               for i in range(0, len(texts), batch_size)]
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec if norm == 0.0 else (vec / norm).astype(np.float32)


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (mat / norms).astype(np.float32)


# ---------------------------------------------------------------------------------------
# Lane registry
# ---------------------------------------------------------------------------------------

_LOCK = threading.Lock()
_LANES: dict[str, Embedder] = {}


def get_embedder(lane: str) -> Embedder:
    """Return the requested lane, constructing it at most once.

    Double-checked locking rather than a plain `lru_cache`: two concurrent first-requests for the
    quality lane would otherwise each load an ONNX session, briefly doubling memory on a box that
    does not have the headroom for it.
    """
    hit = _LANES.get(lane)
    if hit is not None:
        return hit
    with _LOCK:
        hit = _LANES.get(lane)
        if hit is not None:
            return hit
        if lane == "fast":
            emb: Embedder = FastEmbedder()
        elif lane == "quality":
            emb = QualityEmbedder()
        else:
            raise StageError("embed", ErrorKind.INVALID_INPUT, f"unknown embedding lane {lane!r}")
        _LANES[lane] = emb
        return emb


def loaded_lanes() -> list[str]:
    return sorted(_LANES)
