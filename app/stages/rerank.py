"""Cross-encoder reranking — Tier 2 only, never on the Tier 1 critical path.

Why this cannot live where retrieval lives
------------------------------------------
Measured on this deployment's CPU class, `jina-reranker-v2-base-multilingual` int8 ONNX:

    top-10   p50  121.6ms
    top-20   p50  245.4ms
    top-50   p50  621.3ms

The whole answer SLO is 200ms. Reranking 50 pairs is **3.1x the entire budget**, and even top-10
consumes 61% of it while Tier 1 currently answers in ~20ms. So the reranker is architecturally
excluded from the path that produces the guaranteed answer. It runs in the Tier 2 lane, after Tier
1 has already been served, alongside generation — where its cost lands on "full answer time",
which is reported separately and honestly, rather than on "answer visible".

This is the constraint that decided the design, not a preference. A cross-encoder is the single
largest MRR gain available in retrieval, and it is still wrong to put it in front of the SLO the
product is judged on.

Why a cross-encoder helps at all
--------------------------------
Bi-encoders embed query and passage independently, so a passage's vector cannot depend on the
question being asked. A cross-encoder reads both together and can attend across them, which is why
it resolves exactly the failure this corpus exhibits: distractors that are topically near the query
but do not answer it. Measured degradation with corpus size (see docs/BUILD_LOG.md) showed
precision falling 60% from 6k to 310k passages as distractors crowd the top-10 — that is precisely
the error a reranker is built to correct.

Multilingual matters here and narrowed the field. `bge-reranker-base` is predominantly
English/Chinese; this corpus is Gujarati, Hindi, Bengali, Tamil and English, so a reranker that
cannot read the query is worse than none. jina-reranker-v2 covers 100+ languages and its int8 ONNX
export loads without `trust_remote_code`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from app.stages.base import ErrorKind, StageError

log = logging.getLogger("shruti.rerank")

MODEL_ID = "jinaai/jina-reranker-v2-base-multilingual"
# Sequence length is the dominant latency term for a cross-encoder, and corpus passages average
# ~59 words. 256 tokens covers essentially all of them while costing half of what 512 would.
MAX_LENGTH = 256


@dataclass(slots=True)
class RerankResult:
    order: list[int]
    scores: list[float]
    elapsed_ms: float
    depth: int


class CrossEncoderReranker:
    """int8 ONNX cross-encoder. Loaded lazily — never on the startup path."""

    def __init__(self, model_id: str = MODEL_ID, max_length: int = MAX_LENGTH) -> None:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover
            raise StageError(
                "rerank", ErrorKind.INTERNAL, "reranker needs onnxruntime + tokenizers", cause=e
            ) from e

        log.info("loading reranker %s", model_id)
        t0 = time.perf_counter()
        # int8 first: PyTorch-equivalent fp32 is 3-4x slower on CPU, which for a stage measured in
        # hundreds of milliseconds is the difference between usable in Tier 2 and unusable anywhere.
        onnx_path = None
        for candidate in ("onnx/model_int8.onnx", "onnx/model_quantized.onnx", "onnx/model.onnx"):
            try:
                onnx_path = hf_hub_download(model_id, candidate)
                log.info("reranker using %s", candidate)
                break
            except Exception as e:
                log.warning("reranker: %s unavailable (%s)", candidate, type(e).__name__)
        if onnx_path is None:
            raise StageError("rerank", ErrorKind.INTERNAL, f"no ONNX export for {model_id}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}

        self._tok = Tokenizer.from_file(hf_hub_download(model_id, "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding()
        self.max_length = max_length
        log.info("reranker ready in %.1fs", time.perf_counter() - t0)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.empty(0, dtype=np.float32)
        enc = self._tok.encode_batch([(query, p) for p in passages])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        logits = self._session.run(None, feed)[0]
        return np.asarray(logits, dtype=np.float32).reshape(-1)

    def rerank(self, query: str, passages: list[str], depth: int) -> RerankResult:
        """Rerank the first `depth` passages. Returns indices into the original list."""
        head = passages[:depth]
        t0 = time.perf_counter()
        scores = self.score(query, head)
        elapsed = (time.perf_counter() - t0) * 1000
        order = np.argsort(-scores).tolist()
        # Anything beyond `depth` keeps its retrieval order and trails the reranked head.
        order += list(range(len(head), len(passages)))
        return RerankResult(
            order=order,
            scores=scores.tolist(),
            elapsed_ms=elapsed,
            depth=len(head),
        )


_LOCK = threading.Lock()
_INSTANCE: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    """Construct at most once, on first use.

    Deliberately not loaded at startup: the reranker is off by default, and paying its load cost
    on every cold start for a stage most requests never touch would slow the number that is
    actually measured.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    with _LOCK:
        if _INSTANCE is None:
            _INSTANCE = CrossEncoderReranker()
        return _INSTANCE


def is_loaded() -> bool:
    return _INSTANCE is not None
