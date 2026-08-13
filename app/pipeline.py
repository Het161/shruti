"""The pipeline orchestrator.

One object, two entrypoints. `/api/ask` and the voice WebSocket both construct a `QueryInput` and
call `Pipeline.ask`, which is what makes "one implementation, two entrypoints" a structural fact
rather than a claim — the benchmark harness exercises literally the same code path the microphone
does.

Stage order, and why it is this order:

    detect → safety → embed → (dense ‖ lexical) → fuse → scope → extract

Safety runs before retrieval because refusing early saves the work entirely. Scope runs *after*
fusion because its evidence is the retrieval score itself — there is no way to know a question is
out of corpus without looking. Extraction runs last and is pure computation over already-retrieved
text.

Every stage is timed through `RequestTimer`, and the resulting waterfall is returned to the client
in the same response as the answer. A stage that is skipped does not appear, so the waterfall shows
what actually ran rather than a fixed template with zeros in it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.corpus import Corpus
from app.schemas import (
    AskRequest,
    AskResponse,
    ExtractiveAnswer,
    GuardVerdict,
    Passage,
    ScoredPassage,
    StageTiming,
    TimingBreakdown,
)
from app.settings import Settings
from app.stages import guard as guards
from app.stages.answer import extract_answer
from app.stages.base import ErrorKind, StageError
from app.stages.dense import DenseIndex
from app.stages.embed import Embedder, get_embedder
from app.stages.fuse import FusedHit, reciprocal_rank_fusion
from app.stages.lang import detect, retrieval_langs
from app.stages.lexical import LexicalIndex
from app.timing import RequestTimer

log = logging.getLogger("shruti.pipeline")

VERSION = "0.1.0"


@dataclass(slots=True)
class RetrievalResult:
    """Intermediate state handed between the retrieval half and the answer half."""

    hits: list[FusedHit]
    top_dense_score: float | None
    detected_lang: str
    searched_langs: list[str]
    # Number of BM25 hits. Carried so the degenerate-input gate can distinguish "no lexical
    # evidence anywhere in the corpus" from "weak but present evidence" without re-running search.
    lexical_hits: int
    # Carried forward so the extractive stage reuses it. Re-embedding the same query a second time
    # would be pure waste on the one path where every millisecond is the product.
    query_vec: np.ndarray


class Pipeline:
    """Composes stages with timing, bounded failure, and structured verdicts."""

    def __init__(
        self,
        corpus: Corpus,
        dense_index: DenseIndex | None,
        lexical_index: LexicalIndex | None,
        settings: Settings,
    ) -> None:
        self.corpus = corpus
        self.dense_index = dense_index
        self.lexical_index = lexical_index
        self.settings = settings
        self.embedders: dict[str, Embedder] = {}

    # -- lanes --------------------------------------------------------------------------

    def embedder_for(self, lane: str) -> Embedder:
        emb = self.embedders.get(lane)
        if emb is None:
            emb = get_embedder(lane)
            self.embedders[lane] = emb
        return emb

    # -- retrieval ----------------------------------------------------------------------

    def retrieve(
        self,
        text: str,
        timer: RequestTimer,
        *,
        lane: str,
        search_mode: str,
        lang_hint: str | None = None,
    ) -> RetrievalResult:
        with timer.stage("detect"):
            guess = detect(text) if lang_hint is None else None
            detected = lang_hint or (guess.lang if guess else "en")
            langs = retrieval_langs(detected)

        with timer.stage("embed"):
            embedder = self.embedder_for(lane)
            # A lane whose dimensionality does not match the indexed corpus cannot search it, and
            # the failure would otherwise surface as an opaque shape error deep inside a matmul.
            #
            # This is a real limitation of the two-lane design, not a bug to paper over: the fast
            # lane is 256-dim (potion) and the quality lane is 384-dim (e5-small), so switching
            # lanes at query time requires a *second* corpus embedding at the other width — about
            # 477 MB and ~26 minutes of e5 forward passes for this corpus. The toggle is honest
            # about that rather than pretending.
            if embedder.dim != self.corpus.dim:
                raise StageError(
                    "embed",
                    ErrorKind.INVALID_INPUT,
                    f"lane '{lane}' produces {embedder.dim}-dim vectors but the corpus is indexed "
                    f"at {self.corpus.dim} dims. Serving this lane requires a corpus embedded with "
                    f"the same model; see docs/BUILD_LOG.md.",
                )
            query_vec = embedder.encode_query(text)

        with timer.stage("dense"):
            dense_rows, dense_scores = self._dense_search(query_vec, langs, search_mode)

        with timer.stage("bm25"):
            if self.lexical_index is not None:
                lex_rows, lex_scores = self.lexical_index.search(
                    text, self.settings.lexical_top_k
                )
                lex_rows, lex_scores = self._filter_langs(lex_rows, lex_scores, langs)
            else:
                lex_rows = np.empty(0, dtype=np.int64)
                lex_scores = np.empty(0, dtype=np.float32)

        with timer.stage("fuse"):
            hits = reciprocal_rank_fusion(
                dense_rows,
                dense_scores,
                lex_rows,
                lex_scores,
                k=self.settings.rrf_k,
                top_n=max(self.settings.context_top_n, 10),
            )

        top_dense = float(dense_scores[0]) if dense_scores.size else None
        return RetrievalResult(
            hits=hits,
            top_dense_score=top_dense,
            detected_lang=detected,
            searched_langs=langs,
            lexical_hits=int(lex_rows.size),
            query_vec=query_vec,
        )

    def _dense_search(
        self, query_vec: np.ndarray, langs: list[str], search_mode: str
    ) -> tuple[np.ndarray, np.ndarray]:
        k = self.settings.dense_top_k
        if search_mode == "exact":
            mask = self.corpus.mask_for_langs(langs)
            return self.corpus.exact_search(query_vec, k, mask=mask)

        if self.dense_index is None:
            raise StageError(
                "dense", ErrorKind.INDEX_ERROR, "HNSW index not loaded; use search_mode=exact"
            )
        allowed = self.corpus.rows_for_langs(langs)
        return self.dense_index.search(query_vec, k, allowed_rows=allowed)

    def _filter_langs(
        self, rows: np.ndarray, scores: np.ndarray, langs: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Restrict lexical hits to the searched language partitions.

        BM25 has no notion of language, so a Hindi query can match an English passage on a shared
        token — a transliterated name, a number, an ASCII acronym. Those matches are not wrong, but
        they must obey the same partition rule the dense side does or fusion compares lists drawn
        from different corpora.
        """
        if rows.size == 0:
            return rows, scores
        mask = self.corpus.mask_for_langs(langs)
        if mask is None:
            return rows, scores
        keep = mask[rows]
        return rows[keep], scores[keep]

    # -- assembly -----------------------------------------------------------------------

    def _to_scored(self, hits: list[FusedHit]) -> list[ScoredPassage]:
        out: list[ScoredPassage] = []
        for hit in hits:
            ref = self.corpus.get(hit.row)
            out.append(
                ScoredPassage(
                    passage=Passage(
                        passage_id=ref.passage_id,
                        text=ref.text,
                        lang=ref.lang,
                        query_type=ref.query_type,
                        source_query_id=ref.source_query_id,
                        position=ref.position,
                    ),
                    fused_score=hit.fused_score,
                    dense_score=hit.dense_score,
                    lexical_score=hit.lexical_score,
                    dense_rank=hit.dense_rank,
                    lexical_rank=hit.lexical_rank,
                )
            )
        return out

    @staticmethod
    def _breakdown(timer: RequestTimer) -> TimingBreakdown:
        snap = timer.snapshot()
        return TimingBreakdown(
            request_id=str(snap["request_id"]),
            stages=[StageTiming(**s) for s in snap["stages"]],  # type: ignore[arg-type]
            measured_ms=float(snap["measured_ms"]),  # type: ignore[arg-type]
            total_ms=float(snap["total_ms"]),  # type: ignore[arg-type]
            unattributed_ms=float(snap["unattributed_ms"]),  # type: ignore[arg-type]
        )

    # -- the entrypoint -----------------------------------------------------------------

    def ask(self, req: AskRequest, timer: RequestTimer | None = None) -> AskResponse:
        timer = timer or RequestTimer()
        lane = req.lane or self.settings.embed_lane
        search_mode = req.search_mode or self.settings.search_mode
        top_n = req.top_n or self.settings.context_top_n

        with timer.stage("guard_safety"):
            safety = guards.check_safety(req.text)
            # Conversational-intent screen runs alongside safety, before any retrieval. Measured at
            # 80% out-of-domain rejection for 0.05% false refusals (1 in 2000 real queries) — the
            # only gate that separates, after three retrieval-derived signals failed to. See
            # lab/calibrate_scope.py.
            if safety.allowed:
                safety = guards.check_conversational(req.text)

        if not safety.allowed:
            return AskResponse(
                request_id=timer.request_id,
                query=req.text,
                detected_lang=detect(req.text).lang,
                guard=safety,
                answer=None,
                passages=[],
                timings=self._breakdown(timer),
                lane=lane,  # type: ignore[arg-type]
                search_mode=search_mode,  # type: ignore[arg-type]
            )

        result = self.retrieve(
            req.text, timer, lane=lane, search_mode=search_mode, lang_hint=req.lang
        )

        with timer.stage("guard_scope"):
            # Degenerate-input check first: it uses lexical evidence, which separates gibberish
            # from real questions where the cosine threshold demonstrably cannot.
            scope = guards.check_degenerate(req.text, result.lexical_hits)
            if scope.allowed:
                scope = guards.check_scope(result.top_dense_score, self.settings.scope_tau)

        passages = self._to_scored(result.hits[:top_n])

        if not scope.allowed:
            return AskResponse(
                request_id=timer.request_id,
                query=req.text,
                detected_lang=result.detected_lang,
                guard=scope,
                answer=None,
                # Passages are still returned on a scope refusal. Showing the judges what the
                # system *did* find, and that it judged the evidence too weak, is the whole point
                # of an abstention gate — an empty refusal proves nothing.
                passages=passages,
                timings=self._breakdown(timer),
                lane=lane,  # type: ignore[arg-type]
                search_mode=search_mode,  # type: ignore[arg-type]
            )

        with timer.stage("extract"):
            answer: ExtractiveAnswer | None = extract_answer(
                req.text,
                result.query_vec,
                passages,
                self.embedder_for(lane),
                query_type=passages[0].passage.query_type if passages else None,
                prefer_lang=result.detected_lang,
            )

        verdict = scope if answer is not None else GuardVerdict(
            allowed=False,
            gate=None,
            reason="Retrieved passages contained no extractable answer sentence.",
            score=result.top_dense_score,
        )

        return AskResponse(
            request_id=timer.request_id,
            query=req.text,
            detected_lang=result.detected_lang,
            guard=verdict,
            answer=answer,
            passages=passages,
            timings=self._breakdown(timer),
            lane=lane,  # type: ignore[arg-type]
            search_mode=search_mode,  # type: ignore[arg-type]
        )
