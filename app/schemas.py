"""Typed contracts for every pipeline stage.

The task asks for structured orchestration rather than prompt-in/text-out, so each stage declares
what it consumes and what it produces, and the orchestrator composes them against those
declarations.

A note on where pydantic is and is not used. Validation costs microseconds, which is affordable at
stage boundaries but not worth paying inside a tight scoring loop over tens of thousands of
vectors. So: pydantic models at every boundary that crosses a module or leaves the process, plain
NumPy and primitives inside the hot loops. `EmbedOutput` carries a raw `ndarray` rather than a list
of floats for the same reason — serialising 256 floats into Python objects and back would cost more
than the embedding itself.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------------------
# Corpus primitives
# ---------------------------------------------------------------------------------------


class Passage(BaseModel):
    """One retrievable unit, as produced by the corpus builder."""

    passage_id: str
    text: str
    lang: str
    query_type: str | None = None
    source_query_id: int
    position: int = Field(description="Index of this passage within its source query's list")


class ScoredPassage(BaseModel):
    """A passage with the full provenance of how it was retrieved.

    Component scores and ranks are kept, not just the fused score, because the UI shows why a
    passage surfaced and the chunking lab needs per-retriever attribution to explain wins.
    """

    passage: Passage
    fused_score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None


# ---------------------------------------------------------------------------------------
# Stage contracts
# ---------------------------------------------------------------------------------------


class QueryInput(BaseModel):
    """Entry point for both the text path and the voice path.

    The two entrypoints construct this identically, which is what makes 'one pipeline, two
    entrypoints' true rather than aspirational.
    """

    text: str = Field(min_length=1, max_length=2000)
    lang: str | None = Field(default=None, description="ISO code; None means auto-detect")
    request_id: str


class EmbedOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector: np.ndarray
    lane: Literal["fast", "quality"]
    dim: int


class RetrievalOutput(BaseModel):
    """Output of a single retriever, before fusion."""

    candidates: list[ScoredPassage]
    retriever: Literal["dense", "lexical"]


class FusionOutput(BaseModel):
    candidates: list[ScoredPassage]
    top_score: float = Field(description="Top fused score — the signal the scope gate thresholds")


# ---------------------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------------------


class Gate(str, Enum):
    """Which guardrail fired. Shown by name in the UI's refusal lamp."""

    SCOPE = "scope"
    SAFETY = "safety"
    GROUNDING = "grounding"
    INJECTION = "injection"


class GuardVerdict(BaseModel):
    """A guardrail decision.

    Always present in the response, including when nothing fired — the absence of a refusal is
    itself a measured verdict, and rendering it keeps the lamp honest rather than making it a
    branch that only appears on failure.
    """

    allowed: bool
    gate: Gate | None = None
    reason: str | None = None
    score: float | None = Field(default=None, description="The value compared against threshold")
    threshold: float | None = None


# ---------------------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------------------


class Citation(BaseModel):
    """Maps a bracketed marker in the answer to the passage it came from."""

    marker: int = Field(ge=1, description="The n in [n]")
    passage_id: str
    char_start: int | None = None
    char_end: int | None = None


class ExtractiveAnswer(BaseModel):
    """Tier 1 — grounded by construction.

    The text is copied verbatim from retrieved passages, so it cannot hallucinate. This is what
    makes the sub-200ms claim an honest one: it is a complete, citable answer, not a placeholder
    that the real answer replaces.
    """

    text: str
    lang: str
    citations: list[Citation]
    source_passage_ids: list[str]


class GenerativeAnswer(BaseModel):
    """Tier 2 — streamed, and permitted to be withheld."""

    text: str
    lang: str
    citations: list[Citation]
    provider: str
    model: str
    ttft_ms: float | None = None
    total_ms: float | None = None
    grounded: bool = Field(description="False means the post-hoc grounding check rejected it")
    withheld_reason: str | None = None


# ---------------------------------------------------------------------------------------
# API envelope
# ---------------------------------------------------------------------------------------


class StageTiming(BaseModel):
    name: str
    offset_ms: float
    duration_ms: float


class TimingBreakdown(BaseModel):
    """Rendered directly as the waterfall. Present on every response, including refusals."""

    request_id: str
    stages: list[StageTiming]
    measured_ms: float
    total_ms: float
    unattributed_ms: float


class AskRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=2000)]
    lang: str | None = None
    lane: Literal["fast", "quality"] | None = None
    search_mode: Literal["hnsw", "exact"] | None = None
    top_n: int | None = Field(default=None, ge=1, le=20)
    generate: bool = Field(default=False, description="Request Tier 2; Tier 1 returns regardless")


class AskResponse(BaseModel):
    request_id: str
    query: str
    detected_lang: str
    guard: GuardVerdict
    answer: ExtractiveAnswer | None
    generative: GenerativeAnswer | None = None
    passages: list[ScoredPassage]
    timings: TimingBreakdown
    lane: Literal["fast", "quality"]
    search_mode: Literal["hnsw", "exact"]


class HealthResponse(BaseModel):
    """What `/api/health` reports.

    `ready` stays false until warmup completes, so the Space is never announced as healthy while
    the first real query would still pay model-load cost.
    """

    status: Literal["starting", "ready", "degraded"]
    ready: bool
    corpus_passages: int
    corpus_languages: list[str]
    embed_lane: str
    embed_dim: int
    search_mode: str
    warmup_queries_run: int
    warmup_p50_ms: float | None = None
    providers: dict[str, bool]
    uptime_s: float
    version: str
