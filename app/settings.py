"""Runtime configuration.

Fails fast and loudly at startup rather than at first request: a Space that boots healthy and then
500s on the judges' first query is worse than one that refuses to boot.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbedLane = Literal["fast", "quality"]
SearchMode = Literal["hnsw", "exact"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- providers -------------------------------------------------------------------
    # All optional. Absent keys degrade the system rather than breaking it: without a Sarvam key
    # the text path still works; without Cerebras/Groq, Tier 1 extractive answers still ship.
    sarvam_api_key: str | None = Field(default=None, alias="SARVAM_API_KEY")
    cerebras_api_key: str | None = Field(default=None, alias="CEREBRAS_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    artifact_repo: str | None = Field(default=None, alias="SHRUTI_ARTIFACT_REPO")

    # --- retrieval -------------------------------------------------------------------
    embed_lane: EmbedLane = Field(default="fast", alias="SHRUTI_EMBED_LANE")
    search_mode: SearchMode = Field(default="hnsw", alias="SHRUTI_SEARCH_MODE")
    artifact_dir: Path = Field(default=Path("artifacts"), alias="SHRUTI_ARTIFACT_DIR")

    # Candidates pulled from each retriever before fusion.
    dense_top_k: int = Field(default=50, alias="SHRUTI_DENSE_TOP_K")
    lexical_top_k: int = Field(default=50, alias="SHRUTI_LEXICAL_TOP_K")
    # Passages handed to the answerer after fusion.
    context_top_n: int = Field(default=3, alias="SHRUTI_CONTEXT_TOP_N")
    # Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper; kept
    # configurable so the chunking lab can sweep it rather than inherit it on faith.
    rrf_k: int = Field(default=60, alias="SHRUTI_RRF_K")

    # --- guardrails ------------------------------------------------------------------
    # Weak-retrieval threshold on the top dense cosine.
    #
    # Chosen cost-first rather than recall-first, and the distinction is the whole reason a usable
    # number exists. Asking "what threshold rejects 95% of out-of-domain queries" produced 0.7676,
    # which also refuses 78.8% of real ones — useless. Asking instead "what am I willing to lose"
    # and fixing that at 5% gives the 5th percentile of the in-domain score distribution, 0.5701:
    #
    #   in-domain cost    tau      out-of-domain rejected
    #   1%                0.5048   7.1%
    #   2%                0.5418   12.9%
    #   5%                0.5701   24.3%   <- shipped
    #   10%               0.5987   41.4%
    #
    # This is not an out-of-domain detector and is not claimed to be one — `check_conversational`
    # does that job at 80% for 0.025% cost. This is the second layer, for utterances that ARE
    # genuine questions but concern something the corpus has no strong evidence about. It reads as
    # "refuse when retrieval lands in the weakest 5% of what normal queries produce".
    #
    # Motivating case: "તાપમાન કેટલું છે નિકોલ અમદાવાદમાં?" retrieved at 0.424 — below the 1st
    # percentile — and was answered with Palm Harbor city populations and Nicole Brown Simpson.
    scope_tau: float | None = Field(default=0.5701, alias="SHRUTI_SCOPE_TAU")

    # --- timeouts (milliseconds) -----------------------------------------------------
    # Every external call is bounded. On breach the pipeline degrades to a lower tier and logs the
    # event; it never blocks the latency-critical path waiting on someone else's infrastructure.
    stt_finalize_timeout_ms: int = Field(default=2000, alias="SHRUTI_STT_TIMEOUT_MS")
    gen_ttft_timeout_ms: int = Field(default=900, alias="SHRUTI_GEN_TTFT_TIMEOUT_MS")
    gen_total_timeout_ms: int = Field(default=8000, alias="SHRUTI_GEN_TOTAL_TIMEOUT_MS")

    # --- startup ---------------------------------------------------------------------
    warmup_queries: int = Field(default=20, alias="SHRUTI_WARMUP_QUERIES")

    @property
    def has_generation(self) -> bool:
        return bool(self.cerebras_api_key or self.groq_api_key)

    @property
    def has_voice(self) -> bool:
        return bool(self.sarvam_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
