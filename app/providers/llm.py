"""Generation providers — Cerebras and Groq, behind one interface with failover.

Both expose an OpenAI-compatible chat-completions API, so one streaming client covers both and the
difference collapses to a base URL, a model id, and a key. Writing two nearly-identical clients
would have been two places to fix every bug.

Measured on this account (see docs/BUILD_LOG.md):

| provider | model | TTFT | notes |
|---|---|---|---|
| Groq | llama-3.3-70b-versatile | 477ms | primary |
| Groq | openai/gpt-oss-20b | 562ms | secondary |
| Cerebras | — | — | 402: free quota unavailable on this account |

Every one of those is multiples of the 200ms answer budget. That is not a defect to optimise away —
it is the reason Tier 1 exists, and the reason generation is strictly an enhancement layered on top
of an answer that has already shipped.

Two failure modes are handled explicitly because both were observed rather than imagined:

- **Cloudflare 1010.** Groq sits behind Cloudflare, which rejects some default user-agents with a
  403 that looks exactly like an auth failure. `httpx` gets through where `urllib` does not. An
  explicit User-Agent is set so this cannot silently regress.
- **402 payment required.** Cerebras returns this when free quota is unavailable. It is permanent
  for the request, not transient, so the circuit breaker opens immediately rather than retrying.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from app.stages.base import ErrorKind, StageError

log = logging.getLogger("shruti.llm")

USER_AGENT = "shruti/0.1 (+https://github.com/Het161/shruti)"


@dataclass(slots=True)
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str | None
    # Providers that have failed permanently (402/401) are skipped entirely rather than retried on
    # every request — an unavoidable failure that costs a round-trip each time is worse than one
    # that is skipped.
    disabled_reason: str | None = None
    consecutive_failures: int = 0


@dataclass(slots=True)
class GenerationChunk:
    text: str
    is_first: bool


@dataclass(slots=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    ttft_ms: float | None
    total_ms: float
    finish_reason: str | None = None


class CircuitBreaker:
    """Opens after repeated failures so a dead provider stops costing latency on every request."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 60.0) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._opened_at: dict[str, float] = {}

    def is_open(self, name: str) -> bool:
        opened = self._opened_at.get(name)
        if opened is None:
            return False
        if time.monotonic() - opened > self.cooldown_s:
            del self._opened_at[name]
            return False
        return True

    def record_failure(self, cfg: ProviderConfig) -> None:
        cfg.consecutive_failures += 1
        if cfg.consecutive_failures >= self.threshold:
            self._opened_at[cfg.name] = time.monotonic()
            log.warning("circuit opened for %s after %d failures", cfg.name, cfg.consecutive_failures)

    def record_success(self, cfg: ProviderConfig) -> None:
        cfg.consecutive_failures = 0
        self._opened_at.pop(cfg.name, None)


class LLMProvider:
    """Streaming client for one OpenAI-compatible endpoint."""

    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self._client = client

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 400,
        temperature: float = 0.2,
        ttft_timeout_ms: int = 900,
        total_timeout_ms: int = 8000,
    ) -> AsyncIterator[GenerationChunk]:
        if not self.cfg.api_key:
            raise StageError("generate", ErrorKind.PROVIDER_ERROR, f"{self.cfg.name}: no API key")

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        t0 = time.perf_counter()
        first = True
        try:
            async with self._client.stream(
                "POST",
                f"{self.cfg.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(total_timeout_ms / 1000, connect=5.0),
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode()[:300]
                    kind = (
                        ErrorKind.PROVIDER_RATE_LIMIT
                        if response.status_code == 429
                        else ErrorKind.PROVIDER_ERROR
                    )
                    if response.status_code in (401, 402, 403):
                        self.cfg.disabled_reason = f"HTTP {response.status_code}: {body[:120]}"
                    raise StageError(
                        "generate",
                        kind,
                        f"{self.cfg.name} HTTP {response.status_code}: {body}",
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0].get("delta", {})
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    token = delta.get("content") or ""
                    if not token:
                        continue
                    if first:
                        ttft = (time.perf_counter() - t0) * 1000
                        if ttft > ttft_timeout_ms:
                            log.info(
                                "%s TTFT %.0fms exceeded %dms budget; Tier 1 already served",
                                self.cfg.name,
                                ttft,
                                ttft_timeout_ms,
                            )
                    yield GenerationChunk(text=token, is_first=first)
                    first = False

        except httpx.TimeoutException as e:
            raise StageError(
                "generate", ErrorKind.PROVIDER_TIMEOUT, f"{self.cfg.name} timed out", cause=e
            ) from e
        except httpx.HTTPError as e:
            raise StageError(
                "generate", ErrorKind.PROVIDER_ERROR, f"{self.cfg.name}: {e}", cause=e
            ) from e


class ProviderChain:
    """Tries providers in order, skipping disabled and circuit-open ones.

    Never raises on exhaustion: it returns nothing and the caller keeps Tier 1. Generation failing
    is a normal, expected state in this system, not an error condition — the user already has a
    complete cited answer by the time this runs.
    """

    def __init__(self, configs: list[ProviderConfig]) -> None:
        self.configs = configs
        self.breaker = CircuitBreaker()
        self._client = httpx.AsyncClient(headers={"User-Agent": USER_AGENT})

    def available(self) -> list[ProviderConfig]:
        return [
            c
            for c in self.configs
            if c.api_key and not c.disabled_reason and not self.breaker.is_open(c.name)
        ]

    def status(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for c in self.configs:
            if not c.api_key:
                out[c.name] = "no key"
            elif c.disabled_reason:
                out[c.name] = f"disabled: {c.disabled_reason}"
            elif self.breaker.is_open(c.name):
                out[c.name] = "circuit open"
            else:
                out[c.name] = "ready"
        return out

    async def stream(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> AsyncIterator[tuple[ProviderConfig, GenerationChunk]]:
        for cfg in self.available():
            provider = LLMProvider(cfg, self._client)
            try:
                got_any = False
                async for chunk in provider.stream(messages, **kwargs):  # type: ignore[arg-type]
                    got_any = True
                    yield cfg, chunk
                if got_any:
                    self.breaker.record_success(cfg)
                    return
            except StageError as e:
                log.warning("provider %s failed: %s", cfg.name, e.message)
                self.breaker.record_failure(cfg)
                continue
        log.info("no generation provider available; Tier 1 stands alone")

    async def aclose(self) -> None:
        await self._client.aclose()


def build_chain(cerebras_key: str | None, groq_key: str | None) -> ProviderChain:
    """Construct the provider chain in preference order.

    Cerebras is kept first because it is the fastest option when its quota is available, and the
    chain costs nothing to try — a disabled provider is skipped without a round-trip after its
    first 402.
    """
    return ProviderChain(
        [
            ProviderConfig(
                name="cerebras",
                base_url="https://api.cerebras.ai/v1",
                model="gpt-oss-120b",
                api_key=cerebras_key,
            ),
            ProviderConfig(
                name="groq",
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
            ),
            ProviderConfig(
                name="groq-fallback",
                base_url="https://api.groq.com/openai/v1",
                model="openai/gpt-oss-20b",
                api_key=groq_key,
            ),
        ]
    )
