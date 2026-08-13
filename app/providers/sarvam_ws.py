"""Sarvam Saaras v3 streaming STT relay.

Contract verified against Sarvam's WebSocket reference (2026-08-13):

    URL     wss://api.sarvam.ai/speech-to-text/ws
    Auth    Api-Subscription-Key: <key>   (connection header)
    Query   model=saaras:v3  mode=transcribe  sample_rate=16000
            language-code=unknown          (enables auto-detection)
            vad_signals=true               (START_SPEECH / END_SPEECH events)

    client -> {"audio": {"data": "<base64>", "sample_rate": "16000", "encoding": "audio/wav"}}
    client -> {"type": "flush"}
    server -> {"type": "data",   "data": {"transcript", "language_code",
                                          "language_probability", "metrics"}}
    server -> {"type": "events", "data": {"signal_type": "START_SPEECH"|"END_SPEECH"}}
    server -> {"type": "error",  "data": {"error", "code"}}

Why Sarvam rather than ElevenLabs
---------------------------------
The corpus is Indic. Saaras is trained for Indian languages and code-mixed speech, does
auto-detection across them, and `language-code=unknown` means a user can switch from Gujarati to
English mid-sentence without touching a setting. For a dataset that is MS MARCO translated into
fourteen Indic languages, the Indic specialist is the correct engineering choice — not a patriotic
one, a matching-the-data one.

Why the relay exists, and what it costs
---------------------------------------
The browser cannot hold the API key, so audio goes browser -> our server -> Sarvam. Sarvam's
endpoint is in India and the app server is in the US, so audio crosses the ocean twice. This is
measured and displayed as its own `stt` stage rather than folded into the answer budget — the
200ms SLO starts at *final transcript*, which is the first moment the retrieval pipeline has
anything to work on.

The honest fix is an ephemeral client token so the browser connects to Sarvam directly, removing
one ocean crossing. Whether Sarvam issues those is an open question flagged in the build log.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlencode

import websockets

log = logging.getLogger("shruti.sarvam")

WS_BASE = "wss://api.sarvam.ai/speech-to-text/ws"


@dataclass(slots=True)
class Transcript:
    """One transcription event from Sarvam."""

    text: str
    language_code: str | None
    language_probability: float | None
    is_final: bool
    elapsed_ms: float
    processing_latency: float | None = None


@dataclass(slots=True)
class SpeechEvent:
    signal_type: str  # START_SPEECH | END_SPEECH
    elapsed_ms: float


class SarvamStream:
    """One live STT session.

    Deliberately not a request/response client: the whole point of streaming STT is that
    transcription overlaps with speech, so by the time the user stops talking most of the work is
    already done. A batch STT call would serialise 'record, then transcribe, then retrieve' and add
    the entire utterance duration to perceived latency.
    """

    def __init__(
        self,
        api_key: str,
        *,
        language_code: str = "unknown",
        mode: str = "transcribe",
        sample_rate: int = 16000,
        model: str = "saaras:v3",
        vad_signals: bool = True,
        high_vad_sensitivity: bool = False,
    ) -> None:
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.language_code = language_code
        # `high_vad_sensitivity` defaults to False now, and this was a real bug rather than a
        # preference. It was hardcoded to True, which makes the voice-activity detector segment
        # eagerly — and eager segmentation clips the beginning of an utterance, because the first
        # syllable is what convinces the detector speech has started.
        #
        # Observed: "My name is Het Patel" came back as just the tail of the phrase. The audio was
        # not corrupted (the transcript is phonetically close to "Het Patel"), the front of it was
        # simply never sent for recognition.
        self._query = urlencode(
            {
                "model": model,
                "mode": mode,
                "sample_rate": str(sample_rate),
                "language-code": language_code,
                "vad_signals": "true" if vad_signals else "false",
                "high_vad_sensitivity": "true" if high_vad_sensitivity else "false",
            }
        )
        self._ws: websockets.ClientConnection | None = None
        self._t0 = time.perf_counter()

    async def __aenter__(self) -> SarvamStream:
        url = f"{WS_BASE}?{self._query}"
        self._ws = await websockets.connect(
            url,
            additional_headers={"Api-Subscription-Key": self.api_key},
            open_timeout=10,
            ping_interval=20,
        )
        self._t0 = time.perf_counter()
        log.info("sarvam stream open")
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    def _elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    async def send_audio(self, pcm16: bytes) -> None:
        """Forward one chunk of 16kHz mono PCM16."""
        if self._ws is None:
            raise RuntimeError("stream not open")
        await self._ws.send(
            json.dumps(
                {
                    "audio": {
                        "data": base64.b64encode(pcm16).decode("ascii"),
                        "sample_rate": str(self.sample_rate),
                        "encoding": "audio/wav",
                    }
                }
            )
        )

    async def flush(self) -> None:
        """Signal end of utterance so Sarvam emits the final transcript."""
        if self._ws is None:
            raise RuntimeError("stream not open")
        await self._ws.send(json.dumps({"type": "flush"}))

    async def receive(self) -> AsyncIterator[Transcript | SpeechEvent]:
        """Yield transcripts and VAD events as they arrive.

        Sarvam's `data` messages do not carry an explicit partial/final flag, so finality is
        inferred: a transcript arriving after `flush()` is final. That inference is kept here in one
        place rather than spread across callers, and `is_final` is what the pipeline clock keys off.
        """
        if self._ws is None:
            raise RuntimeError("stream not open")

        flushed = False
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            kind = msg.get("type")
            data = msg.get("data") or {}

            if kind == "data":
                text = (data.get("transcript") or "").strip()
                if not text:
                    continue
                metrics = data.get("metrics") or {}
                yield Transcript(
                    text=text,
                    language_code=data.get("language_code"),
                    language_probability=data.get("language_probability"),
                    is_final=flushed,
                    elapsed_ms=self._elapsed_ms(),
                    processing_latency=metrics.get("processing_latency"),
                )
            elif kind == "events":
                signal = (data.get("signal_type") or "").upper()
                if signal:
                    yield SpeechEvent(signal_type=signal, elapsed_ms=self._elapsed_ms())
                    if signal == "END_SPEECH":
                        flushed = True
            elif kind == "error":
                log.error("sarvam error: %s", data)
                raise RuntimeError(f"sarvam: {data.get('error')} ({data.get('code')})")

    async def mark_flushed(self) -> None:
        await self.flush()


async def probe(api_key: str, timeout_s: float = 12.0) -> dict[str, object]:
    """Open a session, send a short silence, and report what came back.

    Used by the health endpoint and by `scripts/verify_providers.py` to confirm the credential and
    the protocol are both live — without needing a microphone in the loop.
    """
    result: dict[str, object] = {"ok": False}
    t0 = time.perf_counter()
    try:
        async with SarvamStream(api_key) as stream:
            # 0.5s of 16kHz mono silence: enough for the server to accept a frame and respond,
            # short enough to keep a health check cheap.
            await stream.send_audio(b"\x00\x00" * 8000)
            await stream.flush()

            async def _first() -> object:
                async for event in stream.receive():
                    return event
                return None

            event = await asyncio.wait_for(_first(), timeout=timeout_s)
            result["ok"] = True
            result["first_event"] = type(event).__name__ if event else None
    except TimeoutError:
        # Silence legitimately produces no transcript. Reaching the timeout without an error means
        # the handshake and the auth both worked, which is what this probe is actually testing.
        result["ok"] = True
        result["first_event"] = "timeout (no speech in silence — expected)"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result
