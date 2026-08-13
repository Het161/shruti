"""Monotonic stage timing.

The product's central claim is a latency number, so the clock is not an afterthought bolted on
near the end — it is the first module written and every stage is measured through it.

Two rules this module exists to enforce:

1. **Monotonic only.** `time.perf_counter_ns()`, never `time.time()`. Wall-clock can step
   backwards under NTP correction, which would silently produce negative or absurd durations in
   published benchmark numbers.
2. **Ordered segments.** The UI renders a waterfall, so a stage carries both its duration and its
   offset from request start. A bare `dict[str, float]` cannot draw a waterfall; a list of spans
   with offsets can.

Overhead is roughly 100ns per `stage()` entry/exit pair — about 0.0001ms against a 200ms budget,
i.e. below the noise floor of what we report.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

_NS_PER_MS = 1_000_000.0


def _now_ns() -> int:
    return time.perf_counter_ns()


def new_request_id() -> str:
    """Short, sortable-enough id for correlating logs with a UI waterfall."""
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True, slots=True)
class Span:
    """A single measured stage.

    `offset_ms` is measured from timer start, which is what lets the frontend lay segments
    end-to-end without inferring order from a dict's iteration order.
    """

    name: str
    offset_ms: float
    duration_ms: float


@dataclass(slots=True)
class RequestTimer:
    """Per-request stopwatch.

    Usage::

        timer = RequestTimer()
        with timer.stage("embed"):
            vec = embed(text)
        ...
        breakdown = timer.snapshot()

    Nested stages are not supported by design: a waterfall with overlapping segments misrepresents
    where time actually went. Stages are sequential and disjoint, matching the pipeline itself.
    """

    request_id: str = field(default_factory=new_request_id)
    _t0: int = field(default_factory=_now_ns)
    _spans: list[Span] = field(default_factory=list)
    _open: str | None = field(default=None, repr=False)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self._open is not None:
            raise RuntimeError(
                f"stage({name!r}) opened while {self._open!r} is still open; "
                "stages must be sequential and disjoint so the waterfall stays truthful"
            )
        self._open = name
        start = _now_ns()
        try:
            yield
        finally:
            end = _now_ns()
            self._open = None
            self._spans.append(
                Span(
                    name=name,
                    offset_ms=(start - self._t0) / _NS_PER_MS,
                    duration_ms=(end - start) / _NS_PER_MS,
                )
            )

    def mark(self, name: str, duration_ms: float) -> None:
        """Record a stage measured elsewhere.

        Needed for work whose clock starts outside this process — chiefly streaming STT, where the
        meaningful duration is 'last audio chunk sent to final transcript received' and is timed by
        the WebSocket handler. Recorded as a real span so it appears in the waterfall rather than
        vanishing into an unlabelled gap.
        """
        self._spans.append(
            Span(
                name=name,
                offset_ms=(_now_ns() - self._t0) / _NS_PER_MS - duration_ms,
                duration_ms=duration_ms,
            )
        )

    def elapsed_ms(self) -> float:
        """Total time since the timer started."""
        return (_now_ns() - self._t0) / _NS_PER_MS

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def snapshot(self) -> dict[str, object]:
        """Serialisable breakdown. Feeds both the API response and the structured log line."""
        spans = self._spans
        measured = sum(s.duration_ms for s in spans)
        total = self.elapsed_ms()
        return {
            "request_id": self.request_id,
            "stages": [
                {
                    "name": s.name,
                    "offset_ms": round(s.offset_ms, 3),
                    "duration_ms": round(s.duration_ms, 3),
                }
                for s in spans
            ],
            "measured_ms": round(measured, 3),
            "total_ms": round(total, 3),
            # Time inside the request not attributed to any named stage. If this grows, the
            # instrumentation has a hole and the waterfall is lying by omission — so it is
            # published rather than swept up into a neighbouring stage.
            "unattributed_ms": round(max(0.0, total - measured), 3),
        }
