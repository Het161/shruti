"""Stage protocol and the error taxonomy the orchestrator dispatches on.

Every stage is a callable with a declared input type, a declared output type, and a name that
appears in the waterfall. The orchestrator does not know what any stage does — it knows how to time
one, bound one, and classify its failure. That separation is what lets the pipeline degrade
gracefully instead of 500-ing.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, TypeVar, runtime_checkable

TIn = TypeVar("TIn", contravariant=True)
TOut = TypeVar("TOut", covariant=True)


class ErrorKind(str, Enum):
    """Failure taxonomy.

    The distinction that matters operationally is *recoverable* vs not. A `PROVIDER_TIMEOUT` on
    generation means Tier 1 stands alone and the user still gets an answer. An `INDEX_ERROR` means
    retrieval itself is broken and there is nothing honest to return.
    """

    TIMEOUT = "timeout"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    INDEX_ERROR = "index_error"
    INVALID_INPUT = "invalid_input"
    INTERNAL = "internal"

    @property
    def degradable(self) -> bool:
        """True when the pipeline can return a lower tier rather than fail the request."""
        return self in {
            ErrorKind.PROVIDER_TIMEOUT,
            ErrorKind.PROVIDER_ERROR,
            ErrorKind.PROVIDER_RATE_LIMIT,
            ErrorKind.TIMEOUT,
        }


class StageError(Exception):
    """Raised by a stage; carries enough structure for the orchestrator to decide what to do."""

    def __init__(
        self,
        stage: str,
        kind: ErrorKind,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(f"[{stage}/{kind.value}] {message}")
        self.stage = stage
        self.kind = kind
        self.message = message
        self.cause = cause

    def as_dict(self) -> dict[str, str]:
        return {"stage": self.stage, "kind": self.kind.value, "message": self.message}


@runtime_checkable
class Stage(Protocol[TIn, TOut]):
    """Structural contract every pipeline stage satisfies."""

    name: str

    def __call__(self, payload: TIn) -> TOut: ...
