"""Guardrails — four gates, each with a visible reason.

The task asks that the system demonstrably know when *not* to answer. Refusal is therefore a
first-class output with a named gate and a stated reason, not an early `return None` buried in the
pipeline. Every response carries a verdict, including the ones that pass, so the UI's refusal lamp
renders from measured state rather than from the absence of an answer.

The four gates, and why each exists
-----------------------------------

**Scope.** The corpus is ~10k MS MARCO queries' worth of passages. Most questions in the world are
not in it, and a retrieval system will always return *something* — its top-k is never empty, merely
sometimes meaningless. The scope gate thresholds the top dense cosine score against a value
calibrated on D3 against an out-of-domain probe set. Until that calibration exists the threshold is
`None` and the gate reports itself as uncalibrated rather than applying a guessed number; a guessed
threshold is worse than an absent one because it looks principled.

Note it thresholds the *dense cosine*, not the fused RRF score. RRF scores are rank-derived and
have no stable cross-query interpretation, so a fixed threshold on them would drift with how many
retrievers happened to return hits. See `fuse.py`.

**Safety.** A fast pattern screen over the transcript. Deliberately narrow: this system answers
factual questions from a fixed passage corpus, so the realistic risk is not that it produces
dangerous content — it cannot, since it can only quote MS MARCO passages — but that it lends an
authoritative interface to a harmful request. Refusing early also saves the retrieval work.

**Grounding.** Tier 1 needs no check; it is copied text. Tier 2 is verified after generation by
token overlap against the passages it cited. Below threshold, the generative answer is withheld and
Tier 1 stands, with the reason shown.

**Injection.** Retrieved passages are untrusted data. MS MARCO is web-scraped, so passages
containing imperative text are ordinary, not adversarial — but a passage reading "ignore previous
instructions" is indistinguishable from an attack once concatenated into a prompt. Passages are
wrapped in explicit delimiters and never interpolated as instructions; this gate additionally flags
passages carrying instruction-like patterns so the prompt builder can mark them.
"""

from __future__ import annotations

import re

from app.schemas import Gate, GuardVerdict
from app.stages.lexical import tokenize

# --- safety -------------------------------------------------------------------------------

# Narrow by design. Each pattern targets operational harm — synthesis routes, weapon manufacture,
# intrusion tooling — rather than topic keywords. A corpus question about the history of explosives
# is legitimate; a request for a synthesis procedure is not, and the distinction is the verb.
_UNSAFE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(how|steps?|guide|instructions?)\b.{0,40}\b(make|build|synthesi[sz]e|manufacture)\b.{0,40}\b(bomb|explosive|nerve agent|sarin|ricin|meth|methamphetamine|fentanyl)\b", re.I), "weapons or controlled-substance synthesis"),
    (re.compile(r"\b(kill|murder|poison|harm)\b.{0,30}\b(someone|person|my|him|her|them)\b", re.I), "violence against a person"),
    (re.compile(r"\b(child|minor|underage)\b.{0,25}\b(sexual|porn|explicit|nude)\b", re.I), "child sexual content"),
    (re.compile(r"\b(hack|ddos|exploit|ransomware|keylogger)\b.{0,40}\b(into|against|deploy|write|build)\b", re.I), "intrusion or malware tooling"),
    (re.compile(r"\b(how|best way)\b.{0,30}\b(kill myself|commit suicide|end my life)\b", re.I), "self-harm"),
)

# --- injection ----------------------------------------------------------------------------

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(the\s+)?(system|previous)\s+(prompt|instructions?)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.I),
    re.compile(r"</?(system|assistant|user)>", re.I),
    re.compile(r"\bnew\s+instructions?\s*:", re.I),
)

# Fraction of an answer's content tokens that must appear in its cited passages.
GROUNDING_THRESHOLD = 0.60


def check_safety(text: str) -> GuardVerdict:
    """Pattern screen over the user's transcript. Microseconds."""
    for pattern, reason in _UNSAFE_PATTERNS:
        if pattern.search(text):
            return GuardVerdict(
                allowed=False,
                gate=Gate.SAFETY,
                reason=f"Request matched an unsafe-content pattern: {reason}.",
            )
    return GuardVerdict(allowed=True)


def check_scope(top_dense_score: float | None, tau: float | None) -> GuardVerdict:
    """Abstain when the best retrieval is too weak to ground an answer.

    An uncalibrated gate reports `allowed=True` with an explicit reason. Failing open is the right
    default here — the corpus is benign, and refusing everything because a constant is unset would
    be a worse failure than answering a marginal query — but it is recorded so the state is
    visible in the response rather than silent.
    """
    if tau is None:
        return GuardVerdict(
            allowed=True,
            score=top_dense_score,
            threshold=None,
            reason="scope gate uncalibrated: no threshold set (see docs/CALIBRATION.md)",
        )
    if top_dense_score is None:
        return GuardVerdict(
            allowed=False,
            gate=Gate.SCOPE,
            reason="No passages retrieved for this query.",
            score=None,
            threshold=tau,
        )
    if top_dense_score < tau:
        return GuardVerdict(
            allowed=False,
            gate=Gate.SCOPE,
            reason="This isn't in my corpus — the closest passages aren't relevant enough to answer from.",
            score=top_dense_score,
            threshold=tau,
        )
    return GuardVerdict(allowed=True, score=top_dense_score, threshold=tau)


def check_injection(passage_texts: list[str]) -> list[int]:
    """Return indices of passages containing instruction-like text.

    Flagged, not dropped. These are usually benign web text, and silently removing them would
    degrade retrieval for honest queries; the prompt builder marks them instead.
    """
    return [
        i
        for i, text in enumerate(passage_texts)
        if any(p.search(text) for p in _INJECTION_PATTERNS)
    ]


def check_grounding(
    answer_text: str, cited_texts: list[str], *, threshold: float = GROUNDING_THRESHOLD
) -> GuardVerdict:
    """Verify a generated answer against the passages it claims to cite.

    Token containment rather than an LLM judge: a judge would cost another provider round-trip and
    another chance to be wrong, on the very path we are trying to keep fast and verifiable. Overlap
    is crude but has the property that matters — it cannot be fooled by fluency.

    Stopwords and citation markers are stripped first, so an answer is not credited for grounding
    on 'the' and 'of'.
    """
    if not cited_texts:
        return GuardVerdict(
            allowed=False,
            gate=Gate.GROUNDING,
            reason="generative answer withheld: cited no passages",
            score=0.0,
            threshold=threshold,
        )

    answer_tokens = set(tokenize(re.sub(r"\[\d+\]", " ", answer_text)))
    if not answer_tokens:
        return GuardVerdict(allowed=True, score=1.0, threshold=threshold)

    source_tokens: set[str] = set()
    for text in cited_texts:
        source_tokens |= set(tokenize(text))

    overlap = len(answer_tokens & source_tokens) / len(answer_tokens)
    if overlap < threshold:
        return GuardVerdict(
            allowed=False,
            gate=Gate.GROUNDING,
            reason=f"generative answer withheld: failed grounding check ({overlap:.0%} of terms traceable to cited passages, need {threshold:.0%})",
            score=overlap,
            threshold=threshold,
        )
    return GuardVerdict(allowed=True, score=overlap, threshold=threshold)
