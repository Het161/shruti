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

# --- conversational intent ------------------------------------------------------------------
#
# The gate that actually catches the failure this system has.
#
# Three retrieval-derived signals were calibrated and all three failed (see
# lab/calibrate_scope.py and docs/BUILD_LOG.md): top-1 cosine AUC 0.713 at an unusable operating
# point, margin-based "peakedness" at or below chance, and lexical coverage at 0.520. The reason
# they failed is not that they were badly tuned — it is that they all answer the question "is
# there topically related text in the corpus", and the corpus is 310k passages of general web
# text, so the answer is essentially always yes.
#
# The real discriminator is grammatical, not semantic. This corpus answers *factual questions
# about the world*. "My name is Het Patel" is a self-introduction; "order me a pizza" is a
# command; "who created you" is about the assistant. None is an information-seeking question, and
# no amount of embedding quality changes that — it is a property of the utterance, not of the
# retrieval.
#
# So this gate is a pattern screen, and it is deliberately high-precision rather than
# high-recall: it fires only on constructions that are unambiguously not corpus questions. A
# missed refusal costs one bad answer; a false refusal on a real question makes the product
# useless. Measured on 500 in-domain queries and 70 out-of-domain probes — see the build log.
_CONVERSATIONAL: tuple[tuple[re.Pattern[str], str], ...] = (
    # Self-introduction and first-person-possessive personal facts. The corpus knows nothing
    # about the speaker, so these are unanswerable by construction rather than by chance.
    (re.compile(r"\bmy name is\b|\bi am called\b|\bi'?m called\b", re.I), "self-introduction"),
    (re.compile(r"\bmy (name|password|bank|account|balance|location|address|phone|email|birthday)\b", re.I), "personal information about the speaker"),
    # Trailing negative lookaheads matter here and `\b` will not do the job. Python's `\b` is
    # defined on word characters, and every Indic letter is a word character, so `আমার মা` happily
    # matched inside `আমার মাথায়` ("my head") and refused a legitimate medical question. The
    # lookahead asserts the next character is not another letter of the same script, which is the
    # word boundary that actually exists in these writing systems.
    (re.compile(r"मेरा नाम(?![ऀ-ॿ])|मेरे बैंक|मेरा पासवर्ड|मेरी माँ(?![ऀ-ॿ])", re.I), "personal information about the speaker"),
    (re.compile(r"મારું નામ(?![઀-૿])|મારા બેંક|મારો પાસવર્ડ|મારી માતા(?![઀-૿])", re.I), "personal information about the speaker"),
    (re.compile(r"আমার নাম(?![ঀ-৿])|আমার ব্যাংক|আমার পাসওয়ার্ড|আমার মা(?![ঀ-৿])", re.I), "personal information about the speaker"),
    (re.compile(r"என் பெயர்(?![஀-௿])|என் வங்கி|என் கடவுச்சொல்|என் அம்மா(?![஀-௿])", re.I), "personal information about the speaker"),
    # Questions addressed to the assistant rather than to the corpus.
    (re.compile(r"\b(what|who)('?s| is| are)\s+your\s+(name|model|purpose|version)\b", re.I), "a question about the assistant, not the corpus"),
    (re.compile(r"\bwho (made|created|built|trained|designed) you\b", re.I), "a question about the assistant, not the corpus"),
    (re.compile(r"\bare you (a |an )?(human|real|conscious|alive|robot|ai|bot)\b", re.I), "a question about the assistant, not the corpus"),
    (re.compile(r"\bhow are you\b|\bwhat are you thinking\b", re.I), "conversational small talk"),
    (re.compile(r"तुम्हारा नाम|तुम कैसे हो|तुम्हें किसने बनाया|क्या तुम इंसान", re.I), "a question about the assistant, not the corpus"),
    (re.compile(r"તમારું નામ|તમે કેમ છો|તમને કોણે બનાવ્યા|શું તમે માણસ", re.I), "a question about the assistant, not the corpus"),
    (re.compile(r"তোমার নাম|তুমি কেমন আছো|তোমাকে কে বানিয়েছে|তুমি কি মানুষ", re.I), "a question about the assistant, not the corpus"),
    (re.compile(r"உங்கள் பெயர்|எப்படி இருக்கிறீர்கள்|உங்களை யார் உருவாக்கியது|நீங்கள் மனிதரா", re.I), "a question about the assistant, not the corpus"),
    # Commands. A retrieval system cannot take actions in the world.
    (re.compile(r"\b(call|text|email|order|book|play|buy|send|delete|shut down|turn off|set)\s+(me|my|an?\s|the\s)", re.I), "a command, not a question"),
    (re.compile(r"\b(sing|tell)\s+me\s+(a|an|the)\b", re.I), "a command, not a question"),
    (re.compile(r"\bset an alarm\b|\bremind me\b|\bwrite me\b", re.I), "a command, not a question"),
    (re.compile(r"ऑर्डर करो|फोन करो|गाना गाओ|चुटकुला सुनाओ", re.I), "a command, not a question"),
    (re.compile(r"ઓર્ડર કરો|ફોન કરો|ગીત ગાઓ|જોક કહો", re.I), "a command, not a question"),
    (re.compile(r"অর্ডার করো|ফোন করো|গান গাও|কৌতুক বলো", re.I), "a command, not a question"),
    (re.compile(r"ஆர்டர் செய்யுங்கள்|அழைக்கவும்|பாடல் பாடுங்கள்|நகைச்சுவை சொல்லுங்கள்", re.I), "a command, not a question"),
)


def check_conversational(text: str) -> GuardVerdict:
    """Refuse utterances that are not factual questions about the world.

    Runs before retrieval, so a refusal costs microseconds rather than a full pipeline pass. The
    reason string names *why* it is out of scope, because "this isn't in my corpus" is far less
    useful to a user than "that's a question about me, not about what I've read".
    """
    for pattern, reason in _CONVERSATIONAL:
        if pattern.search(text):
            return GuardVerdict(
                allowed=False,
                gate=Gate.SCOPE,
                reason=f"Out of scope: this reads as {reason}. I answer factual questions from a "
                f"fixed passage corpus, and I have no information about you or the world outside it.",
            )
    return GuardVerdict(allowed=True)


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
