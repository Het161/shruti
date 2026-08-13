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
    # --- weapons / controlled substances ---
    (re.compile(r"\b(how|steps?|guide|instructions?|way)\b.{0,40}\b(make|build|synthesi[sz]e|manufacture|create)\b.{0,40}\b(bomb|explosive|nerve agent|sarin|ricin|meth|methamphetamine|fentanyl|napalm|thermite)\b", re.I), "weapons or controlled-substance synthesis"),
    (re.compile(r"\b(बम|विस्फोटक)\b.{0,25}(कैसे|बनाऊ|बनाए)", re.I), "weapons or controlled-substance synthesis"),
    (re.compile(r"(બોમ્બ|વિસ્ફોટક).{0,25}(કેવી રીતે|બનાવ)", re.I), "weapons or controlled-substance synthesis"),

    # --- violence against a person ---
    # The English rule no longer demands a separate object word: "how to hurt people" and "how do I
    # kill someone" both have to trip it.
    (re.compile(r"\b(kill|murder|poison|hurt|harm|attack|stab|shoot)\b\s+(someone|somebody|a person|people|my |him|her|them|his |their )", re.I), "violence against a person"),
    (re.compile(r"\bhow (do i|to|can i)\b.{0,20}\b(kill|murder|poison|hurt|harm)\b", re.I), "violence against a person"),
    # Indic: match the *intent* construction, not the bare noun. "नुकसान" alone means damage/loss
    # and appears throughout the corpus in legal and medical passages — gating on it would refuse
    # legitimate questions about car-accident damages, which is exactly what the corpus contains.
    (re.compile(r"(किसी को|किसीको).{0,20}(नुकसान|चोट|हानि).{0,12}(पहुंचा|पहुँचा|कैसे)|(नुकसान|चोट).{0,10}(पहुंचाऊं|पहुँचाऊं|पहुंचाना)", re.I), "violence against a person"),
    (re.compile(r"(किसी को|किसीको).{0,15}(मार|कत्ल|जान से)", re.I), "violence against a person"),
    (re.compile(r"(કોઈને).{0,20}(નુકસાન|ઈજા).{0,12}(પહોંચાડ|કેવી રીતે)|(કોઈને).{0,15}(મારવ|મારી)", re.I), "violence against a person"),
    (re.compile(r"(কাউকে).{0,20}(ক্ষতি|আঘাত).{0,12}(করব|কিভাবে)|(কাউকে).{0,15}(মার|খুন)", re.I), "violence against a person"),
    (re.compile(r"(யாரையாவது|ஒருவரை).{0,20}(கொல்ல|காயப்படுத்த|தீங்கு)", re.I), "violence against a person"),

    # --- CSAM ---
    (re.compile(r"\b(child|minor|underage|teen)\b.{0,25}\b(sexual|porn|explicit|nude|naked)\b", re.I), "child sexual content"),

    # --- intrusion / account takeover ---
    # Previously required a second verb ("hack ... into"), so "how to hack a bank account" sailed
    # through. Now the target noun alone is enough, which is what the request actually is.
    (re.compile(r"\b(hack|hacking|crack|breach|break into|bypass)\b.{0,25}\b(account|bank|password|wi-?fi|email|phone|database|server|someone|system|login)\b", re.I), "intrusion or account takeover"),
    (re.compile(r"\b(ddos|ransomware|keylogger|rootkit|botnet|phishing (kit|page|site))\b", re.I), "malware or attack tooling"),
    (re.compile(r"\b(steal|stealing)\b.{0,20}\b(password|credit card|identity|data|account)\b", re.I), "credential or identity theft"),
    (re.compile(r"(हैक|हैकिंग).{0,20}(खाता|अकाउंट|पासवर्ड|कैसे)", re.I), "intrusion or account takeover"),

    # --- doxxing / private personal data about a third party ---
    (re.compile(r"\b(someone'?s|somebody'?s|his|her|their|a person'?s)\s+(private|home|personal|residential)\s*(address|number|phone|details|information)\b", re.I), "private personal information about a third party"),
    (re.compile(r"\b(give|find|get|tell)\s+me\b.{0,25}\b(private|home)\s+address\b", re.I), "private personal information about a third party"),
    (re.compile(r"\b(find|track|locate)\b.{0,20}\b(someone'?s|his|her|their)\s+(location|address|phone)\b", re.I), "private personal information about a third party"),
    (re.compile(r"\b(social security number|ssn|aadhaar number|credit card number)\b.{0,25}\b(of|for)\b", re.I), "private personal information about a third party"),

    # --- self-harm ---
    (re.compile(r"\b(how|best way|easiest way)\b.{0,30}\b(kill myself|commit suicide|end my life|hurt myself)\b", re.I), "self-harm"),
    (re.compile(r"\b(kill myself|commit suicide|end my life)\b", re.I), "self-harm"),
    (re.compile(r"आत्महत्या.{0,15}(कैसे|तरीका)", re.I), "self-harm"),
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
    # Romanised Hindi/Gujarati possessives. Voice input from Indian users is frequently romanised,
    # and a gate that only reads native script misses half its traffic.
    (re.compile(r"\b(mera|meri|mere)\s+(naam|exam|ghar|phone|account|password|hackathon|interview|order|booking)\b", re.I), "personal information about the speaker"),
    (re.compile(r"\b(what|how)\s+did\s+i\b|\bhow was my\b|\bwhen is my\b", re.I), "personal information about the speaker"),
    (re.compile(r"\bmy\s+(exam|interview|meeting|flight|hackathon|order|booking|schedule|homework)\b", re.I), "personal information about the speaker"),
    (re.compile(r"मेरा\s+(अगला|पिछला)|मेरी\s+(परीक्षा|फ्लाइट)|मेरा\s+(एग्जाम|हैकाथॉन|इंटरव्यू)", re.I), "personal information about the speaker"),
    # Nearby / here — a static corpus has no location context for the speaker.
    (re.compile(r"\bnearest\b|\bnear me\b|\bclosest\b.{0,15}\b(to me|here)\b|\baas ?paas\b|\bkaha hai\b", re.I), "a request about the speaker's surroundings"),

    # --- current world state, beyond a static corpus ---
    # Years at or after 2020 are past MS MARCO's collection window entirely, which makes this a
    # rare high-precision signal: no passage in the corpus can speak to them.
    (re.compile(r"\b(20[2-9]\d|21\d\d)\b", re.I), "an event after this corpus was collected"),
    (re.compile(r"\b(last|this|next)\s+(week|month|night|year)\b|\byesterday\b|\btomorrow\b|\bright now\b|\bcurrently\b|\bat the moment\b", re.I), "a question about current events or the present moment"),
    (re.compile(r"\b(latest|newest|most recent)\b", re.I), "a question about current events or the present moment"),
    # No `\b` on Indic tokens. Many of these words end in a combining vowel sign (Unicode category
    # Mn), which Python's `\w` does not match — so `\bઅત્યારે\b` finds no boundary after the final
    # sign and silently never fires. It is the same defect as the `আমার মা` prefix bug earlier in
    # this file, reintroduced by writing these patterns in the habitual ASCII style. Verified:
    # `\bઅત્યારે\b.{0,14}હવામાન` does not match "અમદાવાદ માં અત્યારે હવામાન કેવું છે"; without the
    # `\b` it does.
    (re.compile(r"आज.{0,12}(सेंसेक्स|कीमत|भाव|मौसम)|अभी.{0,12}(कितना|क्या)|पिछले\s+(हफ्ते|सप्ताह)", re.I), "a question about current events or the present moment"),
    (re.compile(r"અત્યારે.{0,16}(હવામાન|કિંમત|ભાવ|તાપમાન)|આજે.{0,14}(સેન્સેક્સ|ભાવ|હવામાન)", re.I), "a question about current events or the present moment"),
    (re.compile(r"\baaj\b.{0,14}\b(sensex|nifty|price|weather|mausam)\b|\babhi\b.{0,12}\b(kitna|kya)\b", re.I), "a question about current events or the present moment"),
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


def check_degenerate(text: str, lexical_hits: int) -> GuardVerdict:
    """Refuse input with no usable lexical content at all.

    This exists because a cosine threshold provably cannot do the job. Measured against the
    regression set, the highest-scoring nonsense input ("the the the the the", 0.5248) outranks the
    lowest-scoring legitimate question ("where is the taj mahal located", 0.4877) — the windows
    overlap, so **no** value of tau both refuses gibberish and answers a Taj Mahal question.

    Lexical evidence separates them cleanly where similarity cannot:

    - `tokenize()` returning nothing means the input was punctuation or stopwords only —
      "।।।।।।" and "the the the the the" both collapse to zero content tokens.
    - BM25 returning nothing means not one query term appears anywhere in 310k passages, which for
      real questions essentially never happens. "asdf qwerty zxcv" is out-of-vocabulary by
      construction; "what is inflation" is not.

    A dense retriever will always return its nearest neighbours no matter how meaningless the
    query, because every vector has a nearest vector. Absence of lexical evidence is the signal
    that the input was never language about this corpus in the first place.
    """
    if not tokenize(text):
        return GuardVerdict(
            allowed=False,
            gate=Gate.SCOPE,
            reason="No searchable content in that input — it is punctuation or common words only.",
        )
    # Digits alone are not a question. `str.isnumeric()` is Unicode-aware, so this covers Gujarati
    # ૧૨૩૪૫૬ and Devanagari १२३४५६ as well as ASCII — which matters, because a bare numeral string
    # does produce BM25 hits (the corpus is full of statute and ZIP-code numbers) and so slips
    # past the no-lexical-evidence check below.
    stripped = "".join(ch for ch in text if not ch.isspace() and ch.isalnum())
    if stripped and stripped.isnumeric():
        return GuardVerdict(
            allowed=False,
            gate=Gate.SCOPE,
            reason="That is a number with no question attached — there is nothing to look up.",
        )
    if lexical_hits == 0:
        return GuardVerdict(
            allowed=False,
            gate=Gate.SCOPE,
            reason="None of those words appear anywhere in the corpus, so there is nothing to "
            "answer from.",
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
            reason=(
                "The corpus has no strong match for this. Retrieval scored "
                f"{top_dense_score:.3f} against a floor of {tau:.3f} — the weakest 5% of what "
                "normal questions produce — so answering would mean guessing from loosely "
                "related passages."
            ),
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
