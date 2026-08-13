"""Script-based language detection.

A statistical language identifier costs milliseconds and a model load. Detecting the *script* costs
a few microseconds and, for this corpus, is nearly as informative: the indexed languages occupy
disjoint Unicode blocks, so a single pass over the string resolves them.

The one genuine ambiguity is stated rather than hidden. Devanagari is shared by Hindi, Marathi,
Nepali, and Sanskrit — all four are in MSMARCO-XI. Script detection cannot separate them, so
Devanagari resolves to Hindi and `ambiguous_with` records what else it might have been. Since only
Hindi is indexed, this is currently exact; if Marathi is ever added, this is the function that must
grow a real classifier, and the field is here so that need is visible rather than discovered.
"""

from __future__ import annotations

from dataclasses import dataclass

# Unicode block ranges, checked in order. Ranges are inclusive.
_BLOCKS: tuple[tuple[int, int, str], ...] = (
    (0x0900, 0x097F, "hi"),  # Devanagari — also mr, ne, sa
    (0x0980, 0x09FF, "bn"),  # Bengali — also as
    (0x0A00, 0x0A7F, "pa"),  # Gurmukhi
    (0x0A80, 0x0AFF, "gu"),  # Gujarati
    (0x0B00, 0x0B7F, "or"),  # Odia
    (0x0B80, 0x0BFF, "ta"),  # Tamil
    (0x0C00, 0x0C7F, "te"),  # Telugu
    (0x0C80, 0x0CFF, "kn"),  # Kannada
    (0x0D00, 0x0D7F, "ml"),  # Malayalam
    (0x0600, 0x06FF, "ur"),  # Arabic block — Urdu
)

_AMBIGUOUS: dict[str, tuple[str, ...]] = {
    "hi": ("mr", "ne", "sa"),
    "bn": ("as",),
}


@dataclass(frozen=True, slots=True)
class LangGuess:
    lang: str
    confidence: float
    ambiguous_with: tuple[str, ...] = ()
    code_mixed: bool = False


def detect(text: str) -> LangGuess:
    """Identify the dominant script of `text`.

    Confidence is the share of alphabetic characters belonging to the winning script. Code-mixing
    is flagged when a second script holds a meaningful minority — which matters because Sarvam is
    chosen partly for code-mix handling, and the UI should say so when it happens rather than
    silently label a Hinglish utterance as pure Hindi.
    """
    counts: dict[str, int] = {}
    latin = 0
    total = 0

    for ch in text:
        cp = ord(ch)
        if not ch.isalpha():
            continue
        total += 1
        if cp < 0x0250:
            latin += 1
            counts["en"] = counts.get("en", 0) + 1
            continue
        for lo, hi, code in _BLOCKS:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break

    if total == 0 or not counts:
        return LangGuess(lang="en", confidence=0.0)

    lang, n = max(counts.items(), key=lambda kv: kv[1])
    confidence = n / total

    # A minority script above this share is treated as genuine code-mixing rather than noise from
    # a stray loanword or digit-adjacent character.
    others = sorted((v / total, k) for k, v in counts.items() if k != lang)
    code_mixed = bool(others and others[-1][0] >= 0.15)

    return LangGuess(
        lang=lang,
        confidence=confidence,
        ambiguous_with=_AMBIGUOUS.get(lang, ()),
        code_mixed=code_mixed,
    )


def retrieval_langs(detected: str) -> list[str]:
    """Language partitions to search for a query in `detected`.

    Always includes English alongside the detected language. MSMARCO-XI is translated *from*
    English, so the English passages are the originals — they carry no translation loss, and for a
    query whose answer is a name, a number, or a technical term, the English passage is frequently
    the better match even when the user is speaking Gujarati.
    """
    return ["en"] if detected == "en" else [detected, "en"]
