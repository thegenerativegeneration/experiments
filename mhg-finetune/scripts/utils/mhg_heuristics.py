"""
MHG language-detection heuristics.

Middle High German has a distinctive orthographic and morphological fingerprint
that makes lightweight rule-based detection feasible.  These helpers are used
by clean_data.py to flag responses that have slipped into Modern German or
another language.

Rules are intentionally conservative (low false-positive rate) because
overly aggressive filtering would discard rare but valid MHG forms.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Core word lists
# ---------------------------------------------------------------------------

# High-frequency MHG function words that are absent in Modern German.
MHG_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "daz", "dâ", "diu", "des", "dem", "den", "der",
        "ich", "du", "er", "sî", "wir", "ir", "sie",
        "ze", "von", "an", "in", "ûf", "mit", "nâch", "durch",
        "ouch", "hât", "hâst", "ist", "sint", "wart", "wâren",
        "sol", "mac", "muoz", "wil", "kan", "tuot", "tuon",
        "niht", "noch", "und", "oder", "sô", "als", "swaz",
        "swer", "wen", "wem", "waz", "wie", "wâ",
        "mîn", "dîn", "sîn", "unser", "iuwer",
        "got", "man", "wîp", "herze", "lîp",
    }
)

# Modern German words that should not appear in a genuine MHG response.
# Focused on high-frequency words with unambiguous orthographic changes.
# Note: tokens that also appear in MHG_FUNCTION_WORDS (e.g. "noch", "ist")
# are intentionally excluded here to avoid self-contradictory scoring.
MODERN_GERMAN_MARKERS: frozenset[str] = frozenset(
    {
        "dass", "wenn", "weil", "aber", "jedoch", "obwohl", "trotzdem",
        "heute", "jetzt", "schon", "sehr",
        "haben", "werden", "würde", "würden",
        "können", "müssen", "sollen", "wollen", "dürfen",
        "machen", "sagen", "gehen", "kommen", "sehen",
        "sein", "sind", "war", "waren",   # Modern German forms; "ist" excluded (shared with MHG)
        "nicht",  # MHG uses "niht"
        "das",    # MHG uses "daz"
        "für",    # MHG uses "für/vür" but NOT in function-word role
    }
)

# Sanity-check: the two sets must be disjoint so that shared tokens are never
# counted as Modern German contamination in genuine MHG text.
assert not (MHG_FUNCTION_WORDS & MODERN_GERMAN_MARKERS), (
    "Overlap between MHG_FUNCTION_WORDS and MODERN_GERMAN_MARKERS: "
    + str(MHG_FUNCTION_WORDS & MODERN_GERMAN_MARKERS)
)

# Characteristic MHG diacritic patterns (circumflex-marked long vowels).
MHG_LONG_VOWEL_RE = re.compile(r"[âêîôûæœ]")

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class MHGScore(NamedTuple):
    mhg_word_hits: int
    modern_marker_hits: int
    long_vowel_hits: int
    token_count: int

    @property
    def mhg_density(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.mhg_word_hits / self.token_count

    @property
    def modern_contamination(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.modern_marker_hits / self.token_count

    @property
    def is_likely_mhg(self) -> bool:
        return (
            self.mhg_density >= 0.05
            and self.modern_contamination <= 0.06
        )


def _tokenise(text: str) -> list[str]:
    """Lowercase word-tokenise (letters and umlauts only)."""
    return re.findall(r"[a-zäöüâêîôûæœàèìòùëïü]+", text.lower())


def score_text(text: str) -> MHGScore:
    """Return an :class:`MHGScore` for *text*."""
    tokens = _tokenise(text)
    token_set = set(tokens)
    mhg_hits = len(token_set & MHG_FUNCTION_WORDS)
    modern_hits = len(token_set & MODERN_GERMAN_MARKERS)
    long_vowel_hits = len(MHG_LONG_VOWEL_RE.findall(text.lower()))
    return MHGScore(
        mhg_word_hits=mhg_hits,
        modern_marker_hits=modern_hits,
        long_vowel_hits=long_vowel_hits,
        token_count=len(tokens),
    )


def is_likely_mhg(text: str) -> bool:
    """Return True if *text* looks like Middle High German."""
    return score_text(text).is_likely_mhg


def has_modern_contamination(text: str, threshold: float = 0.06) -> bool:
    """Return True if the Modern German marker density exceeds *threshold*."""
    return score_text(text).modern_contamination > threshold
