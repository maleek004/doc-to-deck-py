"""Deterministic checks on every generated card: no second LLM call, just rules.

Mirrors AnkiX's `BulkCardGenerationService.EvaluateCardQuality`. The prompt tells the model these rules;
this gate checks it actually followed them. Layer 3 adds an LLM judge on top and measures both.
"""

import re
from collections.abc import Iterable

from doctodeck.pipeline.models import QualityFlag, QualityScore

MAX_FRONT_LENGTH = 150
MAX_BACK_LENGTH = 250
DUPLICATE_SIMILARITY = 0.80

# Any one of these makes a card "Needs Review" on its own.
CRITICAL_FLAGS = frozenset({QualityFlag.EMPTY_FRONT, QualityFlag.EMPTY_BACK, QualityFlag.BACK_TOO_LONG})

_QUESTION_STARTS = tuple(
    word.lower()
    for word in (
        "What", "How", "Why", "When", "Where", "Which", "Who",
        "Define", "Explain", "Describe", "Compare", "Identify", "Name",
    )
)  # fmt: skip
_STOP_WORDS = frozenset(
    {"because", "through", "between", "against", "without", "example", "explains", "provides", "including"}
)
_WORD_SEPARATORS = re.compile(r"[\s.,;:()\"'`]+")
_NON_WORD = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def evaluate(front: str, back: str, earlier_fronts: Iterable[str]) -> tuple[QualityScore, list[QualityFlag]]:
    """Score one card. `earlier_fronts` are the cards already accepted in this batch (duplicate check)."""
    flags: list[QualityFlag] = []

    if not front:
        flags.append(QualityFlag.EMPTY_FRONT)
    if not back:
        flags.append(QualityFlag.EMPTY_BACK)
    if len(front) > MAX_FRONT_LENGTH:
        flags.append(QualityFlag.FRONT_TOO_LONG)
    if len(back) > MAX_BACK_LENGTH:
        flags.append(QualityFlag.BACK_TOO_LONG)

    # A good cue is a real question: long enough, and either has a "?" or starts like one.
    if front and (len(front) < 15 or ("?" not in front and not front.lower().startswith(_QUESTION_STARTS))):
        flags.append(QualityFlag.AMBIGUOUS_CUE)

    if front and back and _leaks_answer(front, back):
        flags.append(QualityFlag.ANSWER_LEAK)

    normalized = _normalize(front)
    if len(normalized) > 10 and any(_is_near_duplicate(normalized, _normalize(other)) for other in earlier_fronts):
        flags.append(QualityFlag.POSSIBLE_DUPLICATE)

    if not flags:
        score = QualityScore.EXCELLENT
    elif len(flags) == 1 and flags[0] not in CRITICAL_FLAGS:
        score = QualityScore.GOOD
    else:
        score = QualityScore.NEEDS_REVIEW
    return score, flags


def _leaks_answer(front: str, back: str) -> bool:
    """The whole answer appears in the question, or a long, meaningful word from the answer does."""
    front_folded = front.casefold()
    if back.casefold() in front_folded:
        return True
    return any(
        word.casefold() in front_folded
        for word in _WORD_SEPARATORS.split(back)
        if len(word) >= 7 and word.lower() not in _STOP_WORDS
    )


def _normalize(text: str) -> str:
    return _SPACES.sub(" ", _NON_WORD.sub("", text.lower())).strip()


def _is_near_duplicate(a: str, b: str) -> bool:
    """Jaccard similarity of the two word sets: shared words / all distinct words."""
    if len(b) <= 10:
        return False
    if a == b:
        return True
    words_a, words_b = set(a.split(" ")), set(b.split(" "))
    return len(words_a & words_b) / len(words_a | words_b) >= DUPLICATE_SIMILARITY
