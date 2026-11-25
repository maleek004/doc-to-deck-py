from doctodeck.pipeline.models import QualityFlag, QualityScore
from doctodeck.pipeline.quality_gate import evaluate


def check(front: str, back: str, *earlier: str):
    return evaluate(front, back, earlier)


def test_a_clean_question_and_answer_is_excellent():
    score, flags = check("What does await do to the running coroutine?", "Pauses it until the result is ready.")

    assert score == QualityScore.EXCELLENT
    assert flags == []


def test_a_front_that_is_not_a_question_is_an_ambiguous_cue():
    _, flags = check("Async functions and their return values", "They return coroutine objects.")

    assert QualityFlag.AMBIGUOUS_CUE in flags


def test_a_long_answer_word_repeated_in_the_question_is_a_leak():
    _, flags = check("What keyword suspends a coroutine until a result is ready?", "The await keyword suspends it.")

    assert QualityFlag.ANSWER_LEAK in flags  # "keyword" and "suspends" appear on both sides


def test_an_answer_over_250_characters_needs_review_on_its_own():
    score, flags = check("What is a coroutine?", "x" * 251)

    assert QualityFlag.BACK_TOO_LONG in flags
    assert score == QualityScore.NEEDS_REVIEW  # critical flag: one is enough


def test_a_near_identical_question_is_a_possible_duplicate():
    _, flags = check("What does the await keyword do?", "Pauses the coroutine.", "What does the await keyword do here?")

    assert QualityFlag.POSSIBLE_DUPLICATE in flags  # 6 shared words of 7 distinct = 0.86


def test_one_minor_flag_is_still_good():
    score, flags = check("The return value of an async function", "A coroutine object.")

    assert flags == [QualityFlag.AMBIGUOUS_CUE]
    assert score == QualityScore.GOOD
