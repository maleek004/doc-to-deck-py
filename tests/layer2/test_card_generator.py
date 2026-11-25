import json

import pytest

from doctodeck.llm.contracts import LlmError, LlmErrorCode
from doctodeck.llm.fake import FakeLlmClient
from doctodeck.pipeline.models import PipelineOptions, PipelineRequestError, QualityFlag, QualityScore
from tests.layer2.data import TWO_MODULES, generate_request, pipeline


def cards_json(*cards: tuple[str, str, int, str]) -> str:
    return json.dumps(
        {"cards": [{"front": f, "back": b, "moduleIndex": m, "topicTag": t, "rationale": ""} for f, b, m, t in cards]}
    )


async def test_demo_fake_makes_cards_tagged_with_their_module():
    _, generator = pipeline(FakeLlmClient())

    result = await generator.generate(generate_request(TWO_MODULES))

    assert [c.module_index for c in result.cards] == [1, 1, 2, 2]
    assert result.cards[2].module_title == "Cancellation"
    assert result.truncated is False


async def test_only_the_picked_modules_are_sent_to_the_model():
    fake = FakeLlmClient()
    _, generator = pipeline(fake)

    await generator.generate(generate_request([TWO_MODULES[1]]))

    (request,) = fake.requests
    assert "Timeouts" in request.user_prompt
    assert '"title":"Async"' not in request.user_prompt


async def test_long_text_is_cut_and_the_cut_is_reported():
    fake = FakeLlmClient()
    _, generator = pipeline(fake, PipelineOptions(max_source_chars=100))

    result = await generator.generate(generate_request(TWO_MODULES, text="a" * 500))

    assert result.truncated is True
    assert (result.sent_chars, result.total_chars) == (100, 500)
    assert "a" * 101 not in fake.requests[0].user_prompt


async def test_the_cut_never_splits_an_emoji():
    _, generator = pipeline(FakeLlmClient(), PipelineOptions(max_source_chars=100))

    result = await generator.generate(generate_request(TWO_MODULES, text="a" * 99 + "😀" + "b" * 50))

    assert result.sent_chars == 100  # 99 letters + the whole emoji: Python counts code points, not UTF-16 units


async def test_a_card_with_an_unknown_module_index_is_reassigned_and_reported():
    fake = FakeLlmClient().returns(
        cards_json(
            ("What does await hand back?", "Control of the event loop.", 1, "Async"),
            ("What stops a task early?", "Cancelling it.", 7, "cancel"),
        )
    )
    _, generator = pipeline(fake)

    result = await generator.generate(generate_request(TWO_MODULES))

    assert [c.module_index for c in result.cards] == [1, 2]
    assert any("moduleIndex 7" in r for r in result.repairs)
    assert result.cards[0].topic_tag == "async"  # tags are lowercased


async def test_every_card_goes_through_the_quality_gate():
    card = ("What does await hand back?", "Control of the event loop.", 1, "a")
    _, generator = pipeline(FakeLlmClient().returns(cards_json(card, card)))

    result = await generator.generate(generate_request(TWO_MODULES))

    assert result.cards[0].quality_score == QualityScore.EXCELLENT
    assert QualityFlag.POSSIBLE_DUPLICATE in result.cards[1].quality_flags


async def test_empty_cards_are_skipped_and_a_batch_of_only_empty_cards_is_invalid_output():
    _, generator = pipeline(FakeLlmClient().returns(cards_json((" ", "", 1, "x"))))

    with pytest.raises(LlmError) as caught:
        await generator.generate(generate_request(TWO_MODULES))

    assert caught.value.code == LlmErrorCode.INVALID_OUTPUT


async def test_no_modules_picked_is_a_bad_request():
    _, generator = pipeline(FakeLlmClient())

    with pytest.raises(PipelineRequestError):
        await generator.generate(generate_request([]))


async def test_provider_errors_pass_through_unchanged():
    _, generator = pipeline(FakeLlmClient().fails(LlmErrorCode.RATE_LIMITED, "Busy."))

    with pytest.raises(LlmError) as caught:
        await generator.generate(generate_request(TWO_MODULES))

    assert caught.value.code == LlmErrorCode.RATE_LIMITED
