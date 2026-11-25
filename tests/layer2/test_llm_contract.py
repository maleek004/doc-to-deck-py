"""The contract itself: output parsing, the schema the models are sent, the fake, the registry."""

import pytest

from doctodeck.llm.contracts import LlmError, LlmErrorCode, LlmRequest, parse_output, strip_code_fences
from doctodeck.llm.fake import FakeLlmClient
from doctodeck.llm.registry import LlmRegistry
from doctodeck.pipeline.models import CardsOutput, PlanOutput


@pytest.mark.parametrize(
    "raw",
    [
        '{"cards": []}',
        '```json\n{"cards": []}\n```',
        '```JSON\n{"cards": []}\n```',
        '```\n{"cards": []}\n```',
        '  \n```json {"cards": []} ```  \n',
    ],
)
def test_code_fences_around_json_are_stripped(raw):
    assert strip_code_fences(raw) == '{"cards": []}'


def test_text_that_is_not_json_is_an_invalid_output_error():
    with pytest.raises(LlmError) as caught:
        parse_output("Sure! Here are your cards: ...", CardsOutput, "fake")

    assert caught.value.code == LlmErrorCode.INVALID_OUTPUT
    assert caught.value.__cause__ is not None  # Pydantic's ValidationError, kept for the logs


def test_json_of_the_wrong_shape_is_an_invalid_output_error():
    with pytest.raises(LlmError) as caught:
        parse_output('{"cards": [{"front": "Q?"}]}', CardsOutput, "fake")  # back, moduleIndex... missing

    assert caught.value.code == LlmErrorCode.INVALID_OUTPUT


def test_answers_use_camel_case_keys_as_the_schema_says():
    plan = parse_output(
        '{"modules":[{"title":"A","description":"d","estimatedCards":4,"sectionIndices":[0]}]}', PlanOutput, "fake"
    )

    assert plan.modules[0].estimated_cards == 4


def test_the_schema_sent_to_the_model_comes_from_the_pydantic_model():
    schema = LlmRequest("plan", "system", "user", PlanOutput).json_schema()

    module = schema["$defs"]["PlanOutputModule"]
    assert set(module["required"]) == {"title", "description", "estimatedCards", "sectionIndices"}
    assert "3 to 30" in module["properties"]["estimatedCards"]["description"]  # instructions travel in the schema


async def test_scripted_answers_are_used_in_order_then_the_demo_takes_over():
    fake = FakeLlmClient().returns('{"cards": []}').fails(LlmErrorCode.RATE_LIMITED)
    request = LlmRequest("cards", "system", "user", CardsOutput)

    first = await fake.complete(request)
    with pytest.raises(LlmError) as caught:
        await fake.complete(request)

    assert first.data.cards == []
    assert caught.value.code == LlmErrorCode.RATE_LIMITED
    assert len(fake.requests) == 2


async def test_the_fake_reports_usage():
    fake = FakeLlmClient().returns('{"cards": []}')

    result = await fake.complete(LlmRequest("cards", "s" * 40, "u" * 40, CardsOutput))

    assert (result.usage.provider, result.usage.model) == ("fake", "fake-demo")
    assert result.usage.prompt_tokens == 20  # 80 characters / 4
    assert result.usage.latency_ms >= 0


def test_registry_lookup_ignores_case_and_rejects_unknown_providers():
    fake = FakeLlmClient()
    registry = LlmRegistry([fake])

    assert registry.get(" FAKE ") is fake
    with pytest.raises(LlmError) as caught:
        registry.get("gpt-9")
    assert caught.value.code == LlmErrorCode.UNKNOWN_PROVIDER
    assert "Available: fake" in caught.value.message
