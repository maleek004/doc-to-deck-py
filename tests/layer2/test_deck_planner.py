import pytest

from doctodeck.llm.contracts import LlmError, LlmErrorCode
from doctodeck.llm.fake import FakeLlmClient
from doctodeck.pipeline.models import PipelineOptions, PipelineRequestError
from tests.layer2.data import pipeline, plan_request


def module(title: str, indices: list[int], cards: int = 5) -> str:
    return f'{{"title":"{title}","description":"","estimatedCards":{cards},"sectionIndices":{indices}}}'


async def test_demo_fake_groups_sections_into_modules_in_order():
    planner, _ = pipeline(FakeLlmClient())

    result = await planner.plan(plan_request(word_count=6000))  # 6,000 words → 3 modules wanted

    assert len(result.modules) == 3
    assert [i for m in result.modules for i in m.section_indices] == [0, 1, 2, 3]
    assert [m.module_index for m in result.modules] == [1, 2, 3]
    assert result.usage.provider == "fake"
    assert result.repairs == []


async def test_the_prompt_sends_the_outline_not_the_full_text():
    fake = FakeLlmClient()
    planner, _ = pipeline(fake)

    await planner.plan(plan_request())

    (request,) = fake.requests
    assert request.purpose == "plan"
    assert '"title":"Cancellation"' in request.user_prompt
    assert '"snippet":' in request.user_prompt


async def test_hallucinated_and_duplicate_section_indices_are_dropped():
    fake = FakeLlmClient().returns(f'{{"modules":[{module("A", [0, 1, 9])},{module("B", [1, 2, 3])}]}}')
    planner, _ = pipeline(fake)

    result = await planner.plan(plan_request())

    assert result.modules[0].section_indices == [0, 1]  # 9 doesn't exist
    assert result.modules[1].section_indices == [2, 3]  # 1 was already in A
    assert len(result.repairs) == 2


async def test_sections_the_model_forgot_join_the_module_before_them():
    fake = FakeLlmClient().returns(f'{{"modules":[{module("A", [0])},{module("B", [2])}]}}')
    planner, _ = pipeline(fake)

    result = await planner.plan(plan_request())

    assert [m.section_indices for m in result.modules] == [[0, 1], [2, 3]]


async def test_card_estimates_are_clamped_and_missing_titles_filled():
    fake = FakeLlmClient().returns(f'{{"modules":[{module("  ", [0, 1, 2, 3], cards=500)}]}}')
    planner, _ = pipeline(fake)

    (planned,) = (await planner.plan(plan_request())).modules

    assert planned.estimated_cards == 30
    assert planned.title == "Module 1"


async def test_modules_over_the_limit_are_merged_into_earlier_ones():
    four = ",".join(module(t, [i]) for i, t in enumerate("ABCD"))
    planner, _ = pipeline(FakeLlmClient().returns(f'{{"modules":[{four}]}}'), PipelineOptions(max_modules=2))

    result = await planner.plan(plan_request())

    assert [m.section_indices for m in result.modules] == [[0], [1, 2, 3]]


async def test_a_plan_with_no_usable_modules_is_an_invalid_output_error():
    planner, _ = pipeline(FakeLlmClient().returns(f'{{"modules":[{module("X", [42])}]}}'))

    with pytest.raises(LlmError) as caught:
        await planner.plan(plan_request())

    assert caught.value.code == LlmErrorCode.INVALID_OUTPUT


async def test_json_wrapped_in_code_fences_still_parses():
    planner, _ = pipeline(FakeLlmClient().returns(f'```json\n{{"modules":[{module("All", [0, 1, 2, 3])}]}}\n```'))

    assert len((await planner.plan(plan_request())).modules) == 1


async def test_unknown_provider_is_rejected():
    planner, _ = pipeline(FakeLlmClient())

    with pytest.raises(LlmError) as caught:
        await planner.plan(plan_request(provider="gpt-9"))

    assert caught.value.code == LlmErrorCode.UNKNOWN_PROVIDER


async def test_an_empty_outline_is_a_bad_request():
    planner, _ = pipeline(FakeLlmClient())

    with pytest.raises(PipelineRequestError):
        await planner.plan(plan_request().model_copy(update={"sections": []}))
