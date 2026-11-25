"""Every prompt lives here, once, whatever the provider (AnkiX does the same in StructuredLlmClientBase).

The system prompt sets the role and the rules; the user prompt carries the task and the data. The data is
sent as JSON after a marker line, so the model sees clean structure and the fake can read back exactly
what a real model would.
"""

from pydantic import BaseModel

from doctodeck.llm.contracts import LlmRequest
from doctodeck.models import ApiModel
from doctodeck.pipeline.models import CardsOutput, PlanOutput

INPUT_MARKER = "INPUT (JSON):"


class PlanPromptSection(ApiModel):
    index: int
    title: str
    level: int
    words: int
    snippet: str


class PlanPromptInput(ApiModel):
    source_title: str
    word_count: int
    page_count: int
    target_modules: int
    sections: list[PlanPromptSection]


class CardsPromptModule(ApiModel):
    module_index: int
    title: str
    section_titles: list[str]
    section_snippets: list[str]


class CardsPromptInput(ApiModel):
    source_title: str
    target_card_count: int
    modules: list[CardsPromptModule]
    text: str


PLAN_SYSTEM = (
    "You are a curriculum designer for spaced-repetition flashcards. You split learning material into "
    "a few coherent modules, each focused on one learning goal, so a learner is never overloaded. "
    "Answer only with JSON that matches the required schema."
)

CARDS_SYSTEM = """\
You write high-retention flashcards following Piotr Woźniak's 20 Rules of Formulating Knowledge.
1. Minimum information principle: each card tests exactly ONE fact or idea.
2. Front: a precise question of at most 150 characters with a clear retrieval cue. Never put the answer in it.
3. Back: a direct, definitive answer of at most 250 characters.
4. No duplicate, trivial or vague cards.
5. Use only facts from the provided text. Do not invent anything.
Answer only with JSON that matches the required schema."""


def plan_request(data: PlanPromptInput) -> LlmRequest[PlanOutput]:
    prompt = f"""\
Group the outline sections below into about {data.target_modules} thematic modules.
Rules:
1. Every section index belongs to exactly one module. Keep sections in reading order.
2. Group sections that are next to each other and about the same idea.
3. estimatedCards: 3 to 30, based on how much there is to learn in the module.
4. Use only section indices that appear in the input.

{INPUT_MARKER}
{data.model_dump_json(by_alias=True)}"""
    return LlmRequest("plan", PLAN_SYSTEM, prompt, PlanOutput)


def cards_request(data: CardsPromptInput) -> LlmRequest[CardsOutput]:
    prompt = f"""\
Write about {data.target_card_count} flashcards in total from the text below, spread across the modules.
Rules:
1. Set each card's moduleIndex to the module (from the input) whose topic it tests.
2. Prefer the facts that matter most for understanding each module.

{INPUT_MARKER}
{data.model_dump_json(by_alias=True)}"""
    return LlmRequest("cards", CARDS_SYSTEM, prompt, CardsOutput)


def read_input[T: BaseModel](user_prompt: str, input_type: type[T]) -> T:
    """Read the input JSON back out of a prompt. Used by the fake."""
    _, marker, data = user_prompt.partition(INPUT_MARKER)
    if not marker:
        raise ValueError("The prompt has no input block.")
    return input_type.model_validate_json(data)
