"""Data shapes for the plan and generate steps: what the page sends, what the model answers, what the API returns."""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field

from doctodeck.llm.contracts import LlmUsage
from doctodeck.models import ApiModel, DocumentSection

INVALID_REQUEST = "INVALID_REQUEST"


@dataclass(frozen=True)
class PipelineOptions:
    """Limits for the LLM steps. Defaults mirror AnkiX (12,000 characters of source text per call)."""

    max_modules: int = 5
    max_source_chars: int = 12_000
    default_target_cards: int = 10
    max_target_cards: int = 30


class PipelineRequestError(Exception):
    """The page sent something the pipeline can't work with. Becomes a 400 INVALID_REQUEST."""


# ---------- What the model answers ----------
# These classes ARE the output schema: `model_json_schema()` produces the JSON Schema sent to the model,
# and the same class validates its answer. Descriptions go to the model as instructions. Limits such as
# "3 to 30" are written in descriptions, not as `Field(ge=3, le=30)`: a validator would reject the whole
# answer over one bad number, while the planner can clamp it and record the fix.


class PlanOutputModule(ApiModel):
    title: str = Field(description="Short module title, e.g. 'Async fundamentals'.")
    description: str = Field(description="One or two sentences on what this module covers.")
    estimated_cards: int = Field(description="How many atomic flashcards this module deserves (3 to 30).")
    section_indices: list[int] = Field(description="0-based indices of the outline sections in this module.")


class PlanOutput(ApiModel):
    modules: list[PlanOutputModule] = Field(
        description="1 to 5 thematic modules that together cover the document, in reading order."
    )


class CardOutput(ApiModel):
    front: str = Field(description="The question. At most 150 characters. Must not contain the answer.")
    back: str = Field(description="The answer. At most 250 characters.")
    module_index: int = Field(description="The moduleIndex this card belongs to, from the input.")
    topic_tag: str = Field(description="One short lowercase topic tag.")
    rationale: str = Field(description="One sentence on why this fact is worth remembering.")


class CardsOutput(ApiModel):
    cards: list[CardOutput]


# ---------- What the page sends and the API returns ----------


class PlanRequest(ApiModel):
    """POST /api/plan: the outline from /api/parse plus which provider to use."""

    provider: str
    source_title: str
    sections: list[DocumentSection]
    word_count: int
    page_count: int


class PlannedModule(ApiModel):
    module_index: int  # 1-based, assigned by the planner after repair
    title: str
    description: str
    estimated_cards: int
    section_indices: list[int]


class PlanResult(ApiModel):
    """`repairs` lists every fix applied to the model's answer, so nothing is hidden."""

    modules: list[PlannedModule]
    repairs: list[str]
    usage: LlmUsage


class GenerateRequest(ApiModel):
    """POST /api/generate: the outline, the clean text, and the modules the user kept."""

    provider: str
    source_title: str
    sections: list[DocumentSection]
    clean_text: str
    modules: list[PlannedModule]
    target_card_count: int | None = None


class QualityScore(StrEnum):
    EXCELLENT = "Excellent"
    GOOD = "Good"
    NEEDS_REVIEW = "Needs Review"


class QualityFlag(StrEnum):
    EMPTY_FRONT = "EMPTY_FRONT"
    EMPTY_BACK = "EMPTY_BACK"
    BACK_TOO_LONG = "BACK_TOO_LONG"
    FRONT_TOO_LONG = "FRONT_TOO_LONG"
    AMBIGUOUS_CUE = "AMBIGUOUS_CUE"
    ANSWER_LEAK = "ANSWER_LEAK"
    POSSIBLE_DUPLICATE = "POSSIBLE_DUPLICATE"


class GeneratedCard(ApiModel):
    front: str
    back: str
    module_index: int
    module_title: str
    topic_tag: str
    rationale: str
    quality_score: QualityScore
    quality_flags: list[QualityFlag]


class GenerateResult(ApiModel):
    """`truncated` is true when the source text was longer than what the model was sent.
    (AnkiX drops the rest silently; here the page is told.)"""

    cards: list[GeneratedCard]
    truncated: bool
    sent_chars: int
    total_chars: int
    repairs: list[str]
    usage: LlmUsage
