"""Shared inputs for the Layer 2 tests: a four-section outline and a pipeline wired to one fake."""

from doctodeck.llm.fake import FakeLlmClient
from doctodeck.llm.registry import LlmRegistry
from doctodeck.models import DocumentSection
from doctodeck.pipeline.generator import CardGenerator
from doctodeck.pipeline.models import GenerateRequest, PipelineOptions, PlannedModule, PlanRequest
from doctodeck.pipeline.planner import DeckPlanner

SECTIONS = [
    DocumentSection(title="Async basics", level=1, word_count=40,
                    snippet="An async function returns a coroutine. The caller can await it."),
    DocumentSection(title="Await", level=2, word_count=30,
                    snippet="Await hands control back to the event loop until the result is ready."),
    DocumentSection(title="Cancellation", level=1, word_count=35,
                    snippet="Cancelling a task raises CancelledError inside it."),
    DocumentSection(title="Timeouts", level=2, word_count=25,
                    snippet="asyncio.timeout turns a deadline into a cancellation."),
]  # fmt: skip

CLEAN_TEXT = "\n".join(f"{s.title}\n{s.snippet}" for s in SECTIONS)

TWO_MODULES = [
    PlannedModule(module_index=1, title="Async", description="Coroutines and await", estimated_cards=5,
                  section_indices=[0, 1]),
    PlannedModule(module_index=2, title="Cancellation", description="Cancelling and timeouts", estimated_cards=5,
                  section_indices=[2, 3]),
]  # fmt: skip


def plan_request(provider: str = "fake", word_count: int = 130) -> PlanRequest:
    return PlanRequest(provider=provider, source_title="async.md", sections=SECTIONS, word_count=word_count,
                       page_count=1)  # fmt: skip


def generate_request(
    modules: list[PlannedModule], provider: str = "fake", text: str | None = None, cards: int | None = None
) -> GenerateRequest:
    return GenerateRequest(
        provider=provider,
        source_title="async.md",
        sections=SECTIONS,
        clean_text=CLEAN_TEXT if text is None else text,
        modules=modules,
        target_card_count=cards,
    )


def pipeline(fake: FakeLlmClient, options: PipelineOptions | None = None) -> tuple[DeckPlanner, CardGenerator]:
    registry = LlmRegistry([fake])
    options = options or PipelineOptions()
    return DeckPlanner(registry, options), CardGenerator(registry, options)
