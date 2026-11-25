"""Step 2 of 2: ask the model for flashcards for the modules the user kept, then gate every card.

Nothing is saved here; the page shows the staged cards for review. Anchor: AnkiX `BulkCardGenerationService`.
"""

from doctodeck.llm.contracts import LlmError, LlmErrorCode
from doctodeck.llm.registry import LlmRegistry
from doctodeck.pipeline import prompts, quality_gate
from doctodeck.pipeline.models import (
    CardsOutput,
    GeneratedCard,
    GenerateRequest,
    GenerateResult,
    PipelineOptions,
    PipelineRequestError,
    PlannedModule,
)
from doctodeck.pipeline.prompts import CardsPromptInput, CardsPromptModule


class CardGenerator:
    def __init__(self, llms: LlmRegistry, options: PipelineOptions) -> None:
        self._llms = llms
        self._options = options

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        if not request.modules:
            raise PipelineRequestError("Pick at least one module to generate cards for.")
        if not request.clean_text.strip():
            raise PipelineRequestError("The document text is empty. Parse a document first.")

        llm = self._llms.get(request.provider)
        options = self._options
        target = max(3, min(request.target_card_count or options.default_target_cards, options.max_target_cards))

        # Send at most max_source_chars, and say so when the text was cut (AnkiX cuts silently).
        # Python strings index by code point, so this slice can't split an emoji in half
        # (C# strings are UTF-16 and needed a surrogate-pair check here).
        text = request.clean_text.strip()
        truncated = len(text) > options.max_source_chars
        sent = text[: options.max_source_chars]

        sections = request.sections

        def picked(module: PlannedModule) -> list[int]:
            return [i for i in module.section_indices if 0 <= i < len(sections)]

        prompt_input = CardsPromptInput(
            source_title=request.source_title,
            target_card_count=target,
            modules=[
                CardsPromptModule(
                    module_index=m.module_index,
                    title=m.title,
                    section_titles=[sections[i].title for i in picked(m)],
                    section_snippets=[sections[i].snippet for i in picked(m)],
                )
                for m in request.modules
            ],
            text=sent,
        )

        result = await llm.complete(prompts.cards_request(prompt_input))

        repairs: list[str] = []
        cards = _build_cards(result.data, request.modules, repairs)
        if not cards:
            raise LlmError(LlmErrorCode.INVALID_OUTPUT, "The model returned no usable cards.")

        return GenerateResult(
            cards=cards,
            truncated=truncated,
            sent_chars=len(sent),
            total_chars=len(text),
            repairs=repairs,
            usage=result.usage,
        )


def _build_cards(output: CardsOutput, modules: list[PlannedModule], repairs: list[str]) -> list[GeneratedCard]:
    by_index = {m.module_index: m for m in modules}
    cards: list[GeneratedCard] = []

    for n, raw in enumerate(output.cards, start=1):
        front, back = raw.front.strip(), raw.back.strip()
        if not front and not back:
            repairs.append(f"Card {n} was empty and was skipped.")
            continue

        # The model says which module a card belongs to. If that module doesn't exist (or wasn't picked),
        # spread such cards across the picked modules in turn, and say so.
        module_index = raw.module_index
        if module_index not in by_index:
            module_index = modules[len(cards) % len(modules)].module_index
            repairs.append(f"Card {n} had moduleIndex {raw.module_index}; assigned it to module {module_index}.")

        score, flags = quality_gate.evaluate(front, back, (c.front for c in cards))
        cards.append(
            GeneratedCard(
                front=front,
                back=back,
                module_index=module_index,
                module_title=by_index[module_index].title,
                topic_tag=raw.topic_tag.strip().lower(),
                rationale=raw.rationale.strip(),
                quality_score=score,
                quality_flags=flags,
            )
        )

    return cards
