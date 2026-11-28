"""Step 1 of 2: ask the model to group the outline into modules, then check and repair its answer.

The model only ever sees the outline (titles, word counts, snippets), never the full text: that keeps
this call small, cheap and fast. Anchor: AnkiX `DeckDecompositionService`.
"""

import math
from dataclasses import dataclass

from doctodeck.llm.contracts import LlmError, LlmErrorCode
from doctodeck.llm.registry import LlmRegistry
from doctodeck.pipeline import prompts
from doctodeck.pipeline.models import (
    PipelineOptions,
    PipelineRequestError,
    PlannedModule,
    PlanOutput,
    PlanRequest,
    PlanResult,
)
from doctodeck.pipeline.prompts import PlanPromptInput, PlanPromptSection

WORDS_PER_MODULE = 2500


@dataclass
class _Draft:
    title: str
    description: str
    estimated_cards: int
    indices: list[int]


class DeckPlanner:
    def __init__(
        self, llms: LlmRegistry, options: PipelineOptions, prompt_set: prompts.PromptSet = prompts.DEFAULT_PROMPTS
    ) -> None:
        # Composition: the planner is *given* the registry. AnkiX's services inherit from a base client instead.
        self._llms = llms
        self._options = options
        self._prompts = prompt_set  # which prompt version to send (the eval runner compares versions)

    async def plan(self, request: PlanRequest) -> PlanResult:
        if not request.sections:
            raise PipelineRequestError("The outline has no sections. Parse a document first.")

        llm = self._llms.get(request.provider)

        # About one module per 2,500 words, between 1 and max_modules, and never more modules than sections.
        limit = min(self._options.max_modules, len(request.sections))
        target = max(1, min(math.ceil(request.word_count / WORDS_PER_MODULE), limit))

        prompt_input = PlanPromptInput(
            source_title=request.source_title,
            word_count=request.word_count,
            page_count=request.page_count,
            target_modules=target,
            sections=[
                PlanPromptSection(index=i, title=s.title, level=s.level, words=s.word_count, snippet=s.snippet)
                for i, s in enumerate(request.sections)
            ],
        )

        result = await llm.complete(self._prompts.plan(prompt_input))

        repairs: list[str] = []
        modules = self.repair(result.data, len(request.sections), repairs)
        return PlanResult(modules=modules, repairs=repairs, usage=result.usage)

    def repair(self, output: PlanOutput, section_count: int, repairs: list[str]) -> list[PlannedModule]:
        """Never trust the model's indices or numbers: keep only real, unclaimed section indices; give every
        section a home; clamp the card estimates; fill in missing titles. Every fix is recorded."""
        claimed: set[int] = set()
        kept: list[_Draft] = []

        for n, raw in enumerate(output.modules, start=1):
            indices: list[int] = []
            for index in raw.section_indices:
                if not 0 <= index < section_count:
                    repairs.append(f"Module {n}: dropped section index {index} (only {section_count} sections exist).")
                elif index in claimed:
                    repairs.append(f"Module {n}: section {index} was already in an earlier module; kept it there.")
                else:
                    claimed.add(index)
                    indices.append(index)

            if not indices:
                repairs.append(f"Module {n} ('{raw.title}') had no usable sections and was removed.")
                continue
            if len(kept) == self._options.max_modules:
                repairs.append(
                    f"Module {n} is over the {self._options.max_modules}-module limit; "
                    "its sections were merged into earlier modules."
                )
                claimed.difference_update(indices)
                continue

            cards = max(3, min(raw.estimated_cards, 30))
            if cards != raw.estimated_cards:
                repairs.append(f"Module {n}: estimatedCards {raw.estimated_cards} → {cards}.")

            title = raw.title.strip() or f"Module {len(kept) + 1}"
            kept.append(_Draft(title, raw.description.strip(), cards, indices))

        if not kept:
            raise LlmError(LlmErrorCode.INVALID_OUTPUT, "The model's plan contained no usable modules.")

        # Sections the model forgot join the module just before them in reading order, else the first module.
        for missing in (i for i in range(section_count) if i not in claimed):
            home = next((m for m in reversed(kept) if min(m.indices) < missing), kept[0])
            home.indices.append(missing)
            repairs.append(f"Section {missing} wasn't in any module; added it to '{home.title}'.")

        return [
            PlannedModule(
                module_index=i,
                title=m.title,
                description=m.description,
                estimated_cards=m.estimated_cards,
                section_indices=sorted(m.indices),
            )
            for i, m in enumerate(kept, start=1)
        ]
