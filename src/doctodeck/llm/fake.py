"""An LLM stand-in with two modes.

- Demo (default): reads the structured input back out of the prompt and answers with simple,
  deterministic JSON. It powers "try it without a key" on the page and the end-to-end tests.
- Scripted: a test queues exact raw answers or failures with `returns()` / `fails()`, to check how the
  pipeline copes with wrong or broken model output, or with a provider that is down.

Its answers go through the same `parse_output` as a real provider's, so fenced JSON, invalid JSON and
wrong shapes behave exactly as they would live.
"""

import re
import time
from collections import deque
from collections.abc import Callable

from pydantic import BaseModel

from doctodeck.llm.contracts import LlmError, LlmErrorCode, LlmRequest, LlmResult, LlmUsage, parse_output
from doctodeck.pipeline.models import CardOutput, CardsOutput, PlanOutput, PlanOutputModule
from doctodeck.pipeline.prompts import CardsPromptInput, PlanPromptInput, read_input

_SENTENCE_END = re.compile(r"[.!?]")


class FakeLlmClient:
    model = "fake-demo"
    is_real = False

    def __init__(self, provider: str = "fake") -> None:
        self.provider = provider
        # A plain deque is enough: asyncio runs on one thread, so no other code can touch it between
        # two lines here unless there is an `await` in between. (The C# fake needed a ConcurrentQueue.)
        self._script: deque[Callable[[], str]] = deque()
        self.requests: list[LlmRequest] = []  # everything received, so tests can inspect the prompts

    def returns(self, raw: str) -> "FakeLlmClient":
        """Queue one raw answer (the model's text, before parsing). Returns self, so calls chain."""
        self._script.append(lambda: raw)
        return self

    def fails(self, code: LlmErrorCode, message: str = "Injected failure.") -> "FakeLlmClient":
        """Queue one failure, e.g. RATE_LIMITED for "still 429 after retries" or TIMEOUT."""

        def fail() -> str:
            raise LlmError(code, message)

        self._script.append(fail)
        return self

    async def complete[T: BaseModel](self, request: LlmRequest[T]) -> LlmResult[T]:
        started = time.perf_counter()
        self.requests.append(request)

        raw = self._script.popleft()() if self._script else _demo(request)
        data = parse_output(raw, request.output_type, self.provider)

        usage = LlmUsage(
            provider=self.provider,
            model=self.model,
            # About 4 characters per token, so the numbers on the page look realistic.
            prompt_tokens=(len(request.system_prompt) + len(request.user_prompt)) // 4,
            completion_tokens=len(raw) // 4,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        return LlmResult(data, usage)


def _demo(request: LlmRequest) -> str:
    if request.purpose == "plan":
        answer: BaseModel = _demo_plan(read_input(request.user_prompt, PlanPromptInput))
    elif request.purpose == "cards":
        answer = _demo_cards(read_input(request.user_prompt, CardsPromptInput))
    else:
        raise LlmError(LlmErrorCode.INVALID_OUTPUT, f"The fake has no demo answer for '{request.purpose}'.")
    return answer.model_dump_json(by_alias=True)  # camelCase, as a real model following the schema would send


def _demo_plan(data: PlanPromptInput) -> PlanOutput:
    """Split the sections into exactly `target_modules` runs of near-equal size, in reading order.
    Section i goes to group i * count // n, e.g. 4 sections into 3 groups: 0, 0, 1, 2."""
    n = len(data.sections)
    count = max(1, min(data.target_modules, n))
    groups: dict[int, list] = {}
    for i, section in enumerate(data.sections):
        groups.setdefault(i * count // n, []).append(section)

    return PlanOutput(
        modules=[
            PlanOutputModule(
                title=group[0].title,
                description="Covers: " + ", ".join(s.title for s in group),
                estimated_cards=max(3, min(sum(s.words for s in group) // 40, 10)),
                section_indices=[s.index for s in group],
            )
            for group in groups.values()
        ]
    )


def _demo_cards(data: CardsPromptInput) -> CardsOutput:
    """One "What is the key idea of ...?" card per section, answered with that section's first sentence."""
    cards = [
        CardOutput(
            front=f'What is the key idea of "{title}"?',
            back=_first_sentence(snippet),
            module_index=module.module_index,
            topic_tag=(title.lower().split() or ["general"])[0],
            rationale="Demo card built from the section's first sentence.",
        )
        for module in data.modules
        for title, snippet in zip(module.section_titles, module.section_snippets, strict=True)
        if _first_sentence(snippet)
    ]
    return CardsOutput(cards=cards[: data.target_card_count])


def _first_sentence(text: str) -> str:
    end = _SENTENCE_END.search(text)
    sentence = (text[: end.end()] if end else text).strip()
    return sentence if len(sentence) <= 250 else sentence[:247].rstrip() + "..."
