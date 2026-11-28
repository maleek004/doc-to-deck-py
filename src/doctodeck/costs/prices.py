"""List prices of the models doc-to-deck can call, in US dollars per million tokens.

Two things make this more than a dict of numbers:
- Prices change over time. Gemini 3.8 Flash has an introductory price until the end of 2026. Each entry has the
  date it takes effect, and a call is priced on the day it ran. The ledger stores that cost and never recomputes
  it: last month's spend must not change when this table does.
- Some prices depend on the prompt's size. Claude Haiku 5.5 costs more per token once a prompt is over 100K
  tokens, so an entry can have a long-context tier.

Simplification: prompt-cache reads and writes are billed at other rates, but usage here counts every input token
the same, so costs for cached prompts are slightly overstated. That's the safe direction for a budget.

Sources, read 2026-11-28: Anthropic's model reference (Claude); Google's Gemini Developer API pricing page and
"What's new in Gemini 3.8" (Gemini, standard paid tier). Check them before trusting a bill to these numbers.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TokenPrice:
    """US dollars per million tokens, for input (prompt) and output (completion, thinking included)."""

    input_per_million: float
    output_per_million: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.input_per_million + completion_tokens * self.output_per_million) / 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    model: str
    effective_from: date
    base: TokenPrice
    # Above this many prompt tokens, the whole call is billed at the long-context price.
    long_context_above: int | None = None
    long_context: TokenPrice | None = None

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        long = self.long_context is not None and self.long_context_above is not None
        price = self.long_context if long and prompt_tokens > self.long_context_above else self.base
        return price.cost(prompt_tokens, completion_tokens)


class PriceTable:
    def __init__(self, entries: Iterable[ModelPrice]) -> None:
        self._by_model: dict[str, list[ModelPrice]] = {}
        for entry in entries:
            self._by_model.setdefault(entry.model, []).append(entry)
        for history in self._by_model.values():
            history.sort(key=lambda entry: entry.effective_from)

    def price_on(self, model: str, day: date) -> ModelPrice | None:
        """The entry in effect on `day`: the latest one that started on or before it. None if the model is unknown
        or wasn't priced yet: an unknown price stays unknown, never 0."""
        in_effect = [entry for entry in self._by_model.get(model, []) if entry.effective_from <= day]
        return in_effect[-1] if in_effect else None

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int, day: date) -> float | None:
        price = self.price_on(model, day)
        return None if price is None else price.cost(prompt_tokens, completion_tokens)

    def prices_on(self, day: date) -> dict[str, ModelPrice]:
        """Every model's price on `day`, e.g. for the eval report's cost column."""
        return {model: price for model in self._by_model if (price := self.price_on(model, day)) is not None}


_LONG_AGO = date(2025, 1, 1)  # "in effect before anything this app ran"

PRICES = PriceTable(
    [
        # Claude (Anthropic API). Haiku 5.5 costs 5x per token once the prompt is over 100K tokens.
        ModelPrice("claude-haiku-5-5", _LONG_AGO, TokenPrice(0.10, 0.50), 100_000, TokenPrice(0.50, 2.50)),
        ModelPrice("claude-sonnet-5-5", _LONG_AGO, TokenPrice(2.00, 10.00)),
        ModelPrice("claude-opus-5-5", _LONG_AGO, TokenPrice(4.00, 20.00)),
        ModelPrice("claude-fable-5-2", _LONG_AGO, TokenPrice(8.00, 40.00)),
        # Gemini (Developer API, standard paid tier; the free tier costs nothing but is capped by quota).
        ModelPrice("gemini-3.8-flash", _LONG_AGO, TokenPrice(0.75, 3.75)),  # introductory price
        ModelPrice("gemini-3.8-flash", date(2027, 1, 1), TokenPrice(1.50, 7.50)),
        ModelPrice("gemini-2.5-flash", _LONG_AGO, TokenPrice(0.30, 2.50)),
    ]
)
