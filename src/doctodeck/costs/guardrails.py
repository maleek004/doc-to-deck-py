"""Guardrails (4.3): checks that run before a real provider is paid for a call.

1. An input token budget per call. The generator already cuts the source text to `max_source_chars` and says so
   (`truncated`, since 2.1: AnkiX cuts silently). The budget is the backstop for every other path into a prompt,
   such as a plan request with a huge outline, or a future step that forgets to cut: a call over budget is
   refused before it is sent, never sent and billed.
2. A global daily spending cap, read from the usage ledger. Once real providers have cost `daily_cap_usd` today
   (UTC), real calls are refused until midnight UTC. The fake isn't wrapped, so the app keeps working in fake mode.

Where it sits: ledger → cache → guard → provider. Inside the cache, so an answer already in the cache is still
served when the cap is reached (it costs nothing). Inside the ledger, so a refusal is recorded like any failed call.

The cap is soft: two calls that both check while the spend is just under the cap both go ahead, so a day can end
over the cap by what the calls in flight at that moment cost. Making it hard would mean reserving each call's
worst-case cost before it starts (see the flashcards); for one local user that overshoot is a few cents.
"""

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from doctodeck.costs.ledger import Ledger
from doctodeck.costs.prices import PRICES, PriceTable
from doctodeck.llm.contracts import LlmClient, LlmError, LlmErrorCode, LlmRequest, LlmResult

DEFAULT_MAX_INPUT_TOKENS = 8_000
DEFAULT_DAILY_CAP_USD = 1.00
CHARS_PER_TOKEN = 3  # errs high: English prose is nearer 4 characters per token, code and other scripts fewer


def estimate_tokens(text: str) -> int:
    """A rough count, with no network call. Good enough for a refusal threshold. (Both providers have a
    count-tokens endpoint; it's exact, but it's one more request before every call.)"""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Guardrails:
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    daily_cap_usd: float = DEFAULT_DAILY_CAP_USD

    @classmethod
    def from_env(cls) -> "Guardrails":
        """MAX_INPUT_TOKENS and DAILY_SPEND_CAP_USD, or the defaults (8,000 tokens, $1.00 a day)."""
        tokens = os.environ.get("MAX_INPUT_TOKENS", "").strip()
        cap = os.environ.get("DAILY_SPEND_CAP_USD", "").strip()
        return cls(
            max_input_tokens=int(tokens) if tokens else DEFAULT_MAX_INPUT_TOKENS,
            daily_cap_usd=float(cap) if cap else DEFAULT_DAILY_CAP_USD,
        )


class GuardedLlmClient:
    """Refuses a real call that is over the input budget, that would spend past today's cap, or whose model has no
    price (the cap can't be enforced on spending it can't measure: fail closed). Otherwise passes it through."""

    def __init__(
        self,
        inner: LlmClient,
        ledger: Ledger,
        guardrails: Guardrails,
        prices: PriceTable = PRICES,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._guardrails = guardrails
        self._prices = prices
        self._now = now

    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def is_real(self) -> bool:
        return self._inner.is_real

    @property
    def settings(self) -> Mapping[str, Any]:
        """Passed through: the cache outside this wrapper keys on the provider's generation settings."""
        return getattr(self._inner, "settings", None) or {}

    @property
    def inner(self) -> LlmClient:
        return self._inner

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()

    async def complete[T: BaseModel](self, request: LlmRequest[T]) -> LlmResult[T]:
        budget = self._guardrails.max_input_tokens
        estimated = estimate_tokens(request.system_prompt) + estimate_tokens(request.user_prompt)
        if estimated > budget:
            raise LlmError(
                LlmErrorCode.INPUT_TOO_LARGE,
                f"This {request.purpose} request is about {estimated:,} input tokens; the limit is {budget:,} "
                "(MAX_INPUT_TOKENS). Nothing was sent. Use a shorter document or fewer sections.",
            )

        today = self._now().astimezone(UTC).date()
        if self._prices.price_on(self.model, today) is None:
            raise LlmError(
                LlmErrorCode.UNPRICED_MODEL,
                f"{self.provider}/{self.model} has no price in the price table, so the daily spending cap can't be "
                "enforced on it. Add its price to costs/prices.py, or use a priced model.",
            )

        cap = self._guardrails.daily_cap_usd
        spent = await self._ledger.spent(today.isoformat())
        if spent >= cap:
            raise LlmError(
                LlmErrorCode.SPEND_CAP_REACHED,
                f"Real providers have cost ${spent:.4f} today (UTC), at or over the ${cap:.2f} daily cap "
                "(DAILY_SPEND_CAP_USD). They're off until 00:00 UTC; the fake provider still works.",
            )

        return await self._inner.complete(request)
