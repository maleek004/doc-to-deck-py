"""The usage ledger (4.1): one row per LLM call, whatever happened, with what it cost.

Every call is recorded: successes with their tokens and cost, cache hits (cost 0: nothing was spent), failures
(no tokens, cost 0, the error code as the outcome) and fake calls (cost 0, not real). The cost is computed once,
at the price in effect that day, and stored: the ledger is history, and history doesn't change when prices do.

Calls are grouped into decks. A deck is one document's plan and generate calls, which arrive as two separate HTTP
requests; the page sends the same deck id with both. The id travels to the ledger in a ContextVar (`deck_scope`),
not as a parameter through the planner, the generator and every client: it's request context, like a trace id.

`LedgerLlmClient` wraps a client the same way the cache does (decorator pattern), outermost, so it sees what the
user experienced: a cache hit is a call that cost nothing, not a call that never happened.

Storage: stdlib sqlite3 in a worker thread, as the cache. SQLite's `SUM` over a column of the day's rows is also
what the daily spending cap (4.3) will read.
"""

import asyncio
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from doctodeck.costs.prices import PRICES, PriceTable
from doctodeck.llm.contracts import LlmClient, LlmError, LlmRequest, LlmResult
from doctodeck.models import ApiModel

DEFAULT_LEDGER = Path("data/ledger.sqlite3")

# The deck the current task is working on. Each asyncio task runs in a copy of the context it was created in, so
# two requests handled at once each see their own value: no locks, no leaking between requests.
current_deck: ContextVar[str | None] = ContextVar("current_deck", default=None)


@contextmanager
def deck_scope(deck_id: str | None) -> Iterator[None]:
    """Calls made inside are recorded against `deck_id`. Resets on the way out, even after an exception."""
    token = current_deck.set(deck_id)
    try:
        yield
    finally:
        current_deck.reset(token)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    at                TEXT    NOT NULL,  -- UTC, ISO 8601: sorts as text
    day               TEXT    NOT NULL,  -- UTC date, for "spent today"
    deck_id           TEXT,
    purpose           TEXT    NOT NULL,
    provider          TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    is_real           INTEGER NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cached            INTEGER NOT NULL,
    cost_usd          REAL,              -- NULL: no price known for this model on that day
    latency_ms        INTEGER NOT NULL,
    outcome           TEXT    NOT NULL   -- "ok" or an LlmErrorCode
);
CREATE INDEX IF NOT EXISTS calls_day ON calls (day);
CREATE INDEX IF NOT EXISTS calls_deck ON calls (deck_id);
"""


class LedgerEntry(ApiModel):
    id: int
    at: str
    deck_id: str | None
    purpose: str
    provider: str
    model: str
    is_real: bool
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    cost_usd: float | None
    latency_ms: int
    outcome: str


class DeckCost(ApiModel):
    deck_id: str
    started_at: str
    calls: int
    failed_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None  # None if any call in the deck had no known price
    latency_ms: int


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)

    @classmethod
    def from_env(cls) -> "Ledger":
        """LEDGER_DB, or data/ledger.sqlite3. Always on: an app that spends money must know where it went."""
        return cls(os.environ.get("LEDGER_DB", "").strip() or DEFAULT_LEDGER)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row  # rows readable by column name
        return db

    async def record(self, entry: dict) -> int:
        columns = ", ".join(entry)
        marks = ", ".join("?" for _ in entry)

        def insert() -> int:
            with self._connect() as db:
                return db.execute(f"INSERT INTO calls ({columns}) VALUES ({marks})", tuple(entry.values())).lastrowid

        return await asyncio.to_thread(insert)

    async def recent(self, limit: int = 50) -> list[LedgerEntry]:
        def read() -> list[LedgerEntry]:
            with self._connect() as db:
                rows = db.execute(
                    "SELECT id, at, deck_id, purpose, provider, model, is_real, prompt_tokens, completion_tokens,"
                    " cached, cost_usd, latency_ms, outcome FROM calls ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [LedgerEntry.model_validate(dict(row)) for row in rows]

        return await asyncio.to_thread(read)

    async def decks(self, limit: int = 20) -> list[DeckCost]:
        """Cost per deck, newest first. One GROUP BY: SQLite does the adding up.
        `COUNT(cost_usd) < COUNT(*)` means some call had no price (COUNT of a column skips NULLs), so the deck's
        cost is unknown rather than an undercount."""

        def read() -> list[DeckCost]:
            with self._connect() as db:
                rows = db.execute(
                    """
                    SELECT deck_id, MIN(at) AS started_at, COUNT(*) AS calls,
                           SUM(outcome != 'ok') AS failed_calls,
                           SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens,
                           CASE WHEN COUNT(cost_usd) < COUNT(*) THEN NULL ELSE SUM(cost_usd) END AS cost_usd,
                           SUM(latency_ms) AS latency_ms
                    FROM calls WHERE deck_id IS NOT NULL
                    GROUP BY deck_id ORDER BY MIN(id) DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [DeckCost.model_validate(dict(row)) for row in rows]

        return await asyncio.to_thread(read)

    async def spent(self, day: str | None = None) -> float:
        """Known spend on real providers, for one UTC day (YYYY-MM-DD) or ever. Unpriced calls add nothing here:
        the daily cap (4.3) has to decide what to do about them separately."""

        def read() -> float:
            query, args = "SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE is_real = 1", ()
            if day is not None:
                query, args = query + " AND day = ?", (day,)
            with self._connect() as db:
                return float(db.execute(query, args).fetchone()[0])

        return await asyncio.to_thread(read)


class LedgerLlmClient:
    """Records every call made through `inner` in the ledger, then returns or re-raises exactly what it got."""

    def __init__(
        self,
        inner: LlmClient,
        ledger: Ledger,
        prices: PriceTable = PRICES,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._inner = inner
        self._ledger = ledger
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
    def inner(self) -> LlmClient:
        return self._inner

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()

    async def complete[T: BaseModel](self, request: LlmRequest[T]) -> LlmResult[T]:
        started = time.perf_counter()
        try:
            result = await self._inner.complete(request)
        except LlmError as ex:
            # A failed call is recorded too: how often a provider fails, and how long it made us wait, is the
            # ledger's business. No tokens came back, so nothing is charged.
            await self._record(request, self.model, 0, 0, False, round((time.perf_counter() - started) * 1000), ex.code)
            raise
        # asyncio.CancelledError (the client went away mid-call) isn't caught: nothing to record yet. Keeping the
        # record of a call that *did* finish even if the request is cancelled is the charge's problem (4.2: shield).
        usage = result.usage
        await self._record(
            request, usage.model, usage.prompt_tokens, usage.completion_tokens, usage.cached, usage.latency_ms, "ok"
        )
        return result

    async def _record(
        self, request: LlmRequest, model: str, prompt: int, completion: int, cached: bool, latency: int, outcome: str
    ) -> None:
        now = self._now().astimezone(UTC)
        if not self.is_real or cached or outcome != "ok":
            cost: float | None = 0.0  # the fake is free; a cache hit or a failure spent nothing
        else:
            cost = self._prices.cost(model, prompt, completion, now.date())
        await self._ledger.record(
            {
                "at": now.isoformat(timespec="milliseconds"),
                "day": now.date().isoformat(),
                "deck_id": current_deck.get(),
                "purpose": request.purpose,
                "provider": self.provider,
                "model": model,
                "is_real": int(self.is_real),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cached": int(cached),
                "cost_usd": cost,
                "latency_ms": latency,
                "outcome": str(outcome),
            }
        )
