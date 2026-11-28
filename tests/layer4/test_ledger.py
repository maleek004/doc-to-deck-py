"""The usage ledger and price table (4.1). No key, no network: `AsReal` makes the fake look like a priced model."""

import asyncio
from datetime import UTC, date, datetime

import pytest
from fastapi.testclient import TestClient

from doctodeck.app import create_app
from doctodeck.costs.ledger import Ledger, LedgerLlmClient, current_deck, deck_scope
from doctodeck.costs.prices import PRICES, ModelPrice, PriceTable, TokenPrice
from doctodeck.evals.golden import load_golden_set
from doctodeck.evals.runner import EvalSettings, start_eval
from doctodeck.llm.cache import CachingLlmClient, ResponseCache
from doctodeck.llm.contracts import LlmError, LlmErrorCode, LlmRequest, LlmResult
from doctodeck.llm.fake import FakeLlmClient
from doctodeck.llm.registry import LlmRegistry
from doctodeck.pipeline.models import PlanOutput
from doctodeck.pipeline.prompts import DEFAULT_PROMPTS

PLAN_JSON = '{"modules":[{"title":"A","description":"d","estimatedCards":4,"sectionIndices":[0]}]}'
REQUEST = LlmRequest("plan", "You plan decks.", "Plan this outline.", PlanOutput)
NOW = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)


class AsReal:
    """The fake, presented as a real, priced model whose usage is exactly `tokens`."""

    def __init__(self, model="claude-haiku-5-5", tokens=(1_000, 500), provider="claude", fail=None) -> None:
        self._inner = FakeLlmClient(provider=provider)
        self.provider, self.model, self.is_real = provider, model, True
        self.tokens, self.fail = tokens, fail

    async def complete[T](self, request: LlmRequest[T]) -> LlmResult[T]:
        if self.fail:
            raise LlmError(self.fail, "injected")
        self._inner.returns(PLAN_JSON) if request.purpose == "plan" else None
        result = await self._inner.complete(request)
        usage = result.usage.model_copy(
            update={"model": self.model, "prompt_tokens": self.tokens[0], "completion_tokens": self.tokens[1]}
        )
        return LlmResult(result.data, usage)


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "ledger.sqlite3")


def recorded(client, ledger: Ledger, now=NOW) -> LedgerLlmClient:
    return LedgerLlmClient(client, ledger, now=lambda: now)


# ---------- Prices ----------


def test_a_price_change_takes_effect_on_its_date():
    assert PRICES.price_on("gemini-3.8-flash", date(2026, 12, 31)).base == TokenPrice(0.75, 3.75)  # introductory
    assert PRICES.price_on("gemini-3.8-flash", date(2027, 1, 1)).base == TokenPrice(1.50, 7.50)


def test_haiku_is_billed_at_the_long_context_price_above_100k_prompt_tokens():
    haiku = PRICES.price_on("claude-haiku-5-5", date(2026, 11, 28))

    assert haiku.cost(100_000, 1_000) == pytest.approx((100_000 * 0.10 + 1_000 * 0.50) / 1e6)
    assert haiku.cost(100_001, 1_000) == pytest.approx((100_001 * 0.50 + 1_000 * 2.50) / 1e6)


def test_an_unknown_or_not_yet_priced_model_has_no_price_not_a_zero_one():
    table = PriceTable([ModelPrice("m", date(2027, 1, 1), TokenPrice(1, 1))])

    assert table.cost("unknown", 10, 10, date(2027, 6, 1)) is None
    assert table.cost("m", 10, 10, date(2026, 6, 1)) is None  # before its first price
    assert set(PRICES.prices_on(date(2026, 11, 28))) >= {"claude-haiku-5-5", "gemini-3.8-flash", "gemini-2.5-flash"}


# ---------- Recording calls ----------


async def test_a_real_call_is_recorded_with_its_cost(ledger):
    client = recorded(AsReal(tokens=(100_000, 10_000)), ledger)  # at Haiku's 100K threshold: still the base price

    await client.complete(REQUEST)

    [entry] = await ledger.recent()
    assert (entry.provider, entry.model, entry.purpose, entry.outcome) == ("claude", "claude-haiku-5-5", "plan", "ok")
    assert (entry.prompt_tokens, entry.completion_tokens) == (100_000, 10_000)
    assert entry.is_real and not entry.cached
    assert entry.cost_usd == pytest.approx(0.01 + 0.005)  # 100K in at $0.10 + 10K out at $0.50 per million
    assert entry.at.startswith("2026-11-28T12:00:00")
    assert await ledger.spent("2026-11-28") == pytest.approx(0.015)


async def test_the_cost_is_fixed_at_the_price_of_the_day_it_ran(ledger):
    tokens = (1_000_000, 0)
    await recorded(AsReal(model="gemini-3.8-flash", provider="gemini", tokens=tokens), ledger, NOW).complete(REQUEST)
    await recorded(AsReal(model="gemini-3.8-flash", provider="gemini", tokens=tokens), ledger,
                   datetime(2027, 1, 2, tzinfo=UTC)).complete(REQUEST)  # fmt: skip

    later, earlier = await ledger.recent()
    assert (earlier.cost_usd, later.cost_usd) == (pytest.approx(0.75), pytest.approx(1.50))


async def test_fake_calls_are_recorded_and_cost_nothing(ledger):
    await recorded(FakeLlmClient().returns(PLAN_JSON), ledger).complete(REQUEST)

    [entry] = await ledger.recent()
    assert (entry.provider, entry.is_real, entry.cost_usd) == ("fake", False, 0.0)
    assert await ledger.spent() == 0.0


async def test_a_failed_call_is_recorded_charges_nothing_and_still_raises(ledger):
    client = recorded(AsReal(fail=LlmErrorCode.RATE_LIMITED), ledger)

    with pytest.raises(LlmError):
        await client.complete(REQUEST)

    [entry] = await ledger.recent()
    assert entry.outcome == "PROVIDER_RATE_LIMITED"
    assert (entry.prompt_tokens, entry.completion_tokens, entry.cost_usd) == (0, 0, 0.0)


async def test_a_cache_hit_is_recorded_as_a_call_that_cost_nothing(ledger, tmp_path):
    client = recorded(CachingLlmClient(AsReal(), ResponseCache(tmp_path / "cache.sqlite3")), ledger)

    await client.complete(REQUEST)
    await client.complete(REQUEST)

    hit, miss = await ledger.recent()
    assert (miss.cached, hit.cached) == (False, True)
    assert miss.cost_usd > 0 and hit.cost_usd == 0.0
    assert hit.prompt_tokens == miss.prompt_tokens  # the tokens the answer was made with, not what we paid for


async def test_an_unpriced_real_model_is_recorded_as_unpriced(ledger):
    with deck_scope("d1"):
        await recorded(AsReal(model="mystery-model"), ledger).complete(REQUEST)

    [entry] = await ledger.recent()
    [deck] = await ledger.decks()
    assert entry.cost_usd is None
    assert deck.cost_usd is None  # one unknown call makes the deck's total unknown, not an undercount
    assert await ledger.spent() == 0.0


# ---------- Decks ----------


async def test_calls_in_a_deck_scope_are_grouped_and_totalled(ledger):
    client = recorded(AsReal(tokens=(50_000, 0)), ledger)

    with deck_scope("deck-a"):
        await client.complete(REQUEST)
        await client.complete(REQUEST)
    with deck_scope("deck-b"):
        await client.complete(REQUEST)
    await client.complete(REQUEST)  # outside any deck: in the ledger, in no deck

    decks = {deck.deck_id: deck for deck in await ledger.decks()}
    assert set(decks) == {"deck-a", "deck-b"}
    assert (decks["deck-a"].calls, decks["deck-a"].prompt_tokens) == (2, 100_000)
    assert decks["deck-a"].cost_usd == pytest.approx(0.01)  # 2 x 50K in at $0.10 per million
    assert len(await ledger.recent()) == 4


async def test_concurrent_requests_keep_their_own_deck(ledger):
    client = recorded(AsReal(), ledger)

    async def build_deck(deck_id: str) -> None:
        with deck_scope(deck_id):
            for _ in range(3):
                await client.complete(REQUEST)
                await asyncio.sleep(0)  # let the other tasks run in between

    await asyncio.gather(*(build_deck(f"deck-{i}") for i in range(5)))

    decks = await ledger.decks()
    assert sorted(deck.deck_id for deck in decks) == [f"deck-{i}" for i in range(5)]
    assert all(deck.calls == 3 for deck in decks)


def test_a_deck_scope_is_undone_even_after_an_exception():
    with pytest.raises(RuntimeError), deck_scope("inner"):
        raise RuntimeError

    assert current_deck.get() is None


async def test_each_eval_case_is_its_own_deck(ledger, tmp_path):
    llms = LlmRegistry([LedgerLlmClient(FakeLlmClient(), ledger)])
    cases = load_golden_set()[:2]

    run = await start_eval(cases, llms, EvalSettings("fake", DEFAULT_PROMPTS), tmp_path / "runs")

    decks = await ledger.decks()
    assert sorted(deck.deck_id for deck in decks) == [f"{run.path.stem}/{case.id}" for case in cases]
    assert all(deck.calls == 2 for deck in decks)  # plan + cards


# ---------- Through the app ----------


def test_the_page_flow_lands_in_the_ledger_under_its_deck_id():
    with TestClient(create_app()) as client:
        doc = client.post("/api/parse", files={"file": ("a.md", b"# A\n\nAsync waits without blocking.\n")}).json()
        sections = doc["document"]["sections"]
        common = {"provider": "fake", "deckId": "page-deck-1", "sourceTitle": "a.md", "sections": sections}
        plan = client.post("/api/plan", json=common | {"wordCount": 5, "pageCount": 1}).json()
        client.post(
            "/api/generate", json=common | {"cleanText": doc["document"]["cleanText"], "modules": plan["modules"]}
        )
        view = client.get("/api/ledger").json()

    assert [deck["deckId"] for deck in view["decks"]] == ["page-deck-1"]
    assert view["decks"][0]["calls"] == 2
    assert [entry["purpose"] for entry in view["entries"]] == ["cards", "plan"]  # newest first
    assert (view["spentTodayUsd"], view["spentTotalUsd"]) == (0.0, 0.0)  # the fake is free


def test_a_request_without_a_deck_id_is_still_recorded():
    with TestClient(create_app()) as client:
        doc = client.post("/api/parse", files={"file": ("a.md", b"# A\n\nText.\n")}).json()
        sections = doc["document"]["sections"]
        plan = {"provider": "fake", "sourceTitle": "a.md", "sections": sections, "wordCount": 1, "pageCount": 1}
        client.post("/api/plan", json=plan)
        view = client.get("/api/ledger?limit=1").json()

    assert len(view["entries"]) == 1 and view["entries"][0]["deckId"] is None
    assert view["decks"] == []
