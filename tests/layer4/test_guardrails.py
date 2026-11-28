"""Guardrails (4.3): the input token budget and the daily spending cap. No key, no network: every refusal happens
before anything is sent, and the "real" providers are the fake in disguise (`AsReal`)."""

import asyncio
import os
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from doctodeck.app import create_app
from doctodeck.costs.guardrails import GuardedLlmClient, Guardrails, estimate_tokens
from doctodeck.costs.ledger import Ledger, LedgerLlmClient
from doctodeck.evals.golden import load_golden_set
from doctodeck.evals.runner import EvalSettings, start_eval
from doctodeck.llm.cache import CachingLlmClient, ResponseCache
from doctodeck.llm.contracts import LlmError, LlmErrorCode, LlmRequest
from doctodeck.llm.fake import FakeLlmClient
from doctodeck.llm.registry import LlmRegistry
from doctodeck.pipeline.models import PlanOutput
from doctodeck.pipeline.prompts import DEFAULT_PROMPTS
from tests.layer2.data import plan_request
from tests.layer4.test_ledger import NOW, REQUEST, AsReal


async def aspend(ledger: Ledger, usd: float, day: str = "2026-11-28") -> None:
    """Put `usd` of real-provider spending in the ledger, as if calls had been made that day."""
    entry = {"at": f"{day}T08:00:00.000+00:00", "day": day, "deck_id": None, "purpose": "cards", "provider": "claude",
             "model": "claude-haiku-5-5", "is_real": 1, "prompt_tokens": 1, "completion_tokens": 1, "cached": 0,
             "cost_usd": usd, "latency_ms": 1, "outcome": "ok"}  # fmt: skip
    await ledger.record(entry)


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "ledger.sqlite3")


def guarded(inner, ledger: Ledger, max_input_tokens: int = 8_000, cap: float = 1.00) -> GuardedLlmClient:
    return GuardedLlmClient(inner, ledger, Guardrails(max_input_tokens, cap), now=lambda: NOW)


class Counting(AsReal):
    """AsReal that counts the calls that got through to it."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return await super().complete(request)


# ---------- Input token budget ----------


def test_tokens_are_estimated_on_the_high_side():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("a" * 3_001) == 1_001  # 3 characters per token, rounded up


async def test_a_call_over_the_input_budget_is_refused_before_it_is_sent(ledger):
    inner = Counting()
    big = LlmRequest("plan", "You plan decks.", "x" * 3_000, PlanOutput)  # about 1,005 tokens

    with pytest.raises(LlmError) as caught:
        await guarded(inner, ledger, max_input_tokens=1_000).complete(big)

    assert caught.value.code == LlmErrorCode.INPUT_TOO_LARGE
    assert "1,005" in caught.value.message and "1,000" in caught.value.message
    assert inner.calls == 0


async def test_a_call_within_the_budget_goes_through(ledger):
    inner = Counting()

    await guarded(inner, ledger).complete(REQUEST)

    assert inner.calls == 1


# ---------- Daily spending cap ----------


async def test_real_calls_are_refused_once_today_has_cost_the_cap(ledger):
    inner = Counting()
    await aspend(ledger, 0.60)
    await aspend(ledger, 0.40)  # $1.00 today: at the cap

    with pytest.raises(LlmError) as caught:
        await guarded(inner, ledger, cap=1.00).complete(REQUEST)

    assert caught.value.code == LlmErrorCode.SPEND_CAP_REACHED
    assert "$1.0000" in caught.value.message and "fake provider still works" in caught.value.message
    assert inner.calls == 0


async def test_just_under_the_cap_still_goes_through(ledger):
    inner = Counting()
    await aspend(ledger, 0.99)

    await guarded(inner, ledger, cap=1.00).complete(REQUEST)

    assert inner.calls == 1


async def test_yesterdays_spending_does_not_count_against_today(ledger):
    inner = Counting()
    await aspend(ledger, 5.00, day="2026-11-27")

    await guarded(inner, ledger, cap=1.00).complete(REQUEST)

    assert inner.calls == 1


async def test_a_model_without_a_price_is_refused_since_the_cap_cannot_measure_it(ledger):
    inner = Counting(model="mystery-model")

    with pytest.raises(LlmError) as caught:
        await guarded(inner, ledger).complete(REQUEST)

    assert caught.value.code == LlmErrorCode.UNPRICED_MODEL
    assert inner.calls == 0


async def test_a_cached_answer_is_still_served_once_the_cap_is_reached(ledger, tmp_path):
    inner = Counting()
    client = CachingLlmClient(guarded(inner, ledger, cap=1.00), ResponseCache(tmp_path / "cache.sqlite3"))
    await client.complete(REQUEST)  # under the cap: answered by the provider and cached
    await aspend(ledger, 1.00)

    again = await client.complete(REQUEST)  # the cap is reached, but this one costs nothing
    with pytest.raises(LlmError) as caught:
        await client.complete(LlmRequest("plan", "You plan decks.", "Another outline.", PlanOutput))

    assert again.usage.cached
    assert caught.value.code == LlmErrorCode.SPEND_CAP_REACHED
    assert inner.calls == 1


async def test_a_refusal_is_recorded_in_the_ledger_and_costs_nothing(ledger):
    await aspend(ledger, 1.00)
    client = LedgerLlmClient(guarded(Counting(), ledger), ledger, now=lambda: NOW)

    with pytest.raises(LlmError):
        await client.complete(REQUEST)

    refused = (await ledger.recent())[0]
    assert (refused.outcome, refused.cost_usd) == ("SPEND_CAP_REACHED", 0.0)
    assert await ledger.spent("2026-11-28") == pytest.approx(1.00)  # the refusal added nothing


# ---------- Through the app and the eval runner ----------


def test_at_the_cap_real_providers_are_refused_and_the_fake_still_answers(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")  # registered, but never reached: refused first
    monkeypatch.setenv("DAILY_SPEND_CAP_USD", "0.50")
    today = datetime.now(UTC).date().isoformat()
    ledger = Ledger(os.environ["LEDGER_DB"])
    asyncio.run(aspend(ledger, 0.50, day=today))

    with TestClient(create_app()) as client:
        gemini = client.post("/api/plan", json=plan_request(provider="gemini").model_dump(mode="json", by_alias=True))
        fake = client.post("/api/plan", json=plan_request().model_dump(mode="json", by_alias=True))
        view = client.get("/api/ledger").json()

    assert (gemini.status_code, gemini.json()["error"]) == (503, "SPEND_CAP_REACHED")
    assert fake.status_code == 200
    assert view["dailyCapUsd"] == 0.50
    assert [e["outcome"] for e in view["entries"][:2]] == ["ok", "SPEND_CAP_REACHED"]  # newest first


def test_a_plan_over_the_input_budget_is_a_413(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("MAX_INPUT_TOKENS", "100")

    with TestClient(create_app()) as client:
        response = client.post("/api/plan", json=plan_request(provider="gemini").model_dump(mode="json", by_alias=True))

    assert (response.status_code, response.json()["error"]) == (413, "INPUT_TOO_LARGE")


async def test_an_eval_run_stops_when_the_cap_is_reached(ledger, tmp_path):
    llms = LlmRegistry([FakeLlmClient(), guarded(AsReal(), ledger, cap=0.0)])  # $0 cap: refused from the start

    run = await start_eval(load_golden_set()[:2], llms, EvalSettings("claude", DEFAULT_PROMPTS), tmp_path / "runs")

    assert run.stopped is not None and run.stopped.error_code == "SPEND_CAP_REACHED"
    assert run.decks == {}  # nothing scored: a refusal isn't the model's fault


def test_guardrails_come_from_the_environment(monkeypatch):
    assert Guardrails.from_env() == Guardrails(8_000, 1.00)
    monkeypatch.setenv("MAX_INPUT_TOKENS", "2000")
    monkeypatch.setenv("DAILY_SPEND_CAP_USD", "0.25")
    assert Guardrails.from_env() == Guardrails(2_000, 0.25)
