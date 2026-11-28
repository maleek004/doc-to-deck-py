"""The credit wallet and the charge around generation (4.2). Fake providers only: no key, no network."""

import asyncio
import json
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

from doctodeck.app import create_app, get_llm_registry, get_wallet
from doctodeck.costs.billing import InsufficientCredits, generate_charged
from doctodeck.costs.credits import generation_credits, ingest_credits
from doctodeck.costs.ledger import deck_scope
from doctodeck.costs.wallet import Wallet
from doctodeck.llm.contracts import LlmErrorCode, LlmRequest, LlmResult
from doctodeck.llm.fake import FakeLlmClient
from doctodeck.llm.registry import LlmRegistry
from doctodeck.pipeline.generator import CardGenerator
from doctodeck.pipeline.models import PipelineOptions
from tests.layer2.data import CLEAN_TEXT, TWO_MODULES, generate_request

# The fake's demo answer for TWO_MODULES: one card per section, 4 cards. The text is short: the 2-credit minimum.
DEMO_COST = generation_credits(CLEAN_TEXT, 4)  # 2 + 4 = 6
PRE_FLIGHT = ingest_credits(CLEAN_TEXT) + PipelineOptions().default_target_cards  # 2 + 10 cards asked for = 12


def as_json(model) -> dict:
    return model.model_dump(mode="json", by_alias=True)


@pytest.fixture
def wallet_at(tmp_path):
    def make(credits: int) -> Wallet:
        return Wallet(tmp_path / "test-wallet.sqlite3", start_credits=credits)

    return make


def generator(*clients) -> CardGenerator:
    return CardGenerator(LlmRegistry([FakeLlmClient(), *clients]), PipelineOptions())


class Gate:
    """A provider that holds every call until `n` calls have arrived (or until `release()`), then answers like the
    fake. `entered` is set as soon as one call is waiting."""

    model, is_real = "gated", False

    def __init__(self, n: int = 1_000_000, provider: str = "gated") -> None:
        self.provider, self.n, self.arrived = provider, n, 0
        self.entered, self._open = asyncio.Event(), asyncio.Event()
        self._inner = FakeLlmClient(provider=provider)

    def release(self) -> None:
        self._open.set()

    async def complete[T](self, request: LlmRequest[T]) -> LlmResult[T]:
        self.arrived += 1
        self.entered.set()
        if self.arrived >= self.n:
            self._open.set()
        await asyncio.wait_for(self._open.wait(), timeout=5)
        return await self._inner.complete(request)


# ---------- Prices ----------


def test_the_ingest_fee_is_2_credits_per_started_500_words_of_what_is_sent():
    assert ingest_credits("") == 2  # the minimum
    assert ingest_credits("word " * 500) == 2
    assert ingest_credits("word " * 501) == 4
    assert generation_credits("word " * 1200, 10) == 6 + 10


def test_a_cut_document_pays_for_the_text_the_model_was_sent(wallet_at):
    options = PipelineOptions(max_source_chars=5 * 500)  # 500 five-letter words fit ("word ")
    llms = LlmRegistry([FakeLlmClient()])
    wallet = wallet_at(100)
    request = generate_request(TWO_MODULES, text="word " * 3000)  # 3000 words, 6 started 500s: 12 if all were sent

    result = asyncio.run(generate_charged(CardGenerator(llms, options), wallet, request, options))

    assert result.truncated
    assert result.credits.ingest_credits == 2


# ---------- The wallet ----------


async def test_a_new_wallet_opens_with_its_start_balance_once(wallet_at):
    assert await wallet_at(30).balance() == 30
    assert await wallet_at(999).balance() == 30  # an existing wallet keeps its balance

    [opening] = await wallet_at(30).transactions()
    assert (opening.action, opening.amount, opening.balance_after) == ("opening", 30, 30)


async def test_a_charge_takes_the_credits_and_records_why(wallet_at):
    wallet = wallet_at(10)

    assert await wallet.charge(6, "generate", deck_id="d1") == 4

    latest = (await wallet.transactions())[0]
    assert (latest.action, latest.amount, latest.balance_after, latest.deck_id) == ("generate", -6, 4, "d1")


async def test_a_charge_the_balance_cannot_cover_changes_nothing(wallet_at):
    wallet = wallet_at(5)

    assert await wallet.charge(6, "generate") is None
    assert await wallet.balance() == 5
    assert len(await wallet.transactions()) == 1  # only the opening


async def test_a_charge_must_be_positive(wallet_at):
    with pytest.raises(ValueError, match="positive"):
        await wallet_at(5).charge(0, "generate")  # a negative charge would be a top-up in disguise


async def test_concurrent_charges_never_take_more_than_the_balance(wallet_at):
    wallet = wallet_at(20)

    results = await asyncio.gather(*(wallet.charge(1, "generate") for _ in range(50)))

    assert sum(r is not None for r in results) == 20
    assert await wallet.balance() == 0
    assert sorted(r for r in results if r is not None) == list(range(20))  # every balance from 19 down to 0, once


def test_the_database_refuses_a_negative_balance_even_without_the_where(wallet_at):
    wallet = wallet_at(5)

    with sqlite3.connect(wallet.path) as db, pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        db.execute("UPDATE wallet SET balance = balance - 6")


async def test_a_top_up_is_applied_once_per_reference(wallet_at):
    wallet = wallet_at(0)

    first = await wallet.top_up(100, "click-1")
    again = await wallet.top_up(100, "click-1")
    other = await wallet.top_up(50, "click-2")

    assert (first.applied, first.balance) == (True, 100)
    assert (again.applied, again.balance) == (False, 100)
    assert (other.applied, other.balance) == (True, 150)


async def test_concurrent_top_ups_with_one_reference_pay_once(wallet_at):
    wallet = wallet_at(0)

    results = await asyncio.gather(*(wallet.top_up(100, "retried-request") for _ in range(10)))

    assert sum(r.applied for r in results) == 1
    assert await wallet.balance() == 100


# ---------- Generate, then charge ----------


async def test_twenty_concurrent_generates_on_credits_for_five_get_exactly_five(wallet_at):
    wallet = wallet_at(5 * DEMO_COST)
    gate = Gate(n=20)  # every request passes the pre-flight before any of them is charged
    app = create_app()
    app.dependency_overrides[get_llm_registry] = lambda: LlmRegistry([FakeLlmClient(), gate])
    app.dependency_overrides[get_wallet] = lambda: wallet
    body = as_json(generate_request(TWO_MODULES, provider="gated"))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        responses = await asyncio.gather(*(client.post("/api/generate", json=body) for _ in range(20)))

    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200] * 5 + [402] * 15
    assert gate.arrived == 20  # all twenty reached the model: the pre-flight let them through, the UPDATE didn't
    assert await wallet.balance() == 0
    assert sum(t.action == "generate" for t in await wallet.transactions(50)) == 5
    refused = next(r for r in responses if r.status_code == 402).json()
    assert refused["error"] == "INSUFFICIENT_CREDITS"


def test_a_successful_generate_reports_its_charge(monkeypatch):
    with TestClient(create_app()) as client:
        body = as_json(generate_request(TWO_MODULES)) | {"deckId": "deck-1"}
        result = client.post("/api/generate", json=body).json()
        wallet = client.get("/api/wallet").json()

    assert result["credits"] == {"ingestCredits": 2, "cardCredits": 4, "totalCredits": 6, "cardsHeldBack": 0,
                                 "balanceAfter": 94}  # fmt: skip
    assert wallet["balance"] == 94
    latest = wallet["transactions"][0]
    assert (latest["action"], latest["amount"], latest["balanceAfter"]) == ("generate", -6, 94)
    assert latest["deckId"] == "deck-1"  # the same id the usage ledger files the calls under


def test_a_provider_error_charges_nothing():
    scripted = FakeLlmClient(provider="scripted").fails(LlmErrorCode.PROVIDER_ERROR)
    app = create_app()
    app.dependency_overrides[get_llm_registry] = lambda: LlmRegistry([FakeLlmClient(), scripted])
    with TestClient(app) as client:
        response = client.post("/api/generate", json=as_json(generate_request(TWO_MODULES, provider="scripted")))
        wallet = client.get("/api/wallet").json()

    assert response.status_code == 502
    assert wallet["balance"] == 100
    assert [t["action"] for t in wallet["transactions"]] == ["opening"]


def test_the_pre_flight_refuses_before_any_call_is_made(monkeypatch):
    monkeypatch.setenv("WALLET_START_CREDITS", str(PRE_FLIGHT - 1))
    scripted = FakeLlmClient(provider="scripted")
    app = create_app()
    app.dependency_overrides[get_llm_registry] = lambda: LlmRegistry([FakeLlmClient(), scripted])
    with TestClient(app) as client:
        response = client.post("/api/generate", json=as_json(generate_request(TWO_MODULES, provider="scripted")))

    assert response.status_code == 402
    assert response.json()["message"] == f"This needs {PRE_FLIGHT} credits and the wallet has {PRE_FLIGHT - 1}."
    assert scripted.requests == []  # no money spent on the provider


async def test_extra_cards_the_balance_cannot_cover_are_held_back(wallet_at):
    cards = [{"front": f"Question number {i} about topic {i}?", "back": f"Answer {i}.", "moduleIndex": 1,
              "topicTag": "t", "rationale": "r"} for i in range(6)]  # fmt: skip
    scripted = FakeLlmClient(provider="scripted").returns(json.dumps({"cards": cards}))
    wallet = wallet_at(6)  # asked for 3 cards: pre-flight needs 2 + 3 = 5; after the ingest fee, 4 cards are covered
    request = generate_request(TWO_MODULES, provider="scripted", cards=3)

    result = await generate_charged(generator(scripted), wallet, request, PipelineOptions())

    assert len(result.cards) == 4
    assert (result.credits.cards_held_back, result.credits.total_credits, result.credits.balance_after) == (2, 6, 0)


async def test_a_cancellation_after_the_cards_exist_still_charges(wallet_at):
    class SlowWallet(Wallet):
        """Stops at the start of the charge until the test lets it go, so the test can cancel in between."""

        reached, go, charged = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def charge(self, *args, **kwargs):
            self.reached.set()
            await self.go.wait()
            result = await super().charge(*args, **kwargs)
            self.charged.set()
            return result

    wallet = SlowWallet(wallet_at(50).path)
    request = generate_request(TWO_MODULES)

    with deck_scope("cancelled-deck"):
        task = asyncio.create_task(generate_charged(generator(), wallet, request, PipelineOptions()))
    await wallet.reached.wait()
    task.cancel()  # the client went away after the model answered
    with pytest.raises(asyncio.CancelledError):
        await task
    wallet.go.set()
    await asyncio.wait_for(wallet.charged.wait(), timeout=5)

    assert await wallet.balance() == 50 - DEMO_COST
    latest = (await wallet.transactions())[0]
    assert (latest.action, latest.deck_id) == ("generate", "cancelled-deck")  # the shielded task kept the deck id


async def test_a_cancellation_during_the_model_call_charges_nothing(wallet_at):
    wallet = wallet_at(50)
    gate = Gate()

    task = asyncio.create_task(
        generate_charged(generator(gate), wallet, generate_request(TWO_MODULES, provider="gated"), PipelineOptions())
    )
    await gate.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await wallet.balance() == 50


async def test_insufficient_credits_is_an_exception_with_the_numbers(wallet_at):
    with pytest.raises(InsufficientCredits) as caught:
        await generate_charged(generator(), wallet_at(0), generate_request(TWO_MODULES), PipelineOptions())

    assert (caught.value.required, caught.value.balance) == (PRE_FLIGHT, 0)


# ---------- Top-up endpoint ----------


def test_the_top_up_endpoint_is_idempotent_and_validates():
    with TestClient(create_app()) as client:
        first = client.post("/api/wallet/top-up", json={"credits": 100, "reference": "r1"}).json()
        again = client.post("/api/wallet/top-up", json={"credits": 100, "reference": "r1"}).json()
        bad = client.post("/api/wallet/top-up", json={"credits": 0, "reference": "r2"})

    assert first == {"applied": True, "balance": 200}
    assert again == {"applied": False, "balance": 200}
    assert (bad.status_code, bad.json()["error"]) == (400, "INVALID_REQUEST")
