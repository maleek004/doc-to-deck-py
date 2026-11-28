"""The response cache (3.0): repeat requests are answered from SQLite instead of the provider.

The inner client is a scripted FakeLlmClient, so every test can count how many requests really reached the
"provider" (`fake.requests`). One test puts the real GeminiClient behind the cache with respx, to show that the
second identical call never touches the network.
"""

import sqlite3

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from doctodeck.app import create_app
from doctodeck.llm.cache import CachingLlmClient, ResponseCache, cache_key
from doctodeck.llm.contracts import LlmError, LlmErrorCode, LlmRequest
from doctodeck.llm.fake import FakeLlmClient
from doctodeck.llm.gemini import GeminiClient, GeminiOptions
from doctodeck.pipeline.models import CardsOutput, PlanOutput

PLAN_JSON = '{"modules":[{"title":"A","description":"d","estimatedCards":4,"sectionIndices":[0]}]}'
OTHER_PLAN_JSON = '{"modules":[{"title":"B","description":"d","estimatedCards":5,"sectionIndices":[0]}]}'
REQUEST = LlmRequest("plan", "You plan decks.", "Plan this outline.", PlanOutput)


@pytest.fixture
def cache(tmp_path) -> ResponseCache:
    return ResponseCache(tmp_path / "cache" / "llm.sqlite3")  # the parent folder doesn't exist yet: it is made


def scripted(*answers: str, provider: str = "gemini") -> FakeLlmClient:
    fake = FakeLlmClient(provider=provider)
    for answer in answers:
        fake.returns(answer)
    return fake


# ---------- Hits and misses ----------


async def test_the_second_identical_request_is_answered_from_the_cache(cache):
    fake = scripted(PLAN_JSON, OTHER_PLAN_JSON)
    client = CachingLlmClient(fake, cache)

    first = await client.complete(REQUEST)
    second = await client.complete(REQUEST)

    assert len(fake.requests) == 1  # the provider was asked once
    assert second.data == first.data  # the stored answer, not the second scripted one
    assert (first.usage.cached, second.usage.cached) == (False, True)


async def test_a_hit_reports_the_original_calls_tokens_and_latency(cache):
    client = CachingLlmClient(scripted(PLAN_JSON), cache)

    first = await client.complete(REQUEST)
    second = await client.complete(REQUEST)

    # So a side-by-side comparison still shows what the model cost and how long it took.
    assert second.usage.model_dump(exclude={"cached"}) == first.usage.model_dump(exclude={"cached"})


async def test_the_cache_outlives_the_process(cache):
    await CachingLlmClient(scripted(PLAN_JSON), cache).complete(REQUEST)

    fresh = scripted(OTHER_PLAN_JSON)
    result = await CachingLlmClient(fresh, ResponseCache(cache.path)).complete(REQUEST)  # a "restarted app"

    assert result.data.modules[0].title == "A"
    assert fresh.requests == []


@pytest.mark.parametrize(
    "changed",
    [
        LlmRequest("plan", "You plan decks.", "Plan ANOTHER outline.", PlanOutput),
        LlmRequest("plan", "You plan decks carefully.", "Plan this outline.", PlanOutput),
        LlmRequest("plan", "You plan decks.", "Plan this outline.", CardsOutput),
    ],
    ids=["user-prompt", "system-prompt", "schema"],
)
async def test_any_change_to_the_request_is_a_miss(cache, changed):
    second_answer = '{"cards": []}' if changed.output_type is CardsOutput else OTHER_PLAN_JSON
    fake = scripted(PLAN_JSON, second_answer)
    client = CachingLlmClient(fake, cache)

    await client.complete(REQUEST)
    await client.complete(changed)

    assert len(fake.requests) == 2


async def test_another_model_or_provider_is_a_miss(cache):
    gemini, claude = scripted(PLAN_JSON), scripted(PLAN_JSON, provider="claude")
    other_model = scripted(PLAN_JSON)
    other_model.model = "fake-bigger"

    for client in (gemini, claude, other_model):
        await CachingLlmClient(client, cache).complete(REQUEST)

    assert [len(c.requests) for c in (gemini, claude, other_model)] == [1, 1, 1]


async def test_changed_generation_settings_are_a_miss(cache):
    warm, cold = scripted(PLAN_JSON), scripted(PLAN_JSON)
    warm.settings = {"temperature": 0.9}
    cold.settings = {"temperature": 0.2}

    await CachingLlmClient(warm, cache).complete(REQUEST)
    await CachingLlmClient(cold, cache).complete(REQUEST)

    assert len(cold.requests) == 1  # an answer made at 0.9 must not come back for 0.2


def test_the_key_is_canonical_whatever_order_the_settings_were_built_in():
    one, two = FakeLlmClient(), FakeLlmClient()
    one.settings = {"temperature": 0.2, "thinkingConfig": {"thinkingLevel": "low"}}
    two.settings = {"thinkingConfig": {"thinkingLevel": "low"}, "temperature": 0.2}

    assert cache_key(one, REQUEST) == cache_key(two, REQUEST)
    assert len(cache_key(one, REQUEST)) == 64  # a SHA-256 in hex, however long the prompt


async def test_real_clients_put_their_settings_in_the_key():
    async with httpx.AsyncClient() as http:
        cool = GeminiClient(GeminiOptions(api_key="k", temperature=0.2), http)
        hot = GeminiClient(GeminiOptions(api_key="k", temperature=0.9), http)

        assert cache_key(cool, REQUEST) != cache_key(hot, REQUEST)


# ---------- What is never stored ----------


async def test_a_failure_is_not_cached_and_the_next_try_really_tries(cache):
    fake = FakeLlmClient(provider="gemini").fails(LlmErrorCode.RATE_LIMITED, "429").returns(PLAN_JSON)
    client = CachingLlmClient(fake, cache)

    with pytest.raises(LlmError):
        await client.complete(REQUEST)
    assert await cache.count() == 0

    result = await client.complete(REQUEST)

    assert result.data.modules[0].title == "A"
    assert len(fake.requests) == 2
    assert await cache.count() == 1


async def test_an_answer_of_the_wrong_shape_is_not_cached(cache):
    client = CachingLlmClient(scripted('{"modules": "none"}'), cache)

    with pytest.raises(LlmError) as caught:
        await client.complete(REQUEST)

    assert caught.value.code == LlmErrorCode.INVALID_OUTPUT
    assert await cache.count() == 0


async def test_an_unreadable_entry_is_dropped_and_the_provider_asked(cache):
    fake = scripted(PLAN_JSON, OTHER_PLAN_JSON)
    client = CachingLlmClient(fake, cache)
    await client.complete(REQUEST)
    with sqlite3.connect(cache.path) as db:
        db.execute("UPDATE responses SET data = ?", ('{"modules": "tampered"}',))

    result = await client.complete(REQUEST)

    assert result.data.modules[0].title == "B"  # asked again instead of failing the user's request
    assert result.usage.cached is False
    assert len(fake.requests) == 2


# ---------- The wrapper is invisible ----------


async def test_the_wrapper_has_the_inner_clients_identity_and_closes_it(cache):
    class Closable(FakeLlmClient):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    inner = Closable(provider="gemini")
    client = CachingLlmClient(inner, cache)

    assert (client.provider, client.model, client.is_real) == ("gemini", "fake-demo", False)
    await client.aclose()
    assert inner.closed


async def test_with_the_real_gemini_client_the_second_call_never_reaches_the_network(cache):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    answer = {
        "candidates": [{"content": {"parts": [{"text": PLAN_JSON}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 30},
    }
    with respx.mock as mock:
        route = mock.post(url).respond(200, json=answer)
        async with httpx.AsyncClient() as http:
            client = CachingLlmClient(GeminiClient(GeminiOptions(api_key="k"), http), cache)
            first = await client.complete(REQUEST)
            second = await client.complete(REQUEST)

    assert route.call_count == 1
    assert second.usage.prompt_tokens == first.usage.prompt_tokens == 120
    assert second.usage.cached


# ---------- Wiring ----------


def test_llm_cache_wraps_real_providers_only(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LLM_CACHE", str(tmp_path / "llm.sqlite3"))

    with TestClient(create_app()) as client:
        clients = {c.provider: c for c in client.app.state.llms.all}
        providers = client.get("/api/providers").json()

    # The app wraps every client in the ledger (outermost); look underneath it.
    assert isinstance(clients["gemini"].inner, CachingLlmClient)
    assert not isinstance(clients["fake"].inner, CachingLlmClient)  # free, and tests script it
    assert [p["provider"] for p in providers] == ["fake", "gemini"]  # the dropdown can't tell the difference
    assert (tmp_path / "llm.sqlite3").exists()


def test_no_llm_cache_means_no_wrapper(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    with TestClient(create_app()) as client:
        gemini = client.app.state.llms.get("gemini")

    assert isinstance(gemini.inner.inner, GeminiClient)  # under the ledger, only the guard (4.3)
