"""The Gemini client against a faked HTTP server (respx). No test touches the network or needs a key.

respx intercepts httpx at the transport level: the client builds and sends real requests, and respx answers
them from the routes a test declares. Waits between retries are recorded instead of slept.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from doctodeck.app import create_app
from doctodeck.llm.contracts import LlmError, LlmErrorCode, LlmRequest
from doctodeck.llm.gemini import GeminiClient, GeminiOptions
from doctodeck.pipeline.models import PlanOutput

KEY = "test-key-not-real"
URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
PLAN_JSON = '{"modules":[{"title":"A","description":"d","estimatedCards":4,"sectionIndices":[0]}]}'
REQUEST = LlmRequest("plan", "You plan decks.", "Plan this outline.", PlanOutput)


def envelope(text: str = PLAN_JSON, finish: str = "STOP", **usage: int) -> dict:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 30} | usage,
    }


def ok(**kwargs) -> httpx.Response:
    return httpx.Response(200, json=envelope(**kwargs))


def error(status: int, message: str = "boom", headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "message": message}}, headers=headers)


@pytest.fixture
def waits() -> list[float]:
    return []


@pytest.fixture
async def client(waits):
    async def record(seconds: float) -> None:
        waits.append(seconds)

    async with httpx.AsyncClient() as http:
        yield GeminiClient(GeminiOptions(api_key=KEY), http, sleep=record, random_fraction=lambda: 0.5)


@pytest.fixture
def gemini():
    with respx.mock(assert_all_called=False) as mock:
        yield mock.post(URL)


async def fails_with(client: GeminiClient) -> LlmError:
    with pytest.raises(LlmError) as caught:
        await client.complete(REQUEST)
    return caught.value


# ---------- The request ----------


async def test_sends_the_key_in_a_header_and_the_schema_in_gemini_dialect(client, gemini):
    gemini.return_value = ok()

    await client.complete(REQUEST)

    sent = gemini.calls.last.request
    assert sent.headers["x-goog-api-key"] == KEY
    assert KEY not in str(sent.url)
    body = json.loads(sent.content)
    assert body["systemInstruction"]["parts"][0]["text"] == "You plan decks."
    assert body["contents"][0]["parts"][0]["text"] == "Plan this outline."
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"]["type"] == "OBJECT"
    assert config["thinkingConfig"] == {"thinkingLevel": "low"}


@pytest.mark.parametrize(
    ("model", "thinking"),
    [
        ("gemini-3.8-flash", {"thinkingLevel": "low"}),
        ("gemini-2.5-flash", {"thinkingBudget": 0}),  # live run: 821 → 80 output tokens, 30 s timeouts → 1.3 s
        ("gemini-2.5-pro", None),  # Pro can't turn thinking off: leave its default
    ],
)
async def test_thinking_is_kept_low_in_each_models_own_terms(model, thinking):
    async with httpx.AsyncClient() as http:
        client = GeminiClient(GeminiOptions(api_key=KEY, model=model), http)
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent")
            route.return_value = ok()
            await client.complete(REQUEST)

    assert json.loads(route.calls.last.request.content)["generationConfig"].get("thinkingConfig") == thinking


# ---------- A good answer ----------


async def test_returns_the_validated_answer_and_token_usage(client, gemini):
    gemini.return_value = ok(thoughtsTokenCount=50)

    result = await client.complete(REQUEST)

    assert result.data.modules[0].estimated_cards == 4
    assert (result.usage.provider, result.usage.model) == ("gemini", "gemini-3.8-flash")
    assert result.usage.prompt_tokens == 120
    assert result.usage.completion_tokens == 80  # 30 answer + 50 thinking: both billed as output


async def test_skips_thought_parts_and_joins_the_rest(client, gemini):
    half = len(PLAN_JSON) // 2
    parts = [{"text": "Let me think...", "thought": True}, {"text": PLAN_JSON[:half]}, {"text": PLAN_JSON[half:]}]
    gemini.return_value = httpx.Response(200, json={"candidates": [{"content": {"parts": parts}}]})

    result = await client.complete(REQUEST)

    assert result.data.modules[0].title == "A"


async def test_json_of_the_wrong_shape_is_invalid_output(client, gemini):
    gemini.return_value = ok(text='{"modules": "none"}')

    assert (await fails_with(client)).code == LlmErrorCode.INVALID_OUTPUT


# ---------- Retries ----------


async def test_a_429_is_retried_and_then_succeeds(client, gemini, waits):
    gemini.side_effect = [error(429), ok()]

    await client.complete(REQUEST)

    assert gemini.call_count == 2
    assert waits == [0.5 + 0.5 * 0.25]  # base delay + jitter (random_fraction is pinned at 0.5)


async def test_backoff_doubles_each_retry(client, gemini, waits):
    gemini.side_effect = [error(503), error(500), error(502), ok()]

    await client.complete(REQUEST)

    assert waits == [0.625, 1.125, 2.125]  # 0.5, 1, 2 seconds, each + 0.125 jitter


async def test_retry_after_in_seconds_sets_the_wait(client, gemini, waits):
    gemini.side_effect = [error(429, headers={"Retry-After": "7"}), ok()]

    await client.complete(REQUEST)

    assert waits == [7.0]


async def test_retry_after_as_an_http_date_sets_the_wait(client, gemini, waits):
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=5), usegmt=True)
    gemini.side_effect = [error(503, headers={"Retry-After": when}), ok()]

    await client.complete(REQUEST)

    assert 3 < waits[0] <= 5  # the date has 1-second precision, and a moment passes


def quota_error(
    retry_delay: str, quota_id: str = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
) -> httpx.Response:
    """A 429 as Gemini's free tier really sends it: the wait is in the body as RetryInfo, not in a header."""
    return httpx.Response(429, json={"error": {
        "code": 429,
        "message": "You exceeded your current quota. Please retry in 18.2s.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
        ],
    }})  # fmt: skip


async def test_a_short_retry_delay_in_the_body_sets_the_wait(client, gemini, waits):
    gemini.side_effect = [quota_error("3s"), ok()]

    await client.complete(REQUEST)

    assert waits == [3.0]


async def test_a_retry_delay_longer_than_the_cap_fails_at_once_and_says_when(client, gemini, waits):
    gemini.side_effect = [quota_error("18s"), ok()]

    failure = await fails_with(client)

    assert failure.code == LlmErrorCode.RATE_LIMITED
    assert gemini.call_count == 1  # retrying sooner than asked is certain to fail and spends quota
    assert waits == []
    assert "Retry in about 18 s" in failure.message


async def test_the_daily_limit_says_so_and_gives_the_wait_in_hours(client, gemini):
    # The real shape Gemini sent on 2026-11-27 once gemini-2.5-flash's 20 free requests for the day were spent.
    gemini.return_value = quota_error("12678s", "GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    failure = await fails_with(client)

    assert gemini.call_count == 1
    assert "daily request limit" in failure.message
    assert "Retry in about 3.5 h" in failure.message


async def test_a_retry_after_header_longer_than_the_cap_fails_at_once(client, gemini, waits):
    gemini.side_effect = [error(503, headers={"Retry-After": "3600"}), ok()]

    failure = await fails_with(client)

    assert failure.code == LlmErrorCode.RATE_LIMITED
    assert gemini.call_count == 1
    assert "Retry in about 60 min" in failure.message


async def test_still_rate_limited_after_all_retries_is_rate_limited(client, gemini, waits):
    gemini.return_value = error(429)

    failure = await fails_with(client)

    assert failure.code == LlmErrorCode.RATE_LIMITED
    assert gemini.call_count == 4  # the first attempt + 3 retries
    assert len(waits) == 3  # no wait after the last attempt


async def test_an_overloaded_model_says_so_in_geminis_own_words(client, gemini):
    # The real body Gemini sent on 2026-11-25, when every flash model was overloaded.
    gemini.return_value = error(503, "This model is currently experiencing high demand. Please try again later.")

    failure = await fails_with(client)

    assert failure.code == LlmErrorCode.RATE_LIMITED
    assert "overloaded" in failure.message
    assert "experiencing high demand" in failure.message


async def test_a_quota_error_is_told_apart_from_an_overload(client, gemini):
    gemini.return_value = error(429, "Quota exceeded for metric: generate_content_free_tier_requests.")

    failure = await fails_with(client)

    assert failure.code == LlmErrorCode.RATE_LIMITED
    assert "quota" in failure.message
    assert "generate_content_free_tier_requests" in failure.message


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_errors_that_would_repeat_are_not_retried(client, gemini, waits, status):
    gemini.return_value = error(status, "API key not valid. Please pass a valid API key.")

    failure = await fails_with(client)

    assert failure.code == LlmErrorCode.PROVIDER_ERROR
    assert gemini.call_count == 1
    assert waits == []
    assert KEY not in failure.message


async def test_a_400_carries_geminis_own_message(client, gemini):
    gemini.return_value = error(400, "Invalid JSON payload received.")

    assert "Invalid JSON payload received." in (await fails_with(client)).message


async def test_a_timeout_is_retried_only_once_then_reported_as_timeout(client, gemini, waits):
    gemini.side_effect = httpx.ReadTimeout("slow")

    failure = await fails_with(client)

    assert failure.code == LlmErrorCode.TIMEOUT
    assert gemini.call_count == 2  # max_timeout_retries=1: a slow request will likely be slow again
    assert len(waits) == 1
    assert isinstance(failure.__cause__, httpx.ReadTimeout)


async def test_a_timeout_then_an_overload_still_gets_the_other_retries(client, gemini):
    gemini.side_effect = [httpx.ReadTimeout("slow"), error(503), ok()]

    result = await client.complete(REQUEST)

    assert result.data.modules[0].title == "A"
    assert gemini.call_count == 3  # the timeout limit is its own; other transient failures use max_retries


def test_the_default_deadline_leaves_room_for_a_big_batch():
    assert GeminiOptions(api_key=KEY).attempt_timeout >= 90


async def test_a_slow_attempt_hits_the_overall_deadline(gemini, waits):
    attempts = []

    async def slow(request):
        attempts.append(request)  # respx records a call only once it returns, and these never do
        await asyncio.sleep(5)
        return ok()

    gemini.side_effect = slow
    options = GeminiOptions(api_key=KEY, attempt_timeout=0.05, max_retries=1)

    async def record(seconds: float) -> None:
        waits.append(seconds)

    async with httpx.AsyncClient() as http:
        failure = await fails_with(GeminiClient(options, http, sleep=record))

    assert failure.code == LlmErrorCode.TIMEOUT
    assert len(attempts) == 2


async def test_a_network_error_is_retried(client, gemini):
    gemini.side_effect = [httpx.ConnectError("connection refused"), ok()]

    result = await client.complete(REQUEST)

    assert result.data.modules[0].title == "A"
    assert gemini.call_count == 2


async def test_a_cancelled_call_stops_at_once_and_is_not_retried(gemini):
    gemini.return_value = error(503)
    # Real sleeping this time, with a long backoff, so the task is parked in the wait when it is cancelled.
    async with httpx.AsyncClient() as http:
        client = GeminiClient(GeminiOptions(api_key=KEY, base_delay=60), http)
        task = asyncio.create_task(client.complete(REQUEST))
        while gemini.call_count == 0:
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):  # not an LlmError: cancellation is never translated
            await task

    assert gemini.call_count == 1


# ---------- Unusable 200 responses ----------


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"promptFeedback": {"blockReason": "SAFETY"}}, LlmErrorCode.REFUSED),
        ({"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}, LlmErrorCode.REFUSED),
        ({"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": '{"mod'}]}}]},
         LlmErrorCode.TRUNCATED),
        ({"candidates": []}, LlmErrorCode.PROVIDER_ERROR),
        ({"candidates": [{"finishReason": "STOP", "content": {"parts": []}}]}, LlmErrorCode.INVALID_OUTPUT),
        ([], LlmErrorCode.PROVIDER_ERROR),
    ],
)  # fmt: skip
async def test_unusable_200_responses_become_typed_errors(client, gemini, waits, body, code):
    gemini.return_value = httpx.Response(200, json=body)

    assert (await fails_with(client)).code == code
    assert waits == []  # a 200 is never retried: asking again would cost money and likely fail the same way


async def test_a_200_that_is_not_json_is_a_provider_error(client, gemini):
    gemini.return_value = httpx.Response(200, text="<html>proxy error</html>")

    assert (await fails_with(client)).code == LlmErrorCode.PROVIDER_ERROR


# ---------- Wiring ----------


def test_gemini_is_offered_only_when_a_key_is_set(monkeypatch):
    with TestClient(create_app()) as client:
        assert [p["provider"] for p in client.get("/api/providers").json()] == ["fake"]

    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    with TestClient(create_app()) as client:
        providers = client.get("/api/providers").json()

    assert providers[1] == {"provider": "gemini", "model": "gemini-test-model", "isReal": True}
