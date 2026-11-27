"""The Claude client with the real `anthropic` SDK, over a fake transport. No network, no key.

The SDK runs on httpx2, so respx (which patches httpx) can't see its traffic. Instead the SDK is given its own
`DefaultAsyncHttpxClient` with an `httpx2.MockTransport`: every request the SDK builds, retries included, goes
to a handler function in the test. Error responses carry `retry-after-ms: 1`, which the SDK honours, so retry
tests don't wait out real backoff.
"""

import asyncio
import json
from collections.abc import Callable

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from doctodeck.app import create_app
from doctodeck.llm.claude import ClaudeClient, ClaudeOptions
from doctodeck.llm.contracts import LlmError, LlmErrorCode, LlmRequest
from doctodeck.pipeline.models import PlanOutput

KEY = "sk-ant-test-not-real"
PLAN_JSON = '{"modules":[{"title":"A","description":"d","estimatedCards":4,"sectionIndices":[0]}]}'
REQUEST = LlmRequest("plan", "You plan decks.", "Plan this outline.", PlanOutput)


def message(text: str = PLAN_JSON, stop_reason: str = "end_turn", model: str = "claude-haiku-5-5", **extra) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 300, "output_tokens": 90, "cache_read_input_tokens": 20},
    } | extra


def ok(**kwargs) -> httpx2.Response:
    return httpx2.Response(200, json=message(**kwargs))


def error(status: int, kind: str, text: str = "boom") -> httpx2.Response:
    body = {"type": "error", "error": {"type": kind, "message": text}}
    return httpx2.Response(status, json=body, headers={"retry-after-ms": "1"})


class FakeClaude:
    """Answers each request with the next scripted response (or raises it, for network errors)."""

    def __init__(self, *responses: httpx2.Response | Exception | Callable) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        next_response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(next_response, Exception):
            raise next_response
        return next_response

    def client(self, model: str = "claude-haiku-5-5", max_retries: int = 3) -> ClaudeClient:
        options = ClaudeOptions(api_key=KEY, model=model, max_retries=max_retries)
        http = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self))
        sdk = anthropic.AsyncAnthropic(api_key=KEY, http_client=http, max_retries=max_retries)
        return ClaudeClient(options, sdk)

    def body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)


async def fails_with(client: ClaudeClient) -> LlmError:
    with pytest.raises(LlmError) as caught:
        await client.complete(REQUEST)
    return caught.value


# ---------- The request ----------


async def test_sends_the_prompts_and_the_schema_as_structured_output():
    fake = FakeClaude(ok())

    await fake.client().complete(REQUEST)

    sent = fake.requests[0]
    assert sent.headers["authorization"].endswith(KEY)  # SDK 2.x sends the key as Authorization, not x-api-key
    assert KEY not in str(sent.url)
    body = fake.body()
    assert body["model"] == "claude-haiku-5-5"
    assert body["system"] == "You plan decks."
    assert body["messages"] == [{"role": "user", "content": "Plan this outline."}]
    assert body["output_config"]["effort"] == "low"
    output_format = body["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    module = output_format["schema"]["$defs"]["PlanOutputModule"]  # $ref is fine for Claude: no inlining
    assert module["additionalProperties"] is False  # added by the SDK's transform_schema
    assert "temperature" not in body  # current models reject non-default sampling parameters


async def test_haiku_gets_no_fallback_while_a_model_that_supports_it_does():
    fake = FakeClaude(ok())

    await fake.client().complete(REQUEST)
    await fake.client(model="claude-sonnet-5-5").complete(REQUEST)

    assert "fallbacks" not in fake.body(0)
    assert fake.body(1)["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in fake.requests[1].headers["anthropic-beta"]


# ---------- A good answer ----------


async def test_returns_the_validated_answer_and_token_usage():
    result = await FakeClaude(ok()).client().complete(REQUEST)

    assert result.data.modules[0].estimated_cards == 4
    assert (result.usage.provider, result.usage.model) == ("claude", "claude-haiku-5-5")
    assert result.usage.prompt_tokens == 320  # 300 uncached + 20 read from cache: all input the call used
    assert result.usage.completion_tokens == 90


async def test_usage_names_the_model_that_actually_answered():
    result = await FakeClaude(ok(model="claude-opus-4-8")).client(model="claude-sonnet-5-5").complete(REQUEST)

    assert result.usage.model == "claude-opus-4-8"  # a fallback model served it: bill and report that one


async def test_json_of_the_wrong_shape_is_invalid_output():
    failure = await fails_with(FakeClaude(ok(text='{"modules": "none"}')).client())

    assert failure.code == LlmErrorCode.INVALID_OUTPUT


# ---------- Stop reasons, checked before the content ----------


async def test_a_refusal_is_refused_even_though_its_text_is_not_the_schema():
    refusal = ok(text="I can't help with that.", stop_reason="refusal",
                 stop_details={"type": "refusal", "category": "cyber", "explanation": None})  # fmt: skip

    failure = await fails_with(FakeClaude(refusal).client())

    assert failure.code == LlmErrorCode.REFUSED  # not INVALID_OUTPUT: why it stopped is read first
    assert "cyber" in failure.message


async def test_an_answer_cut_off_at_max_tokens_is_truncated():
    failure = await fails_with(FakeClaude(ok(text='{"modules":[{"tit', stop_reason="max_tokens")).client())

    assert failure.code == LlmErrorCode.TRUNCATED


async def test_an_answer_with_no_text_is_invalid_output():
    no_text = httpx2.Response(
        200, json=message() | {"content": [{"type": "thinking", "thinking": "", "signature": "s"}]}
    )

    assert (await fails_with(FakeClaude(no_text).client())).code == LlmErrorCode.INVALID_OUTPUT


# ---------- Retries: the SDK's, not ours ----------


async def test_the_sdk_retries_a_529_overload_and_then_succeeds():
    fake = FakeClaude(error(529, "overloaded_error"), ok())

    result = await fake.client().complete(REQUEST)

    assert result.data.modules[0].title == "A"
    assert len(fake.requests) == 2  # no retry loop in claude.py: this was the SDK


async def test_still_overloaded_after_the_sdks_retries_is_rate_limited_with_claudes_words():
    fake = FakeClaude(error(529, "overloaded_error", "Overloaded"))

    failure = await fails_with(fake.client(max_retries=3))

    assert failure.code == LlmErrorCode.RATE_LIMITED
    assert len(fake.requests) == 4  # the first attempt + 3 SDK retries
    assert "overloaded" in failure.message and "Overloaded" in failure.message


async def test_a_429_after_retries_is_rate_limited():
    failure = await fails_with(FakeClaude(error(429, "rate_limit_error", "Number of requests too high")).client())

    assert failure.code == LlmErrorCode.RATE_LIMITED
    assert "Number of requests too high" in failure.message


@pytest.mark.parametrize(
    ("status", "kind"),
    [(400, "invalid_request_error"), (401, "authentication_error"), (403, "permission_error"),
     (404, "not_found_error")],
)  # fmt: skip
async def test_errors_that_would_repeat_are_not_retried(status, kind):
    fake = FakeClaude(error(status, kind, "model: unknown"))

    failure = await fails_with(fake.client())

    assert failure.code == LlmErrorCode.PROVIDER_ERROR
    assert len(fake.requests) == 1
    assert KEY not in failure.message


async def test_a_timeout_is_timeout():
    failure = await fails_with(FakeClaude(httpx2.ReadTimeout("slow")).client(max_retries=0))

    assert failure.code == LlmErrorCode.TIMEOUT


async def test_a_network_error_is_retried_by_the_sdk():
    fake = FakeClaude(httpx2.ConnectError("refused"), ok())

    result = await fake.client(max_retries=1).complete(REQUEST)

    assert result.data.modules[0].title == "A"
    assert len(fake.requests) == 2


async def test_a_cancelled_call_stops_and_is_not_translated():
    started = asyncio.Event()

    async def hang(request: httpx2.Request) -> httpx2.Response:
        started.set()
        await asyncio.sleep(60)
        return ok()

    http = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(hang))
    client = ClaudeClient(ClaudeOptions(api_key=KEY), anthropic.AsyncAnthropic(api_key=KEY, http_client=http))

    task = asyncio.create_task(client.complete(REQUEST))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):  # not an LlmError, and not retried
        await task


# ---------- Wiring ----------


def test_claude_is_offered_only_when_a_key_is_set(monkeypatch):
    with TestClient(create_app()) as client:
        assert [p["provider"] for p in client.get("/api/providers").json()] == ["fake"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    with TestClient(create_app()) as client:
        providers = client.get("/api/providers").json()

    assert providers[1] == {"provider": "claude", "model": "claude-haiku-5-5", "isReal": True}
