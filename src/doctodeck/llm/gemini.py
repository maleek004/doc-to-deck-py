"""Gemini over its REST API, by hand: no SDK, no retry library.

Mirrors AnkiX's `GeminiClient.cs`: schema-enforced JSON output, a retry loop for transient failures,
and defensive parsing of the response envelope. In production you'd likely reach for `tenacity` (or the
google-genai SDK); writing the loop yourself is the point of this kata, and Claude (2.3) shows the other
side: an SDK whose retries are built in.
"""

import asyncio
import logging
import os
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import BaseModel

from doctodeck.llm.contracts import LlmError, LlmErrorCode, LlmRequest, LlmResult, LlmUsage, parse_output
from doctodeck.llm.gemini_schema import to_gemini_schema

log = logging.getLogger(__name__)

# 408 request timeout, 429 rate limited, 500/502/504 server hiccups, 503 overloaded.
# Anything else (400 bad request, 401/403 bad key, 404 unknown model) would fail the same way again.
TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_SAFETY_FINISH_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"})
_JITTER_SECONDS = 0.25


@dataclass(frozen=True)
class GeminiOptions:
    api_key: str
    model: str = "gemini-3.8-flash"  # the model AnkiX uses
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    max_retries: int = 3  # retries after the first attempt: up to 4 attempts in all
    attempt_timeout: float = 30.0  # seconds one whole attempt may take
    base_delay: float = 0.5  # first backoff; doubles each retry: 0.5 s, 1 s, 2 s (+ jitter)
    max_delay: float = 10.0  # no single wait is longer, even if Retry-After asks for more
    temperature: float = 0.2  # extracting facts should be faithful and repeatable, not creative

    @classmethod
    def from_env(cls) -> "GeminiOptions | None":
        """None when GEMINI_API_KEY isn't set: no key, no "gemini" in the dropdown."""
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            return None
        model = os.environ.get("GEMINI_MODEL", "").strip()
        return cls(api_key=key, model=model) if model else cls(api_key=key)


class GeminiClient:
    provider = "gemini"
    is_real = True

    def __init__(
        self,
        options: GeminiOptions,
        http: httpx.AsyncClient,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_fraction: Callable[[], float] = random.random,
    ) -> None:
        """`http` is shared and owned by the app (opened and closed in its lifespan), so connections are
        pooled across calls. `sleep` and `random_fraction` are injected so tests can record the backoff
        instead of waiting for it."""
        self._options = options
        self._http = http
        self._sleep = sleep
        self._random = random_fraction

    @property
    def model(self) -> str:
        return self._options.model

    async def complete[T: BaseModel](self, request: LlmRequest[T]) -> LlmResult[T]:
        options = self._options
        url = f"{options.base_url}/models/{options.model}:generateContent"
        # The key goes in a header, never the URL: URLs end up in logs, proxies and error messages.
        headers = {"x-goog-api-key": options.api_key}
        body = self._build_body(request)
        started = time.perf_counter()

        for attempt in range(options.max_retries + 1):
            last = attempt == options.max_retries
            try:
                # One deadline for the whole attempt. httpx's own timeouts are per phase (connect, each
                # read...), so a server trickling bytes could otherwise hold an attempt open far longer.
                async with asyncio.timeout(options.attempt_timeout):
                    response = await self._http.post(url, json=body, headers=headers)
            except (TimeoutError, httpx.TimeoutException) as ex:
                # asyncio.timeout turns its own cancellation into TimeoutError. A cancellation from outside
                # (the client went away) stays CancelledError, which isn't caught here: it is never retried.
                if last:
                    raise LlmError(
                        LlmErrorCode.TIMEOUT,
                        f"Gemini didn't answer within {options.attempt_timeout:g} s, {attempt + 1} times.",
                    ) from ex
                await self._wait(attempt, None, "timeout")
                continue
            except httpx.TransportError as ex:  # connection refused, DNS failure, connection reset...
                if last:
                    raise LlmError(LlmErrorCode.PROVIDER_ERROR, "Couldn't reach Gemini (network error).") from ex
                await self._wait(attempt, None, "network error")
                continue

            if response.is_success:
                text, prompt_tokens, completion_tokens = _read_envelope(response)
                usage = LlmUsage(
                    provider=self.provider,
                    model=options.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=round((time.perf_counter() - started) * 1000),  # includes retries: what the user waited
                )
                return LlmResult(parse_output(text, request.output_type, self.provider), usage)

            if response.status_code in TRANSIENT_STATUSES and not last:
                await self._wait(attempt, response.headers.get("retry-after"), f"HTTP {response.status_code}")
                continue

            raise _to_error(response, attempts=attempt + 1)

        raise AssertionError("unreachable: the last attempt always returns or raises")

    async def _wait(self, attempt: int, retry_after: str | None, reason: str) -> None:
        """Exponential backoff with jitter: 0.5 s, 1 s, 2 s... plus up to 250 ms of randomness, capped.
        A server's Retry-After wins (still capped). Jitter spreads out clients that failed at the same
        moment, so they don't all come back in the same instant and knock the server over again."""
        options = self._options
        asked = _retry_after_seconds(retry_after)
        if asked is not None and asked > 0:
            delay = asked
        else:
            delay = options.base_delay * 2**attempt + self._random() * _JITTER_SECONDS
        delay = min(delay, options.max_delay)

        log.warning("Gemini %s; retry %d of %d in %d ms.", reason, attempt + 1, options.max_retries, delay * 1000)
        await self._sleep(delay)

    def _build_body(self, request: LlmRequest) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseSchema": to_gemini_schema(request.json_schema()),
            "temperature": self._options.temperature,
        }
        # Newer models "think" before answering, and thinking tokens bill at the output rate. Extraction needs
        # little of it, and unbounded thinking can push one call past the attempt timeout. The knob differs by
        # generation: Gemini 3 takes a level; Gemini 2.5 Flash takes a token budget, where 0 turns it off.
        # (2.5 Pro can't turn thinking off, so it is left at its default.)
        model = self._options.model.lower()
        if model.startswith("gemini-3"):
            generation_config["thinkingConfig"] = {"thinkingLevel": "low"}
        elif model.startswith("gemini-2.5-flash"):
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}

        return {
            "systemInstruction": {"parts": [{"text": request.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": request.user_prompt}]}],
            "generationConfig": generation_config,
        }


def _read_envelope(response: httpx.Response) -> tuple[str, int, int]:
    """Read Gemini's 200 response: candidates[0].content.parts[].text is the answer, usageMetadata the
    tokens. Safety blocks and cut-off answers become typed errors. Returns (text, prompt, completion)."""
    try:
        root = response.json()
    except ValueError as ex:
        raise LlmError(LlmErrorCode.PROVIDER_ERROR, "Gemini sent a response that isn't JSON.") from ex
    if not isinstance(root, dict):
        raise LlmError(LlmErrorCode.PROVIDER_ERROR, "Gemini sent a response of an unexpected shape.")

    # The whole prompt was blocked before any answer was written.
    block_reason = (root.get("promptFeedback") or {}).get("blockReason")
    if block_reason and block_reason != "BLOCK_REASON_UNSPECIFIED":
        raise LlmError(LlmErrorCode.REFUSED, f"Gemini blocked the request ({block_reason}).")

    candidates = root.get("candidates") or []
    if not candidates:
        raise LlmError(LlmErrorCode.PROVIDER_ERROR, "Gemini returned no answer candidates.")
    candidate = candidates[0]

    finish_reason = candidate.get("finishReason")
    if finish_reason in _SAFETY_FINISH_REASONS:
        raise LlmError(LlmErrorCode.REFUSED, f"Gemini stopped for safety reasons ({finish_reason}).")
    if finish_reason == "MAX_TOKENS":
        raise LlmError(LlmErrorCode.TRUNCATED, "Gemini's answer was cut off at its output limit. Ask for fewer cards.")

    # The answer can arrive in several parts; "thought" parts are the model's reasoning, not the answer.
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    if not text:
        raise LlmError(LlmErrorCode.INVALID_OUTPUT, "Gemini's answer had no text.")

    usage = root.get("usageMetadata") or {}
    # Thinking tokens are billed at the output price, so they count as completion tokens for costing.
    completion = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    return text, usage.get("promptTokenCount", 0), completion


def _retry_after_seconds(value: str | None) -> float | None:
    """Retry-After is either a number of seconds ("7") or an HTTP date ("Wed, 25 Nov 2026 10:00:00 GMT")."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
    except (TypeError, ValueError):
        return None  # unreadable: fall back to our own backoff


def _to_error(response: httpx.Response, attempts: int) -> LlmError:
    status = response.status_code
    # Gemini errors look like {"error": {"code": 400, "message": "...", "status": "INVALID_ARGUMENT"}}.
    try:
        detail = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        detail = response.text
    if len(detail) > 300:
        detail = detail[:300] + "…"

    if status in (401, 403):
        return LlmError(
            LlmErrorCode.PROVIDER_ERROR, f"Gemini rejected the API key (HTTP {status}). Check GEMINI_API_KEY."
        )
    # Both map to RATE_LIMITED (our 503: "try later"), but they mean different things to the person reading:
    # 429 is *our* quota (waiting may not help: check the plan's limits); 503 is Google overloaded (it passes).
    # Gemini's own message says which, e.g. "This model is currently experiencing high demand", so pass it on.
    if status == 429:
        return LlmError(
            LlmErrorCode.RATE_LIMITED, f"Gemini rate limit or quota reached (HTTP 429, {attempts} attempts): {detail}"
        )
    if status == 503:
        return LlmError(
            LlmErrorCode.RATE_LIMITED,
            f"Gemini is overloaded (HTTP 503, {attempts} attempts). Try again shortly, or set another GEMINI_MODEL. "
            f"Gemini said: {detail}",
        )
    if status in (408, 504):
        return LlmError(LlmErrorCode.TIMEOUT, f"Gemini timed out (HTTP {status}, {attempts} attempts): {detail}")
    return LlmError(LlmErrorCode.PROVIDER_ERROR, f"Gemini error (HTTP {status}): {detail}")
