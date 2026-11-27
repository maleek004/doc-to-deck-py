"""Claude over the official `anthropic` SDK. Mirrors AnkiX's `ClaudeClient.cs`.

The contrast with `gemini.py` is the point:
- Retries are the SDK's: 408, 409, 429 and 5xx (529 "overloaded" included) and connection errors, with
  exponential backoff and the server's `retry-after`. This file only sets how many and maps the final error.
- Structured output is native: the schema goes in `output_config.format`, so no dialect translation; the SDK's
  `transform_schema` adds what the API requires (e.g. `additionalProperties: false` on every object).
- Why `messages.create()` and not the SDK's `messages.parse()`: `parse()` validates the text while building the
  response, so a refusal or a `max_tokens` cut-off (half-written JSON) would raise a Pydantic error *before*
  `stop_reason` could be read, and be misreported as invalid output. Here stop_reason is checked first.
"""

import os
import time
from dataclasses import dataclass
from typing import Any

import anthropic
from pydantic import BaseModel

from doctodeck.llm.contracts import LlmError, LlmErrorCode, LlmRequest, LlmResult, LlmUsage, parse_output

# Models that accept the server-side refusal fallback (on a policy decline the API reruns the request on another
# model inside the same call). Claude Haiku 5.5, the default here as in AnkiX, has none: a refusal just stops.
FALLBACK_MODELS = frozenset(
    {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class ClaudeOptions:
    api_key: str
    model: str = "claude-haiku-5-5"  # the model AnkiX uses
    max_retries: int = 3  # the SDK's own retries (its default is 2); 3 matches the Gemini client
    timeout: float = 60.0  # seconds per attempt, the SDK's setting; retries can make the total longer
    max_tokens: int = 16_000  # room for a full card batch; non-streaming requests stay under SDK timeouts
    # Thinking can't be switched off on current Claude models; effort sets how much of it (and how many tokens)
    # a call spends. Extraction needs little: "low", as the Gemini client asks for low thinking.
    # No temperature: current models reject non-default sampling parameters with a 400.
    effort: str = "low"

    @classmethod
    def from_env(cls) -> "ClaudeOptions | None":
        """None when ANTHROPIC_API_KEY isn't set: no key, no "claude" in the dropdown."""
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            return None
        model = os.environ.get("ANTHROPIC_MODEL", "").strip()
        return cls(api_key=key, model=model) if model else cls(api_key=key)


class ClaudeClient:
    provider = "claude"
    is_real = True

    def __init__(self, options: ClaudeOptions, sdk: anthropic.AsyncAnthropic | None = None) -> None:
        """`sdk` is injected by tests (with a fake transport). Otherwise the client builds its own; the SDK keeps
        its own connection pool (on httpx2), so it can't share the app's httpx client the Gemini client uses."""
        self._options = options
        self._sdk = sdk or anthropic.AsyncAnthropic(
            api_key=options.api_key, max_retries=options.max_retries, timeout=options.timeout
        )

    @property
    def model(self) -> str:
        return self._options.model

    async def aclose(self) -> None:
        await self._sdk.close()

    async def complete[T: BaseModel](self, request: LlmRequest[T]) -> LlmResult[T]:
        options = self._options
        params: dict[str, Any] = {
            "model": options.model,
            "max_tokens": options.max_tokens,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": request.user_prompt}],
            "output_config": {
                "effort": options.effort,
                "format": {"type": "json_schema", "schema": anthropic.transform_schema(request.json_schema())},
            },
        }
        started = time.perf_counter()
        try:
            if options.model in FALLBACK_MODELS:
                # "default" lets the API pick the fallback by refusal category: no model list to maintain.
                message = await self._sdk.beta.messages.create(**params, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                message = await self._sdk.messages.create(**params)
        # Most specific first: APITimeoutError is a subclass of APIConnectionError, and every status error
        # below is an APIStatusError. All of these arrive only after the SDK's own retries ran out.
        except anthropic.APITimeoutError as ex:
            raise LlmError(LlmErrorCode.TIMEOUT, f"Claude didn't answer within {options.timeout:g} s.") from ex
        except anthropic.APIConnectionError as ex:
            raise LlmError(LlmErrorCode.PROVIDER_ERROR, "Couldn't reach Claude (network error).") from ex
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as ex:
            raise LlmError(
                LlmErrorCode.PROVIDER_ERROR,
                f"Claude rejected the API key (HTTP {ex.status_code}). Check ANTHROPIC_API_KEY.",
            ) from ex
        except anthropic.RateLimitError as ex:
            raise LlmError(LlmErrorCode.RATE_LIMITED, f"Claude rate limit reached (HTTP 429): {_detail(ex)}") from ex
        except (anthropic.OverloadedError, anthropic.ServiceUnavailableError) as ex:
            raise LlmError(
                LlmErrorCode.RATE_LIMITED,
                f"Claude is overloaded (HTTP {ex.status_code}). Try again shortly. Claude said: {_detail(ex)}",
            ) from ex
        except (anthropic.RequestTimeoutError, anthropic.GatewayTimeoutError) as ex:
            raise LlmError(LlmErrorCode.TIMEOUT, f"Claude timed out (HTTP {ex.status_code}): {_detail(ex)}") from ex
        except anthropic.APIStatusError as ex:
            raise LlmError(LlmErrorCode.PROVIDER_ERROR, f"Claude error (HTTP {ex.status_code}): {_detail(ex)}") from ex
        # asyncio.CancelledError is a BaseException, not caught above: a cancelled call stops at once.

        # Why it stopped decides what the content means, so read stop_reason before the content.
        if message.stop_reason == "refusal":
            details = message.stop_details
            category = f" ({details.category})" if details and details.category else ""
            raise LlmError(LlmErrorCode.REFUSED, f"Claude declined the request{category}.")
        if message.stop_reason == "max_tokens":
            raise LlmError(
                LlmErrorCode.TRUNCATED, "Claude's answer was cut off at its output limit. Ask for fewer cards."
            )

        # Content is a list of blocks; thinking blocks come first and aren't the answer.
        text = "".join(block.text for block in message.content if block.type == "text")
        if not text:
            raise LlmError(LlmErrorCode.INVALID_OUTPUT, "Claude's answer had no text.")

        usage = message.usage
        prompt_tokens = (
            usage.input_tokens + (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
        )
        return LlmResult(
            parse_output(text, request.output_type, self.provider),
            LlmUsage(
                provider=self.provider,
                model=message.model,  # the model that answered: with a fallback it can differ from the one asked
                prompt_tokens=prompt_tokens,
                completion_tokens=usage.output_tokens,  # includes thinking tokens, billed at the output rate
                latency_ms=round((time.perf_counter() - started) * 1000),
            ),
        )


def _detail(ex: anthropic.APIStatusError) -> str:
    """Claude's own error text. Error bodies look like {"type": "error", "error": {"type": "...", "message": "..."}};
    the SDK's ex.message is "Error code: 429 - {...whole body...}", so take the inner message when there is one."""
    body = ex.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict) and body["error"].get("message"):
        detail = str(body["error"]["message"])
    else:
        detail = ex.message
    return detail if len(detail) <= 300 else detail[:300] + "…"
