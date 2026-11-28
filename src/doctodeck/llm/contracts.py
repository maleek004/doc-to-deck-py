"""The provider-neutral LLM contract.

One structured-output call is: who the model is (system prompt), what to do (user prompt), and the
Pydantic model its answer must match. Every provider (fake, Gemini, Claude) turns that into its own
wire format and returns the *validated* model plus what the call cost. The planner and generator only
ever see this module, so swapping providers never touches pipeline code.

AnkiX does this with a base class (`StructuredLlmClientBase`) that providers inherit from. Here it is a
`Protocol`: structural typing, so a client needs no base class, only the right methods (composition
over inheritance).
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ValidationError

from doctodeck.models import ApiModel


@dataclass(frozen=True)
class LlmRequest[T: BaseModel]:
    """One structured-output call. `output_type` is both the schema sent to the model
    (`output_type.model_json_schema()`) and the validator for its answer."""

    purpose: str  # "plan" or "cards": shows up in logs, the usage ledger (Layer 4) and the fake
    system_prompt: str
    user_prompt: str
    output_type: type[T]

    def json_schema(self) -> dict:
        return self.output_type.model_json_schema()


class LlmUsage(ApiModel):
    """What the call cost. Providers bill by tokens, so every call reports them (priced in Layer 4)."""

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    # True when the answer came from the response cache (llm/cache.py). Tokens and latency are then those of the
    # call that first made the answer, so comparisons still hold; Layer 4 charges nothing for a cached call.
    cached: bool = False


@dataclass(frozen=True)
class LlmResult[T: BaseModel]:
    data: T
    usage: LlmUsage


class LlmClient(Protocol):
    """What every provider implements. Any class with these members is an LlmClient; no inheritance."""

    @property
    def provider(self) -> str:
        """Short id used in requests and the page's dropdown: "fake", "gemini", "claude"."""
        ...

    @property
    def model(self) -> str: ...

    @property
    def is_real(self) -> bool:
        """False for the fake. Layer 4 charges only for real providers."""
        ...

    async def complete[T: BaseModel](self, request: LlmRequest[T]) -> LlmResult[T]:
        """Send the request and return the validated answer, or raise LlmError.

        No cancellation token as in C#: in asyncio, cancelling the task raises CancelledError at the
        current `await`, and a client must let it propagate (never retry it, never wrap it)."""
        ...


class LlmErrorCode(StrEnum):
    """Every way an LLM call can fail, as stable codes the API maps to HTTP statuses."""

    UNKNOWN_PROVIDER = "UNKNOWN_PROVIDER"
    RATE_LIMITED = "PROVIDER_RATE_LIMITED"  # still 429 after the client's own retries
    TIMEOUT = "PROVIDER_TIMEOUT"
    REFUSED = "PROVIDER_REFUSED"  # a safety filter declined
    TRUNCATED = "PROVIDER_OUTPUT_TRUNCATED"  # hit the output token limit
    PROVIDER_ERROR = "PROVIDER_ERROR"  # 5xx, bad key, network
    INVALID_OUTPUT = "INVALID_MODEL_OUTPUT"  # not JSON, or the wrong shape
    # Guardrails (4.3): refused before anything is sent, so nothing is spent.
    INPUT_TOO_LARGE = "INPUT_TOO_LARGE"  # over the input token budget
    SPEND_CAP_REACHED = "SPEND_CAP_REACHED"  # real providers have cost the daily cap today
    UNPRICED_MODEL = "UNPRICED_MODEL"  # no price known, so the cap could not be enforced


class LlmError(Exception):
    def __init__(self, code: LlmErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


_CODE_FENCE = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL | re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    """Even with schema enforcement, models sometimes wrap JSON in ```json ... ``` fences."""
    text = text.strip()
    match = _CODE_FENCE.fullmatch(text)
    return match.group(1) if match else text


def parse_output[T: BaseModel](raw: str, output_type: type[T], provider: str) -> T:
    """Turn a model's text into the expected Pydantic model, or raise one typed error.

    The schema guarantees the *shape* (fields, types). It says nothing about whether the values make
    sense (a section index that doesn't exist); the planner and generator check that afterwards."""
    try:
        return output_type.model_validate_json(strip_code_fences(raw))
    except ValidationError as ex:  # covers both "not JSON at all" and "JSON of the wrong shape"
        raise LlmError(
            LlmErrorCode.INVALID_OUTPUT, f"{provider} returned an answer that doesn't match the expected JSON shape."
        ) from ex
