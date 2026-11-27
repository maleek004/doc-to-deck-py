"""Picks a provider per request by name (AnkiX: `AiModuleExtensions` registers the clients)."""

from collections.abc import Iterable

import httpx

from doctodeck.llm.contracts import LlmClient, LlmError, LlmErrorCode
from doctodeck.llm.fake import FakeLlmClient
from doctodeck.llm.gemini import GeminiClient, GeminiOptions


class LlmRegistry:
    def __init__(self, clients: Iterable[LlmClient]) -> None:
        self._by_provider = {client.provider.lower(): client for client in clients}

    @property
    def all(self) -> list[LlmClient]:
        return list(self._by_provider.values())

    def get(self, provider: str | None) -> LlmClient:
        client = self._by_provider.get((provider or "").strip().lower())
        if client is None:
            known = ", ".join(self._by_provider)
            raise LlmError(LlmErrorCode.UNKNOWN_PROVIDER, f"Unknown provider '{provider}'. Available: {known}.")
        return client


def build_registry(http: httpx.AsyncClient) -> LlmRegistry:
    """The app's providers: the fake always; Gemini only when GEMINI_API_KEY is set (Claude joins in 2.3).
    A provider with no key is simply absent, so the app still works end to end on the fake."""
    clients: list[LlmClient] = [FakeLlmClient()]
    if gemini := GeminiOptions.from_env():
        clients.append(GeminiClient(gemini, http))
    return LlmRegistry(clients)
