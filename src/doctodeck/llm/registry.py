"""Picks a provider per request by name (AnkiX: `AiModuleExtensions` registers the clients)."""

from collections.abc import Iterable

import httpx

from doctodeck.llm.claude import ClaudeClient, ClaudeOptions
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

    async def aclose(self) -> None:
        """Close what clients own (the Claude SDK's connection pool). Called once, at app shutdown."""
        for client in self._by_provider.values():
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()


def build_registry(http: httpx.AsyncClient) -> LlmRegistry:
    """The app's providers: the fake always; Gemini and Claude only when their keys are set.
    A provider with no key is simply absent, so the app still works end to end on the fake."""
    clients: list[LlmClient] = [FakeLlmClient()]
    if gemini := GeminiOptions.from_env():
        clients.append(GeminiClient(gemini, http))
    if claude := ClaudeOptions.from_env():
        clients.append(ClaudeClient(claude))
    return LlmRegistry(clients)
