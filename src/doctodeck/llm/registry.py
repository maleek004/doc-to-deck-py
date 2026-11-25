"""Picks a provider per request by name (AnkiX: `AiModuleExtensions` registers the clients)."""

from collections.abc import Iterable

from doctodeck.llm.contracts import LlmClient, LlmError, LlmErrorCode
from doctodeck.llm.fake import FakeLlmClient


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


def build_registry() -> LlmRegistry:
    """The app's providers. 2.1: only the fake. 2.2 and 2.3 add Gemini and Claude when their keys are set."""
    return LlmRegistry([FakeLlmClient()])
