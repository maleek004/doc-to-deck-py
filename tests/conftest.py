"""Applies to every test: no provider key from the developer's environment may leak in.

Without this, a machine with GEMINI_API_KEY exported would register the real Gemini client in every
app a test builds: the providers list would differ, and a bug could make a test call the real API.
(The C# kata's TestAppFactory blanks the same keys.) LLM_CACHE too: a test must never read or write the
developer's real cache file.
"""

import pytest


@pytest.fixture(autouse=True)
def no_provider_keys(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "LLM_CACHE"):
        monkeypatch.delenv(name, raising=False)
    # Every app a test builds records into its own throwaway ledger, never data/ledger.sqlite3.
    monkeypatch.setenv("LEDGER_DB", str(tmp_path / "ledger.sqlite3"))
    # And its own wallet, opened with the default balance, never data/wallet.sqlite3.
    monkeypatch.setenv("WALLET_DB", str(tmp_path / "wallet.sqlite3"))
    monkeypatch.delenv("WALLET_START_CREDITS", raising=False)
