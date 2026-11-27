"""Applies to every test: no provider key from the developer's environment may leak in.

Without this, a machine with GEMINI_API_KEY exported would register the real Gemini client in every
app a test builds: the providers list would differ, and a bug could make a test call the real API.
(The C# kata's TestAppFactory blanks the same keys.)
"""

import pytest


@pytest.fixture(autouse=True)
def no_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(name, raising=False)
