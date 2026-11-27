"""A response cache in front of any real LLM client, so the same request is paid for once.

Why: the Gemini free tier allows 20 requests per model per day, and evals (Layer 3) rerun the same documents
over and over. With the cache, a rerun of an unchanged prompt costs nothing and returns at once, so a change
in the numbers means the prompt or the model changed, not the dice.

How it works (interview version):
- The key is a SHA-256 of everything that decides the answer: provider, model, the client's generation settings
  (temperature, thinking, effort), both prompts and the output JSON Schema, plus a version number. Change any one
  and it is a different key, so a stale answer can't come back. Bump CACHE_VERSION to invalidate everything.
- Only successful, validated answers are stored. An error (429, timeout, refusal) is never cached: the next try
  should really try again.
- A hit returns the usage recorded when the answer was made, marked `cached: true`, so a comparison still shows
  what the model cost and how long it took; Layer 4 charges nothing for a cached call.
- Storage is stdlib `sqlite3` (one file, no server). sqlite3 calls block, so they run in a worker thread
  (`asyncio.to_thread`) and never stall the event loop that serves other requests.

It is a wrapper (the decorator pattern): `CachingLlmClient` is itself an `LlmClient`, so the planner, generator
and registry don't know it exists. AnkiX has no equivalent.
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from doctodeck.llm.contracts import LlmClient, LlmRequest, LlmResult, LlmUsage

log = logging.getLogger(__name__)

# Bump when the stored format, or anything the key leaves out, changes meaning: every old entry becomes a miss.
CACHE_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key        TEXT PRIMARY KEY,
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL,
    purpose    TEXT NOT NULL,
    data       TEXT NOT NULL,  -- the validated answer, as JSON
    usage      TEXT NOT NULL,  -- LlmUsage of the call that made it, as JSON
    created_at TEXT NOT NULL
)
"""


def cache_key(client: LlmClient, request: LlmRequest) -> str:
    """SHA-256 over a canonical JSON of everything that decides the answer.

    `sort_keys` and fixed separators make the JSON canonical: the same inputs always give the same bytes, whatever
    order a dict was built in. Hashing keeps the key short and fixed-length, whatever the prompt size."""
    settings: Mapping[str, Any] = getattr(client, "settings", None) or {}
    material = {
        "version": CACHE_VERSION,
        "provider": client.provider,
        "model": client.model,
        "settings": dict(settings),
        "system": request.system_prompt,
        "user": request.user_prompt,
        "schema": request.json_schema(),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """A SQLite file of answers by key. Each call opens its own short-lived connection inside a worker thread:
    sqlite3 connections shouldn't be shared across threads, and opening one on a local file takes microseconds."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer, and the writer doesn't block them
            db.execute(_SCHEMA)

    @classmethod
    def from_env(cls) -> "ResponseCache | None":
        """LLM_CACHE is the file's path, e.g. data/llm-cache.sqlite3. Unset or empty: no cache."""
        path = os.environ.get("LLM_CACHE", "").strip()
        return cls(path) if path else None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    async def get(self, key: str) -> tuple[str, str] | None:
        """(data JSON, usage JSON) or None."""

        def read() -> tuple[str, str] | None:
            with self._connect() as db:
                return db.execute("SELECT data, usage FROM responses WHERE key = ?", (key,)).fetchone()

        return await asyncio.to_thread(read)

    async def put(self, key: str, request: LlmRequest, result: LlmResult) -> None:
        data = result.data.model_dump_json(by_alias=True)
        usage = result.usage.model_dump_json(by_alias=True)
        created = datetime.now(UTC).isoformat(timespec="seconds")

        def write() -> None:
            # `with` on a connection commits on success and rolls back on an exception (it doesn't close it).
            with self._connect() as db:
                db.execute(
                    "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (key, result.usage.provider, result.usage.model, request.purpose, data, usage, created),
                )

        await asyncio.to_thread(write)

    async def delete(self, key: str) -> None:
        def remove() -> None:
            with self._connect() as db:
                db.execute("DELETE FROM responses WHERE key = ?", (key,))

        await asyncio.to_thread(remove)

    async def count(self) -> int:
        def counted() -> int:
            with self._connect() as db:
                return db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

        return await asyncio.to_thread(counted)


class CachingLlmClient:
    """An LlmClient that answers from the cache when it can and asks `inner` when it can't."""

    def __init__(self, inner: LlmClient, cache: ResponseCache) -> None:
        self._inner = inner
        self._cache = cache

    # The same identity as the wrapped client: the dropdown, the usage and Layer 4 see the real provider.
    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def is_real(self) -> bool:
        return self._inner.is_real

    @property
    def inner(self) -> LlmClient:
        return self._inner

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()

    async def complete[T: BaseModel](self, request: LlmRequest[T]) -> LlmResult[T]:
        key = cache_key(self._inner, request)
        started = time.perf_counter()

        hit = await self._cache.get(key)
        if hit is not None:
            data_json, usage_json = hit
            try:
                data = request.output_type.model_validate_json(data_json)
                usage = LlmUsage.model_validate_json(usage_json)
            except ValidationError:
                # The schema is part of the key, so this "can't happen" unless the file was edited or a model's
                # validators changed without its schema changing. Don't fail the user's request over it: drop the
                # entry and ask the model.
                log.warning("Dropping an unreadable cache entry %s… and calling %s.", key[:12], self.provider)
                await self._cache.delete(key)
            else:
                log.info(
                    "Cache hit for %s %s (%s…) in %.1f ms.",
                    self.provider,
                    request.purpose,
                    key[:12],
                    (time.perf_counter() - started) * 1000,
                )
                return LlmResult(data, usage.model_copy(update={"cached": True}))

        # A miss. Errors propagate and are never stored: the next attempt should really try again.
        # (Two identical requests in flight at once both miss and both call the model; the second write just
        # replaces the first. Collapsing them into one call, "single-flight", is a possible refinement.)
        result = await self._inner.complete(request)
        await self._cache.put(key, request, result)
        return result
