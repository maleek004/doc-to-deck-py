"""The credit wallet (4.2): one local balance, charged atomically. Anchor: AnkiX `AiCreditService.cs`.

The one concurrency story of the kata. Read-then-write ("balance is 30, cost is 6, so write 24") loses money
when two requests interleave: both read 30, both write 24, one charge vanishes, and with enough requests the
wallet pays for more than it holds. The fix is to let the database do check and write in one statement:

    UPDATE wallet SET balance = balance - :cost WHERE id = 1 AND balance >= :cost

SQLite runs one write at a time, so the condition is checked against the balance as it is at that moment. If
it no longer covers the cost, no row matches and nothing changes: the caller is told, and the balance can't go
negative. `CHECK (balance >= 0)` on the column is the backstop if a future query forgets the WHERE.

Every change is a row in `wallet_transactions`, written in the same transaction as the balance, so the history
always adds up to the balance. Top-ups are idempotent on their reference: a retried request can't pay twice.
"""

import asyncio
import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from doctodeck.models import ApiModel

DEFAULT_WALLET = Path("data/wallet.sqlite3")
DEFAULT_START_CREDITS = 100
MAX_TOP_UP = 10_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS wallet (
    id      INTEGER PRIMARY KEY CHECK (id = 1),     -- a single wallet
    balance INTEGER NOT NULL CHECK (balance >= 0)   -- the database itself refuses a negative balance
);
CREATE TABLE IF NOT EXISTS wallet_transactions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    at            TEXT    NOT NULL,
    action        TEXT    NOT NULL,  -- opening, top_up, generate
    amount        INTEGER NOT NULL,  -- + credit, - charge
    balance_after INTEGER NOT NULL,
    reference     TEXT,
    deck_id       TEXT               -- joins to the usage ledger's deck
);
-- A top-up reference can be used once: the retry of a top-up fails this index instead of paying twice.
CREATE UNIQUE INDEX IF NOT EXISTS wallet_top_up_once ON wallet_transactions (reference) WHERE action = 'top_up';
"""


class WalletTransaction(ApiModel):
    id: int
    at: str
    action: str
    amount: int
    balance_after: int
    reference: str | None
    deck_id: str | None


class TopUpResult(ApiModel):
    applied: bool  # False: this reference was already used, nothing added this time
    balance: int


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Wallet:
    def __init__(self, path: str | Path, start_credits: int = DEFAULT_START_CREDITS) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)
            # First run only: open the wallet with the starting balance, recorded like any other change.
            if db.execute("INSERT OR IGNORE INTO wallet (id, balance) VALUES (1, ?)", (start_credits,)).rowcount:
                db.execute(
                    "INSERT INTO wallet_transactions (at, action, amount, balance_after) VALUES (?, 'opening', ?, ?)",
                    (_now(), start_credits, start_credits),
                )

    @classmethod
    def from_env(cls) -> "Wallet":
        """WALLET_DB (default data/wallet.sqlite3); WALLET_START_CREDITS is the balance of a new wallet only."""
        start = int(os.environ.get("WALLET_START_CREDITS", "").strip() or DEFAULT_START_CREDITS)
        return cls(os.environ.get("WALLET_DB", "").strip() or DEFAULT_WALLET, start)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """One connection per operation (they're cheap, and a connection can't be shared across threads).
        `with db:` commits on success and rolls back on an exception; `closing` then closes the connection."""
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.row_factory = sqlite3.Row
            yield db

    async def balance(self) -> int:
        def read() -> int:
            with self._transaction() as db:
                return db.execute("SELECT balance FROM wallet WHERE id = 1").fetchone()[0]

        return await asyncio.to_thread(read)

    async def charge(
        self, amount: int, action: str, reference: str | None = None, deck_id: str | None = None
    ) -> int | None:
        """Take `amount` credits if the balance covers it. Returns the new balance, or None if it didn't (and then
        nothing changed). Check and write are one UPDATE: see the module docstring."""
        if amount <= 0:
            raise ValueError(f"A charge must be positive, got {amount}.")

        def run() -> int | None:
            with self._transaction() as db:
                row = db.execute(
                    "UPDATE wallet SET balance = balance - ? WHERE id = 1 AND balance >= ? RETURNING balance",
                    (amount, amount),
                ).fetchone()
                if row is None:
                    return None  # not enough credits: no row matched, nothing written
                db.execute(
                    "INSERT INTO wallet_transactions (at, action, amount, balance_after, reference, deck_id)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (_now(), action, -amount, row[0], reference, deck_id),
                )
                return row[0]

        return await asyncio.to_thread(run)

    async def top_up(self, credits: int, reference: str) -> TopUpResult:
        """Add credits once per `reference`. The transaction row is inserted first, so the unique index claims the
        reference before any credit is added: two requests with the same reference can't both get through."""
        if not 0 < credits <= MAX_TOP_UP:
            raise ValueError(f"A top-up must be between 1 and {MAX_TOP_UP} credits, got {credits}.")
        if not reference.strip():
            raise ValueError("A top-up needs a reference, so a retry can't pay twice.")

        def run() -> TopUpResult:
            with self._transaction() as db:
                try:
                    row_id = db.execute(
                        "INSERT INTO wallet_transactions (at, action, amount, balance_after, reference)"
                        " VALUES (?, 'top_up', ?, 0, ?)",
                        (_now(), credits, reference),
                    ).lastrowid
                except sqlite3.IntegrityError:
                    balance = db.execute("SELECT balance FROM wallet WHERE id = 1").fetchone()[0]
                    return TopUpResult(applied=False, balance=balance)
                balance = db.execute(
                    "UPDATE wallet SET balance = balance + ? WHERE id = 1 RETURNING balance", (credits,)
                ).fetchone()[0]
                db.execute("UPDATE wallet_transactions SET balance_after = ? WHERE id = ?", (balance, row_id))
                return TopUpResult(applied=True, balance=balance)

        return await asyncio.to_thread(run)

    async def transactions(self, limit: int = 20) -> list[WalletTransaction]:
        def read() -> list[WalletTransaction]:
            with self._transaction() as db:
                rows = db.execute(
                    "SELECT id, at, action, amount, balance_after, reference, deck_id FROM wallet_transactions"
                    " ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [WalletTransaction.model_validate(dict(row)) for row in rows]

        return await asyncio.to_thread(read)
