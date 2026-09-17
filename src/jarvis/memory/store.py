"""Durable memory: a SQLite file holding the chat log, learned facts and usage.

SQLite is the right default here -- one file, no server, survives restarts, and
it is what the mature companion projects use for "continue where we left off".
Semantic recall is layered on top as an embedding column rather than a second
datastore, so there is still only one file to back up.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .embeddings import Embedder, build_embedder, cosine, pack, unpack

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session   TEXT NOT NULL,
    ts        REAL NOT NULL,
    role      TEXT NOT NULL,
    content   TEXT NOT NULL,
    meta      TEXT NOT NULL DEFAULT '{}',
    embedder  TEXT,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_turns_session_ts ON turns(session, ts);

CREATE TABLE IF NOT EXISTS facts (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'user',
    confidence REAL NOT NULL DEFAULT 1.0,
    ts         REAL NOT NULL,
    embedder   TEXT,
    embedding  BLOB
);

CREATE TABLE IF NOT EXISTS observations (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    summary TEXT NOT NULL,
    app     TEXT,
    spoken  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_observations_ts ON observations(ts);

CREATE TABLE IF NOT EXISTS usage (
    day               TEXT NOT NULL,
    model             TEXT NOT NULL,
    calls             INTEGER NOT NULL DEFAULT 0,
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL NOT NULL DEFAULT 0.0,
    cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, model)
);
"""


@dataclass(frozen=True)
class Turn:
    id: int
    session: str
    ts: float
    role: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.ts, tz=UTC)


@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    source: str
    confidence: float
    ts: float


@dataclass(frozen=True)
class Recalled:
    score: float
    kind: str  # "turn" | "fact"
    text: str
    ts: float


def _today() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


class MemoryStore:
    """Thread-safe SQLite-backed memory. Use as a context manager or call ``close()``."""

    def __init__(
        self,
        path: str | Path = "data/jarvis.sqlite3",
        session: str = "default",
        embedder: Embedder | str = "hashing",
        embedding_dim: int = 256,
    ) -> None:
        self.path = Path(path)
        self.session = session
        self.embedder: Embedder = (
            build_embedder(embedder, embedding_dim) if isinstance(embedder, str) else embedder
        )
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._db.executescript(SCHEMA)
            self._migrate()
            self._db.commit()

    # Columns added after the first release. SQLite has no "ADD COLUMN IF NOT
    # EXISTS", so each one is applied only when the table lacks it -- an
    # existing jarvis.sqlite3 must keep working across an upgrade.
    MIGRATIONS: tuple[tuple[str, str, str], ...] = (
        ("usage", "cache_read_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ("usage", "cache_write_tokens", "INTEGER NOT NULL DEFAULT 0"),
    )

    def _migrate(self) -> None:
        for table, column, declaration in self.MIGRATIONS:
            existing = {
                row["name"] for row in self._db.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in existing:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._db.commit()
            self._db.close()

    # -- chat log ----------------------------------------------------------

    def add_turn(self, role: str, content: str, **meta: Any) -> Turn:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"unknown role: {role}")
        ts = time.time()
        vector = self.embedder.embed(content) if role != "system" else []
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO turns (session, ts, role, content, meta, embedder, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session,
                    ts,
                    role,
                    content,
                    json.dumps(meta),
                    self.embedder.name if vector else None,
                    pack(vector) if vector else None,
                ),
            )
            self._db.commit()
            turn_id = int(cursor.lastrowid or 0)
        return Turn(turn_id, self.session, ts, role, content, meta)

    def recent_turns(self, limit: int = 20, session: str | None = None) -> list[Turn]:
        """The last ``limit`` turns, oldest first, ready to feed straight to the model."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM turns WHERE session = ? ORDER BY ts DESC, id DESC LIMIT ?",
                (session or self.session, max(0, limit)),
            ).fetchall()
        return [self._row_to_turn(row) for row in reversed(rows)]

    def turn_count(self, session: str | None = None) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM turns WHERE session = ?",
                (session or self.session,),
            ).fetchone()
        return int(row["n"])

    def sessions(self) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT session, MAX(ts) AS last FROM turns GROUP BY session ORDER BY last DESC"
            ).fetchall()
        return [row["session"] for row in rows]

    # -- facts -------------------------------------------------------------

    def remember(
        self, key: str, value: str, source: str = "user", confidence: float = 1.0
    ) -> Fact:
        """Upsert a durable fact, e.g. ``remember("name", "Daigo")``."""
        key = key.strip().lower()
        if not key:
            raise ValueError("fact key cannot be empty")
        ts = time.time()
        vector = self.embedder.embed(f"{key}: {value}")
        with self._lock:
            self._db.execute(
                "INSERT INTO facts (key, value, source, confidence, ts, embedder, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value, source=excluded.source,"
                " confidence=excluded.confidence, ts=excluded.ts, embedder=excluded.embedder,"
                " embedding=excluded.embedding",
                (
                    key,
                    value,
                    source,
                    confidence,
                    ts,
                    self.embedder.name if vector else None,
                    pack(vector) if vector else None,
                ),
            )
            self._db.commit()
        return Fact(key, value, source, confidence, ts)

    def forget(self, key: str) -> bool:
        with self._lock:
            cursor = self._db.execute("DELETE FROM facts WHERE key = ?", (key.strip().lower(),))
            self._db.commit()
        return cursor.rowcount > 0

    def facts(self) -> list[Fact]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM facts ORDER BY key").fetchall()
        return [
            Fact(r["key"], r["value"], r["source"], float(r["confidence"]), float(r["ts"]))
            for r in rows
        ]

    def get_fact(self, key: str) -> Fact | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM facts WHERE key = ?", (key.strip().lower(),)
            ).fetchone()
        if row is None:
            return None
        return Fact(
            row["key"], row["value"], row["source"], float(row["confidence"]), float(row["ts"])
        )

    # -- recall ------------------------------------------------------------

    def recall(self, query: str, limit: int = 5, min_score: float = 0.15) -> list[Recalled]:
        """Hybrid recall: cosine over stored vectors, with a keyword fallback.

        Returns facts and past turns interleaved, best first. Facts get a small
        boost because an explicitly remembered preference beats an offhand line
        from three weeks ago.
        """
        query = query.strip()
        if not query:
            return []
        vector = self.embedder.embed(query)
        results: list[Recalled] = []

        with self._lock:
            fact_rows = self._db.execute("SELECT * FROM facts").fetchall()
            turn_rows = self._db.execute(
                "SELECT * FROM turns WHERE session = ? AND role != 'system'"
                " ORDER BY ts DESC LIMIT 500",
                (self.session,),
            ).fetchall()

        for row in fact_rows:
            text = f"{row['key']}: {row['value']}"
            score = self._score(vector, row["embedding"], query, text)
            if score >= min_score:
                results.append(Recalled(min(1.0, score * 1.1), "fact", text, float(row["ts"])))

        for row in turn_rows:
            score = self._score(vector, row["embedding"], query, row["content"])
            if score >= min_score:
                results.append(
                    Recalled(score, "turn", f"{row['role']}: {row['content']}", float(row["ts"]))
                )

        results.sort(key=lambda item: (item.score, item.ts), reverse=True)
        return results[:limit]

    def _score(self, vector: list[float], blob: bytes | None, query: str, text: str) -> float:
        if vector and blob:
            return cosine(vector, unpack(blob))
        return _keyword_overlap(query, text)

    # -- observations and usage -------------------------------------------

    def record_observation(
        self, summary: str, app: str | None = None, spoken: bool = False
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO observations (ts, summary, app, spoken) VALUES (?, ?, ?, ?)",
                (time.time(), summary, app, int(spoken)),
            )
            self._db.commit()

    def recent_observations(self, limit: int = 10) -> list[tuple[float, str, str | None, bool]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, summary, app, spoken FROM observations ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(float(r["ts"]), r["summary"], r["app"], bool(r["spoken"])) for r in rows]

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO usage (day, model, calls, input_tokens, output_tokens, cost_usd,"
                " cache_read_tokens, cache_write_tokens)"
                " VALUES (?, ?, 1, ?, ?, ?, ?, ?)"
                " ON CONFLICT(day, model) DO UPDATE SET calls = calls + 1,"
                " input_tokens = input_tokens + excluded.input_tokens,"
                " output_tokens = output_tokens + excluded.output_tokens,"
                " cost_usd = cost_usd + excluded.cost_usd,"
                " cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,"
                " cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens",
                (
                    _today(),
                    model,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    cache_read_tokens,
                    cache_write_tokens,
                ),
            )
            self._db.commit()

    def cache_stats(self) -> dict[str, int]:
        """Today's cache reads and writes. Zero reads across many calls means
        something is invalidating the prefix -- that is the signal to look for."""
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(cache_read_tokens), 0) AS reads,"
                " COALESCE(SUM(cache_write_tokens), 0) AS writes,"
                " COALESCE(SUM(calls), 0) AS calls FROM usage WHERE day = ?",
                (_today(),),
            ).fetchone()
        return {
            "reads": int(row["reads"]),
            "writes": int(row["writes"]),
            "calls": int(row["calls"]),
        }

    def spend_today(self, model: str | None = None) -> float:
        with self._lock:
            if model:
                row = self._db.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM usage"
                    " WHERE day = ? AND model = ?",
                    (_today(), model),
                ).fetchone()
            else:
                row = self._db.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM usage WHERE day = ?",
                    (_today(),),
                ).fetchone()
        return float(row["total"])

    def usage_summary(self, days: int = 7) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT day, model, calls, input_tokens, output_tokens, cost_usd,"
                " cache_read_tokens, cache_write_tokens FROM usage"
                " GROUP BY day, model ORDER BY day DESC LIMIT ?",
                (days * 8,),
            ).fetchall()
        return [dict(row) for row in rows]

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _row_to_turn(row: sqlite3.Row) -> Turn:
        return Turn(
            int(row["id"]),
            row["session"],
            float(row["ts"]),
            row["role"],
            row["content"],
            json.loads(row["meta"] or "{}"),
        )


def _keyword_overlap(query: str, text: str) -> float:
    from .embeddings import tokenize  # local import keeps the module import graph flat

    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def format_context(items: Sequence[Recalled] | Iterable[Recalled]) -> str:
    """Render recalled items as a compact block for the system prompt."""
    lines = [f"- ({item.kind}) {item.text}" for item in items]
    return "\n".join(lines)
