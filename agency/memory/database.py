"""SQLite baglantisi ve sema yonetimi.

Neden SQLite?
    * Tek dosya, sifir kurulum, ACID.
    * Backtest/arastirma kayitlari icin yeterli performans (binlerce satir).
    * Ayni dosya hem iliskisel kayitlari hem de vektor hafiza JSON'larini tasir.

Uretimde PostgreSQL'e gecmek icin yalnizca bu moduldeki ``connect`` ve DDL
degistirilir; repository arayuzu ayni kalir.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from agency.utils import ensure_dir

SCHEMA_VERSION = 3

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    cycle INTEGER NOT NULL UNIQUE,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    duration_seconds REAL DEFAULT 0,
    symbol TEXT,
    timeframe TEXT,
    data_source TEXT,
    market_json TEXT,
    research_summary TEXT,
    candidate_count INTEGER DEFAULT 0,
    best_version TEXT,
    best_score REAL DEFAULT 0,
    promoted_json TEXT,
    archived_json TEXT,
    lessons_json TEXT,
    directives_json TEXT,
    error TEXT,
    report_path TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS research_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER NOT NULL,
    created_at TEXT,
    symbol TEXT,
    timeframe TEXT,
    data_source TEXT,
    regime TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_cycle ON research_reports(cycle);

CREATE TABLE IF NOT EXISTS news_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER,
    title TEXT,
    source TEXT,
    url TEXT,
    published_at TEXT,
    sentiment TEXT,
    keywords_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts TEXT NOT NULL,
    bar_index INTEGER,
    close REAL,
    forward_return_pct REAL,
    label TEXT,
    features_json TEXT NOT NULL,
    vector_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_snapshots_cycle ON market_snapshots(cycle);
CREATE INDEX IF NOT EXISTS idx_snapshots_label ON market_snapshots(label);

CREATE TABLE IF NOT EXISTS strategy_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    family TEXT,
    cycle INTEGER,
    parent_version TEXT,
    attempt INTEGER DEFAULT 1,
    code_path TEXT,
    code_hash TEXT,
    spec_json TEXT NOT NULL,
    fingerprint TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    score REAL DEFAULT 0,
    metrics_json TEXT,
    review_json TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_versions_score ON strategy_versions(score DESC);
CREATE INDEX IF NOT EXISTS idx_versions_cycle ON strategy_versions(cycle);

CREATE TABLE IF NOT EXISTS backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    cycle INTEGER,
    split TEXT,
    status TEXT,
    metrics_json TEXT,
    oos_json TEXT,
    errors_json TEXT,
    runtime_seconds REAL,
    report_path TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_backtests_version ON backtests(version);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    cycle INTEGER,
    verdict TEXT,
    score REAL,
    payload_json TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_reviews_version ON reviews(version);

CREATE TABLE IF NOT EXISTS memory_strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    name TEXT,
    family TEXT,
    score REAL,
    metrics_json TEXT,
    spec_json TEXT,
    code_path TEXT,
    regime TEXT,
    symbol TEXT,
    timeframe TEXT,
    promoted_cycle INTEGER,
    times_reused INTEGER DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER,
    scope TEXT,
    grade TEXT,
    message TEXT NOT NULL,
    directive TEXT,
    source_version TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_lessons_cycle ON lessons(cycle);
"""


def connect(db_path: Path | str, *, readonly: bool = False) -> sqlite3.Connection:
    """Ayarlanmis bir sqlite baglantisi dondurur."""
    path = Path(db_path)
    if not readonly:
        ensure_dir(path.parent)
    uri = f"file:{path}?mode={'ro' if readonly else 'rwc'}"
    connection = sqlite3.connect(uri, uri=True, timeout=30.0, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=15000")
    connection.execute("PRAGMA synchronous=NORMAL")
    try:
        connection.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:  # pragma: no cover - readonly baglanti
        pass
    return connection


def initialize(db_path: Path | str) -> sqlite3.Connection:
    """Semayi olusturur (idempotent) ve baglantiyi dondurur."""
    connection = connect(db_path)
    connection.executescript(SCHEMA_SQL)
    connection.execute(
        "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    connection.commit()
    return connection


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Hata durumunda geri alan islem blogu."""
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise


class Database:
    """Baglanti yasam dongusunu yoneten ince sarmalayici."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._connection: Optional[sqlite3.Connection] = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = initialize(self.path)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def table_counts(self) -> dict:
        tables = (
            "cycles",
            "research_reports",
            "news_items",
            "market_snapshots",
            "strategy_versions",
            "backtests",
            "reviews",
            "memory_strategies",
            "lessons",
        )
        counts = {}
        for table in tables:
            row = self.connection.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
            counts[table] = int(row["c"])
        return counts
