"""Hafiza (Memory) katmani: tum ajanlarin ortak veri deposu.

Sorumluluklar:
    * Dongu (cycle) kayitlari ve rapor referanslari
    * Arastirma raporlari, haberler, piyasa snapshot'lari (vektorlu)
    * Strateji surumleri, backtest sonuclari, critic kararlari
    * Kalici hafiza (elite stratejiler) ve dersler/direktifler
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from agency.memory.database import Database, transaction
from agency.memory.vector_store import FeatureVectorizer
from agency.schemas import (
    BacktestReport,
    CycleResult,
    ResearchReport,
    ReviewDecision,
    StrategyBuild,
)
from agency.utils import iso_now


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _loads(payload: Optional[str], default: Any = None) -> Any:
    if not payload:
        return default
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return default


class MemoryStore:
    """SQLite uzerinde tum ajans hafizasini yoneten facade."""

    def __init__(self, database: Database, *, vectorizer: Optional[FeatureVectorizer] = None):
        self.db = database
        self.conn: sqlite3.Connection = database.connection
        self.vectorizer = vectorizer or FeatureVectorizer()

    # ------------------------------------------------------------------ util
    @staticmethod
    def _row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        return dict(row) if row is not None else None

    # ---------------------------------------------------------------- cycles
    def next_cycle_number(self) -> int:
        row = self.conn.execute("SELECT MAX(cycle) AS c FROM cycles").fetchone()
        return int(row["c"] or 0) + 1

    def save_cycle(
        self,
        result: CycleResult,
        *,
        run_id: Optional[str] = None,
        report_path: Optional[str] = None,
        symbol: str = "",
        timeframe: str = "",
    ) -> None:
        payload = result.to_dict()
        with transaction(self.conn) as conn:
            conn.execute(
                """
                INSERT INTO cycles (
                    run_id, cycle, status, started_at, finished_at, duration_seconds,
                    symbol, timeframe, data_source, market_json, research_summary,
                    candidate_count, best_version, best_score, promoted_json,
                    archived_json, lessons_json, directives_json, error, report_path
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(cycle) DO UPDATE SET
                    status=excluded.status,
                    finished_at=excluded.finished_at,
                    duration_seconds=excluded.duration_seconds,
                    data_source=excluded.data_source,
                    market_json=excluded.market_json,
                    research_summary=excluded.research_summary,
                    candidate_count=excluded.candidate_count,
                    best_version=excluded.best_version,
                    best_score=excluded.best_score,
                    promoted_json=excluded.promoted_json,
                    archived_json=excluded.archived_json,
                    lessons_json=excluded.lessons_json,
                    directives_json=excluded.directives_json,
                    error=excluded.error,
                    report_path=COALESCE(excluded.report_path, cycles.report_path)
                """,
                (
                    run_id,
                    payload["cycle"],
                    payload["status"],
                    payload["started_at"],
                    payload["finished_at"],
                    payload["duration_seconds"],
                    symbol,
                    timeframe,
                    payload["data_source"],
                    _dumps(payload["market"]),
                    payload["research_summary"],
                    payload["candidate_count"],
                    payload["best_version"],
                    payload["best_score"],
                    _dumps(payload["promoted"]),
                    _dumps(payload["archived"]),
                    _dumps(payload["lessons"]),
                    _dumps(payload["next_directives"]),
                    payload["error"],
                    report_path,
                ),
            )

    def get_cycle(self, cycle: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM cycles WHERE cycle=?", (cycle,)).fetchone()
        data = self._row(row)
        if data is None:
            return None
        data["promoted"] = _loads(data.pop("promoted_json"), [])
        data["archived"] = _loads(data.pop("archived_json"), [])
        data["lessons"] = _loads(data.pop("lessons_json"), [])
        data["next_directives"] = _loads(data.pop("directives_json"), [])
        data["market"] = _loads(data.pop("market_json"), {})
        return data

    def last_cycle(self) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT MAX(cycle) AS c FROM cycles").fetchone()
        latest = int(row["c"] or 0)
        return self.get_cycle(latest) if latest else None

    def list_cycles(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT cycle, status, started_at, finished_at, best_version, best_score, "
            "candidate_count, data_source FROM cycles ORDER BY cycle DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]


    # -------------------------------------------------------------- research
    def save_research(self, report: ResearchReport) -> int:
        payload = report.to_dict()
        with transaction(self.conn) as conn:
            cursor = conn.execute(
                "INSERT INTO research_reports "
                "(cycle, created_at, symbol, timeframe, data_source, regime, payload_json) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    report.cycle,
                    report.generated_at or iso_now(),
                    report.market.symbol,
                    report.market.timeframe,
                    report.data_source,
                    report.market.regime,
                    _dumps(payload),
                ),
            )
            return int(cursor.lastrowid or 0)

    def latest_research(self) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT payload_json FROM research_reports ORDER BY id DESC LIMIT 1"
        ).fetchone()
        data = self._row(row)
        return _loads(data.get("payload_json")) if data else None

    def save_news(self, cycle: int, items: Sequence[Mapping[str, Any]]) -> int:
        if not items:
            return 0
        with transaction(self.conn) as conn:
            for item in items:
                conn.execute(
                    "INSERT INTO news_items "
                    "(cycle, title, source, url, published_at, sentiment, keywords_json) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        cycle,
                        item.get("title", ""),
                        item.get("source", ""),
                        item.get("url", ""),
                        str(item.get("published_at", "")),
                        item.get("sentiment", "neutral"),
                        _dumps(item.get("keywords", [])),
                    ),
                )
        return len(items)

    def recent_news(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT cycle, title, source, url, published_at, sentiment "
            "FROM news_items ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------- snapshots
    def save_snapshots(
        self,
        cycle: int,
        symbol: str,
        timeframe: str,
        rows: Iterable[Mapping[str, Any]],
    ) -> int:
        payload = []
        for row in rows:
            features = row.get("features") or {}
            vector = row.get("vector")
            if vector is None:
                vector = self.vectorizer.embed_dict(features)
            payload.append(
                (
                    cycle,
                    symbol,
                    timeframe,
                    str(row.get("ts", "")),
                    int(row.get("bar_index", 0)),
                    float(row.get("close", 0.0)),
                    float(row.get("forward_return_pct", 0.0)),
                    str(row.get("label", "neutral")),
                    _dumps(features),
                    _dumps(vector),
                )
            )
        if not payload:
            return 0
        with transaction(self.conn) as conn:
            conn.executemany(
                "INSERT INTO market_snapshots "
                "(cycle, symbol, timeframe, ts, bar_index, close, forward_return_pct, "
                " label, features_json, vector_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                payload,
            )
        return len(payload)

    def similar_snapshots(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 6,
        label: Optional[str] = None,
        exclude_cycle: Optional[int] = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        query = "SELECT id, cycle, ts, close, forward_return_pct, label, features_json, vector_json " \
                "FROM market_snapshots"
        clauses: List[str] = []
        params: List[Any] = []
        if label:
            clauses.append("label=?")
            params.append(label)
        if exclude_cycle is not None:
            clauses.append("cycle<>?")
            params.append(int(exclude_cycle))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        scored: List[Dict[str, Any]] = []
        for row in self.conn.execute(query, params).fetchall():
            item = dict(row)
            vector = _loads(item.pop("vector_json"), [])
            if not vector:
                continue
            item["similarity"] = round(self.vectorizer.cosine(query_vector, vector), 6)
            item["features"] = _loads(item.pop("features_json"), {})
            scored.append(item)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[: max(1, int(top_k))]

    def snapshot_stats(self) -> Dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN label='winner' THEN 1 ELSE 0 END) AS winners, "
            "AVG(forward_return_pct) AS avg_ret "
            "FROM market_snapshots"
        ).fetchone()
        return {
            "total": int(row["total"] or 0),
            "winners": int(row["winners"] or 0),
            "avg_forward_return_pct": round(float(row["avg_ret"] or 0.0), 4),
        }


    # ------------------------------------------------------------ strategies
    def save_strategy_version(
        self,
        build: StrategyBuild,
        *,
        status: str = "draft",
        metrics: Optional[Mapping[str, Any]] = None,
        score: float = 0.0,
    ) -> None:
        now = iso_now()
        with transaction(self.conn) as conn:
            conn.execute(
                """
                INSERT INTO strategy_versions (
                    version, name, slug, family, cycle, parent_version, attempt,
                    code_path, code_hash, spec_json, fingerprint, status, score,
                    metrics_json, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(version) DO UPDATE SET
                    status=excluded.status,
                    score=excluded.score,
                    metrics_json=COALESCE(excluded.metrics_json, strategy_versions.metrics_json),
                    updated_at=excluded.updated_at
                """,
                (
                    build.version,
                    build.name,
                    build.slug,
                    build.spec.family,
                    build.cycle,
                    build.parent_version,
                    build.attempt,
                    build.module_path,
                    build.code_hash,
                    _dumps(build.spec.to_dict()),
                    build.spec.fingerprint(),
                    status,
                    float(score),
                    _dumps(metrics) if metrics else None,
                    build.created_at or now,
                    now,
                ),
            )

    def update_strategy_result(
        self,
        version: str,
        *,
        status: str,
        score: float,
        metrics: Optional[Mapping[str, Any]] = None,
        review: Optional[ReviewDecision] = None,
    ) -> None:
        with transaction(self.conn) as conn:
            conn.execute(
                "UPDATE strategy_versions SET status=?, score=?, "
                "metrics_json=COALESCE(?, metrics_json), review_json=COALESCE(?, review_json), "
                "updated_at=? WHERE version=?",
                (
                    status,
                    float(score),
                    _dumps(metrics) if metrics else None,
                    _dumps(review.to_dict()) if review else None,
                    iso_now(),
                    version,
                ),
            )

    def get_strategy(self, version: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM strategy_versions WHERE version=?", (version,)
        ).fetchone()
        data = self._row(row)
        if data is None:
            return None
        data["spec"] = _loads(data.pop("spec_json"), {})
        data["metrics"] = _loads(data.pop("metrics_json"), {})
        data["review"] = _loads(data.pop("review_json"), {})
        return data

    def list_strategies(
        self,
        limit: int = 20,
        *,
        status: Optional[str] = None,
        order_by: str = "score",
    ) -> List[Dict[str, Any]]:
        column = order_by if order_by in {"score", "created_at", "cycle"} else "score"
        query = (
            "SELECT version, name, family, cycle, status, score, code_path, created_at "
            "FROM strategy_versions"
        )
        params: List[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += f" ORDER BY {column} DESC, id DESC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    def best_strategy(self, *, status: Optional[str] = "promoted") -> Optional[Dict[str, Any]]:
        query = "SELECT version FROM strategy_versions"
        params: List[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY score DESC, id DESC LIMIT 1"
        row = self.conn.execute(query, params).fetchone()
        return self.get_strategy(row["version"]) if row else None

    def count_versions(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM strategy_versions").fetchone()
        return int(row["c"])

    # ------------------------------------------------------------- backtests
    def save_backtest(
        self,
        report: BacktestReport,
        *,
        cycle: int,
        report_path: Optional[str] = None,
    ) -> None:
        with transaction(self.conn) as conn:
            conn.execute(
                "INSERT INTO backtests (version, cycle, split, status, metrics_json, "
                "oos_json, errors_json, runtime_seconds, report_path, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    report.strategy_version,
                    cycle,
                    report.split,
                    report.status,
                    _dumps(report.metrics.to_dict()),
                    _dumps(report.oos_metrics.to_dict()) if report.oos_metrics else None,
                    _dumps(report.errors.to_dict()),
                    float(report.runtime_seconds),
                    report_path,
                    report.created_at or iso_now(),
                ),
            )

    def backtest_history(self, version: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT cycle, split, status, metrics_json, runtime_seconds, created_at "
            "FROM backtests WHERE version=? ORDER BY id ASC",
            (version,),
        ).fetchall()
        history = []
        for row in rows:
            item = dict(row)
            item["metrics"] = _loads(item.pop("metrics_json"), {})
            history.append(item)
        return history


    # --------------------------------------------------------------- reviews
    def save_review(self, decision: ReviewDecision, *, cycle: int) -> None:
        with transaction(self.conn) as conn:
            conn.execute(
                "INSERT INTO reviews (version, cycle, verdict, score, payload_json, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    decision.strategy_version,
                    int(cycle),
                    decision.verdict,
                    float(decision.score),
                    _dumps(decision.to_dict()),
                    decision.created_at or iso_now(),
                ),
            )

    def review_history(self, version: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT cycle, verdict, score, payload_json, created_at FROM reviews "
            "WHERE version=? ORDER BY id ASC",
            (version,),
        ).fetchall()
        history = []
        for row in rows:
            item = dict(row)
            payload = _loads(item.pop("payload_json"), {})
            item["directives"] = payload.get("directives", [])
            history.append(item)
        return history

    # --------------------------------------------------------------- lessons
    def add_lesson(
        self,
        cycle: int,
        message: str,
        *,
        directive: Optional[str] = None,
        scope: str = "strategy",
        grade: str = "info",
        source_version: Optional[str] = None,
    ) -> None:
        with transaction(self.conn) as conn:
            conn.execute(
                "INSERT INTO lessons (cycle, scope, grade, message, directive, source_version, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (int(cycle), scope, grade, str(message), directive, source_version, iso_now()),
            )

    def recent_lessons(self, limit: int = 12) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT cycle, scope, grade, message, directive, source_version, created_at "
            "FROM lessons ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    def pending_directives(self, limit: int = 8) -> List[str]:
        rows = self.conn.execute(
            "SELECT directive FROM lessons WHERE directive IS NOT NULL AND directive <> '' "
            "ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [str(r["directive"]) for r in rows]


    # ---------------------------------------------------------------- memory
    def promote_strategy(
        self,
        version: str,
        *,
        cycle: int,
        regime: str,
        score: float,
        symbol: str,
        timeframe: str,
    ) -> None:
        record = self.get_strategy(version)
        if record is None:
            raise KeyError(f"Strateji bulunamadi: {version}")
        with transaction(self.conn) as conn:
            conn.execute(
                """
                INSERT INTO memory_strategies (
                    version, name, family, score, metrics_json, spec_json, code_path,
                    regime, symbol, timeframe, promoted_cycle, times_reused, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?)
                ON CONFLICT(version) DO UPDATE SET
                    score=excluded.score,
                    metrics_json=excluded.metrics_json,
                    regime=excluded.regime,
                    promoted_cycle=excluded.promoted_cycle
                """,
                (
                    version,
                    record.get("name"),
                    record.get("family"),
                    float(score),
                    _dumps(record.get("metrics") or {}),
                    _dumps(record.get("spec") or {}),
                    record.get("code_path"),
                    regime,
                    symbol,
                    timeframe,
                    int(cycle),
                    iso_now(),
                ),
            )

    def memory_strategies(
        self,
        *,
        limit: int = 5,
        family: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM memory_strategies"
        clauses: List[str] = []
        params: List[Any] = []
        if family:
            clauses.append("family=?")
            params.append(family)
        if regime:
            clauses.append("(regime=? OR regime IS NULL)")
            params.append(regime)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY score DESC, id DESC LIMIT ?"
        params.append(int(limit))
        rows = self.conn.execute(query, params).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["metrics"] = _loads(item.pop("metrics_json"), {})
            item["spec"] = _loads(item.pop("spec_json"), {})
            out.append(item)
        return out

    def mark_memory_reused(self, version: str) -> None:
        with transaction(self.conn) as conn:
            conn.execute(
                "UPDATE memory_strategies SET times_reused=times_reused+1 WHERE version=?",
                (version,),
            )

    # ----------------------------------------------------------------- stats
    def stats(self) -> Dict[str, Any]:
        versions = self.conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN status='promoted' THEN 1 ELSE 0 END) AS promoted, "
            "SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) AS rejected, "
            "MAX(score) AS best_score, AVG(score) AS avg_score "
            "FROM strategy_versions"
        ).fetchone()
        snapshots = self.snapshot_stats()
        memory_count = self.conn.execute("SELECT COUNT(*) AS c FROM memory_strategies").fetchone()
        lesson_count = self.conn.execute("SELECT COUNT(*) AS c FROM lessons").fetchone()
        cycle_count = self.conn.execute("SELECT COUNT(*) AS c FROM cycles").fetchone()
        backtests = self.conn.execute("SELECT COUNT(*) AS c FROM backtests").fetchone()
        return {
            "cycles": int(cycle_count["c"] or 0),
            "versions": int(versions["total"] or 0),
            "promoted": int(versions["promoted"] or 0),
            "rejected": int(versions["rejected"] or 0),
            "best_score": round(float(versions["best_score"] or 0.0), 2),
            "avg_score": round(float(versions["avg_score"] or 0.0), 2),
            "snapshots": snapshots["total"],
            "winner_snapshots": snapshots["winners"],
            "memory_strategies": int(memory_count["c"] or 0),
            "lessons": int(lesson_count["c"] or 0),
            "backtests": int(backtests["c"] or 0),
        }

    def leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT version, name, family, status, score, cycle, code_path "
            "FROM strategy_versions ORDER BY score DESC, id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.db.close()




