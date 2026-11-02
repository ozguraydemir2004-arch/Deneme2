"""Surum yonetimi ve arsivleme.

Her deneme puanlanir; dosya kopyasi ``runtime/archive`` altina
``vNNN_isim_scoreXX.X_verdict.py`` olarak yazilir. Boylece "hangi kod, hangi
skorla, hangi kararla arsivlendi" sorusu her zaman yanitlanabilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agency.config import Settings
from agency.memory.repositories import MemoryStore
from agency.schemas import BacktestReport, ReviewDecision, StrategyBuild
from agency.strategies.registry import StrategyRegistry


@dataclass
class ArchiveRecord:
    version: str
    score: float
    verdict: str
    archive_path: str
    net_profit_pct: float
    win_rate: float
    max_drawdown_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "score": round(self.score, 2),
            "verdict": self.verdict,
            "archive_path": self.archive_path,
            "net_profit_pct": round(self.net_profit_pct, 3),
            "win_rate": round(self.win_rate, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 3),
        }


class VersionManager:
    """Strateji surumlerinin yasam dongusu (puanlama + arsiv + siralama)."""

    def __init__(self, settings: Settings, registry: StrategyRegistry, memory: MemoryStore):
        self.settings = settings
        self.registry = registry
        self.memory = memory

    def archive(
        self, build: StrategyBuild, report: BacktestReport, decision: ReviewDecision
    ) -> ArchiveRecord:
        score = float(decision.score)
        target = self.registry.archive(build, score=score, status=decision.verdict)
        metrics = report.metrics
        return ArchiveRecord(
            version=build.version,
            score=score,
            verdict=decision.verdict,
            archive_path=str(target),
            net_profit_pct=metrics.net_profit_pct,
            win_rate=metrics.win_rate,
            max_drawdown_pct=metrics.max_drawdown_pct,
        )

    def leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        rows = self.memory.leaderboard(limit=limit)
        for row in rows:
            row["score"] = round(float(row.get("score") or 0.0), 2)
        return rows

    def best_promoted(self) -> Optional[Dict[str, Any]]:
        best = self.memory.best_strategy(status="promoted")
        if best:
            return {
                "version": best["version"],
                "name": best["name"],
                "score": round(float(best.get("score") or 0.0), 2),
                "code_path": best.get("code_path"),
                "metrics": best.get("metrics"),
            }
        return None

    def library_stats(self) -> Dict[str, Any]:
        files = self.registry.list_version_files()
        archive_files = list(Path(self.settings.versioning.archive_dir).glob("*.py"))
        return {
            "library_versions": len(files),
            "archived_versions": len(archive_files),
            "database": self.memory.stats(),
        }
