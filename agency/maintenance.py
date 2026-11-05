"""Terfi eden stratejilerin periyodik yeniden dogrulanmasi (decay check).

Kalici hafizaya alinan bir strateji, piyasa rejimi degistiginde performansini
kaybedebilir. Bu modul:

    1. Promoted stratejileri **guncel veri** uzerinde yeniden kosar,
    2. Ayni gate tablosuyla degerlendirir,
    3. Basarisiz olanlari (istege bagli) **demote** eder: kalici hafizadan
       cikarir, `status='demoted'` isaretler ve ders kaydi birakir,
    4. `runtime/reports/REVALIDATION.md` raporunu uretir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agency.backtest.engine import BacktestEngine, EngineConfig
from agency.backtest.metrics import evaluate_gates, score_metrics
from agency.config import Settings
from agency.data.market_data import MarketDataProvider
from agency.exceptions import AgencyError
from agency.memory.database import Database
from agency.memory.repositories import MemoryStore
from agency.schemas import BacktestMetrics
from agency.strategies.registry import StrategyRegistry
from agency.utils import iso_now, write_text_atomic

CORE_GATES = ("net_profit", "win_rate", "drawdown", "sharpe", "profit_factor", "trades")


@dataclass
class RevalidationResult:
    """Tek bir stratejinin yeniden dogrulama sonucu."""

    version: str
    name: str
    ok: bool = False
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    gates: Dict[str, bool] = field(default_factory=dict)
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    demoted: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "ok": self.ok,
            "metrics": self.metrics.to_dict(),
            "gates": self.gates,
            "score": round(self.score, 2),
            "reasons": self.reasons,
            "demoted": self.demoted,
            "error": self.error,
        }


class StrategyMaintenance:
    """Promoted stratejilerin canlilik (decay) denetimi."""

    def __init__(
        self,
        settings: Settings,
        *,
        memory: Optional[MemoryStore] = None,
        registry: Optional[StrategyRegistry] = None,
        provider: Optional[MarketDataProvider] = None,
    ):
        self.settings = settings
        self._owns_memory = memory is None
        self.memory = memory or MemoryStore(Database(settings.memory.database))
        self.registry = registry or StrategyRegistry(settings)
        self.provider = provider or MarketDataProvider(settings)
        self.reports_dir = Path(settings.runtime.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _targets(self) -> Dict[str, float]:
        targets = self.settings.targets
        return {
            "min_net_profit_pct": float(targets.min_net_profit_pct),
            "min_win_rate": float(targets.min_win_rate),
            "max_drawdown_pct": float(targets.max_drawdown_pct),
            "min_sharpe": float(targets.min_sharpe),
            "min_profit_factor": float(targets.min_profit_factor),
            # Taze pencere daha kisa oldugu icin islem esigi yarilanir
            "min_trades": float(max(2, self.settings.backtest.min_trades // 2)),
        }

    def revalidate(
        self,
        *,
        limit: int = 10,
        bars: Optional[int] = None,
        demote: bool = False,
        cycle: int = 0,
        use_cache: bool = False,
    ) -> List[RevalidationResult]:
        """Kalici hafizadaki stratejileri guncel veriyle yeniden test eder."""
        data = self.provider.fetch_ohlcv(limit=bars, use_cache=use_cache)
        frame = data.frame
        engine = BacktestEngine(EngineConfig.from_settings(self.settings))
        targets = self._targets()
        cycle = cycle or int(self.memory.next_cycle_number())

        results: List[RevalidationResult] = []
        for row in self.memory.list_strategies(limit=limit, status="promoted"):
            version = str(row["version"])
            detail = self.memory.get_strategy(version)
            result = RevalidationResult(version=version, name=str(row.get("name", "")))
            try:
                if not detail or not detail.get("code_path"):
                    raise AgencyError("Strateji kodu bulunamadi")
                strategy = self.registry.instantiate(detail["code_path"])
                backtest = engine.run(strategy, frame)
                metrics = backtest.metrics
                gates = evaluate_gates(metrics.to_dict(), targets)
                score, _ = score_metrics(
                    metrics.to_dict(),
                    weights=self.settings.versioning.score_weights,
                    targets={**targets, "min_score": float(self.settings.targets.min_score)},
                )
                result.metrics = metrics
                result.gates = gates
                result.score = score
                result.ok = all(gates.get(name, False) for name in CORE_GATES)
                result.reasons = [
                    f"net%={metrics.net_profit_pct:.2f} win%={metrics.win_rate:.1f} "
                    f"dd%={metrics.max_drawdown_pct:.2f} sharpe={metrics.sharpe:.2f} "
                    f"pf={metrics.profit_factor:.2f} trades={metrics.trades}",
                    "kalan gate'ler: "
                    + str([name for name, passed in gates.items() if not passed] or "yok"),
                ]
                if not result.ok and demote:
                    self.memory.demote_strategy(
                        version,
                        cycle=cycle,
                        reason=f"decay check basarisiz: {metrics.short()}",
                        source="maintenance",
                    )
                    result.demoted = True
                    result.reasons.append("kalici hafizadan cikarildi (demoted)")
            except Exception as exc:  # tek strateji hatasi denetimi durdurmaz
                result.error = f"{type(exc).__name__}: {exc}"
                result.reasons = [result.error]
            results.append(result)

        self._write_report(results, data.source, len(frame), demote=demote)
        return results


    def _write_report(
        self,
        results: List[RevalidationResult],
        source: str,
        bars: int,
        *,
        demote: bool,
    ) -> Path:
        path = self.reports_dir / "REVALIDATION.md"
        lines = ["# Strateji Yeniden Dogrulama (Decay Check)", ""]
        lines.append(f"- **Tarih:** {iso_now()}")
        lines.append(f"- **Veri:** {source} ({bars} bar)")
        lines.append(f"- **Demote aktif:** {'evet' if demote else 'hayir'}")
        lines.append("")
        if not results:
            lines.append("_Kalici hafizada terfi etmis strateji yok._")
        else:
            lines.append(
                "| Surum | Isim | Durum | Net % | Win % | DD % | Sharpe | PF | Islem | Skor |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for item in results:
                state = "GECTI" if item.ok else ("DEMOTE" if item.demoted else "BASARISIZ")
                if item.error:
                    state = "HATA"
                m = item.metrics
                lines.append(
                    f"| {item.version} | {item.name[:24]} | {state} | "
                    f"{m.net_profit_pct:.2f} | {m.win_rate:.1f} | {m.max_drawdown_pct:.2f} | "
                    f"{m.sharpe:.2f} | {m.profit_factor:.2f} | {m.trades} | {item.score:.1f} |"
                )
            lines.append("")
            for item in results:
                lines.append(f"### {item.version} - {item.name}")
                lines.extend(f"- {reason}" for reason in item.reasons)
                lines.append("")
        write_text_atomic(path, "\n".join(lines) + "\n")
        return path

    def close(self) -> None:
        self.provider.close()
        if self._owns_memory:
            self.memory.close()


