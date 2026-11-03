"""YONETICI VE OPTIMIZASYON AJANI (Critic & Optimizer Agent).

Gorevi:
    * Backtest sonuclarini hedef esiklerle karsilastirmak (gate tablosu).
    * 0-100 skor uretmek (net kar, win rate, drawdown, sharpe, profit factor).
    * Karar vermek: ``promote`` (kalici hafizaya al) / ``rework`` (coder'a
      direktifle geri gonder) / ``archive`` (arsivle, dongu devam etsin).
    * Basarili stratejiyi kalici hafizaya yazmak; basarisizliktan DERS
      cikarip bir sonraki donguye tasimak.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

from agency.agents.base import BaseAgent
from agency.agents.prompts import CRITIC_SYSTEM, critic_prompt
from agency.backtest.metrics import evaluate_gates, score_metrics
from agency.schemas import BacktestReport, ReviewDecision, StrategyBuild
from agency.utils import iso_now


class CriticAgent(BaseAgent):
    """Skorlama, direktif uretimi ve kalici hafiza terfisi."""

    name = "critic"
    role = "Yonetici ve optimizasyon"

    def target_dict(self) -> Dict[str, float]:
        targets = self.settings.targets
        return {
            "min_net_profit_pct": float(targets.min_net_profit_pct),
            "min_win_rate": float(targets.min_win_rate),
            "max_drawdown_pct": float(targets.max_drawdown_pct),
            "min_sharpe": float(targets.min_sharpe),
            "min_profit_factor": float(targets.min_profit_factor),
            "min_score": float(targets.min_score),
            "min_trades": float(self.settings.backtest.min_trades),
        }

    # -------------------------------------------------------------- review
    def review(
        self,
        build: StrategyBuild,
        report: BacktestReport,
        *,
        cycle: int,
        market_summary: str,
        attempt: int,
        max_attempts: int,
    ) -> ReviewDecision:
        targets = self.target_dict()
        metrics = report.metrics.to_dict()
        oos_metrics = report.oos_metrics.to_dict() if report.oos_metrics else None

        gates = evaluate_gates(metrics, targets)
        folds = list(report.fold_metrics or [])
        fold_nets = [float(m.net_profit_pct) for m in folds]
        fold_positive = sum(1 for value in fold_nets if value > 0)
        folds_ok = bool(folds) and fold_positive >= math.ceil(len(folds) * 0.6)
        oos_confirmed = False
        oos_net = float(oos_metrics.get("net_profit_pct", 0.0)) if oos_metrics else None
        if oos_metrics:
            # OOS penceresi daha kisa oldugu icin islem esigi daha dusuk tutulur
            oos_targets = dict(targets)
            oos_targets["min_trades"] = max(2.0, float(targets["min_trades"]) / 2.0)
            oos_gates = evaluate_gates(oos_metrics, oos_targets)
            gates.update({f"oos_{k}": v for k, v in oos_gates.items()})
            gates["oos_not_catastrophic"] = bool(oos_net is not None and oos_net > -1.5)
        if folds:
            gates["folds_positive"] = folds_ok
            gates["folds_mean_positive"] = (sum(fold_nets) / len(fold_nets)) > 0.0

        not_catastrophic = oos_net is None or oos_net > -1.5
        oos_confirmed = (folds_ok or bool(oos_net is not None and oos_net > 0.0)) and not_catastrophic

        score, breakdown = score_metrics(
            metrics,
            weights=self.settings.versioning.score_weights,
            targets=targets,
        )
        if folds:
            if fold_positive == len(folds):
                score = min(100.0, score * 1.05)
                breakdown["folds_bonus"] = round(score * 0.05, 2)
            elif fold_positive == 0:
                score *= 0.8
                breakdown["folds_penalty"] = -round(score * 0.2, 2)
        if oos_net is not None and oos_net < 0:
            score *= 0.9
            breakdown["oos_penalty"] = -round(score * 0.1, 2)


        fatal_error = report.errors.has_errors
        core_gates = ("net_profit", "win_rate", "drawdown", "sharpe", "profit_factor", "trades")
        passed_core = all(gates.get(g, False) for g in core_gates)
        # Terfi icin out-of-sample onayi sart: yeterli islem varsa PF, yoksa folds
        oos_pf_ok = True
        if oos_metrics and int(oos_metrics.get("trades", 0)) >= 3:
            oos_pf_ok = float(oos_metrics.get("profit_factor", 0.0)) >= max(
                1.0, float(targets["min_profit_factor"]) * 0.9
            )
        min_score = float(targets["min_score"])
        can_rework = attempt < int(max_attempts)

        if fatal_error:
            verdict = "rework" if can_rework else "archive"
        elif passed_core and score >= min_score and oos_confirmed and oos_pf_ok:
            verdict = "promote"
        elif can_rework and score >= min_score * 0.55:
            verdict = "rework"
        elif can_rework and attempt == 0:
            verdict = "rework"
        else:
            verdict = "archive"

        reasons = self._reasons(metrics, oos_metrics, gates, score, targets, report)
        directives = self._directives(metrics, oos_metrics, gates, targets, report, build)

        llm_note = ""
        if self.llm.available:
            prompt = critic_prompt(
                market_summary=market_summary,
                metrics=metrics,
                oos_metrics=oos_metrics,
                gates=gates,
                targets=targets,
                trades_summary=self._trade_summary(report),
                errors=[e.message for e in report.errors.entries],
                attempt=attempt + 1,
                max_attempts=max_attempts,
            )
            payload = self.ask_json(CRITIC_SYSTEM, prompt, label="critic_review")
            if payload:
                llm_directives = [
                    str(d).strip() for d in (payload.get("directives") or []) if str(d).strip()
                ]
                llm_reasons = [str(r).strip() for r in (payload.get("reasons") or []) if str(r).strip()]
                if llm_directives:
                    directives = list(dict.fromkeys(llm_directives + directives))[:8]
                reasons.extend([r for r in llm_reasons if r not in reasons][:4])
                llm_note = f"LLM ek gerekcesi: {payload.get('verdict')}"

        elite = score >= float(self.settings.versioning.elite_score)
        decision = ReviewDecision(
            strategy_version=build.version,
            verdict=verdict,
            score=score,
            score_breakdown=breakdown,
            metric_gates=gates,
            reasons=reasons,
            directives=directives,
            oos_confirmed=oos_confirmed,
            elite=elite and verdict == "promote",
            created_at=iso_now(),
        )

        self._persist(build, decision, report, cycle=cycle, metrics=metrics, market_summary=market_summary)
        self.info(
            "review_done",
            f"{build.version} -> {verdict} | skor={score:.1f} | gates={sum(1 for v in gates.values() if v)}/{len(gates)}",
            verdict=verdict,
            score=round(score, 2),
            failed_gates=[k for k, v in gates.items() if not v],
            directives=directives[:5],
            oos_confirmed=oos_confirmed,
            note=llm_note,
        )
        return decision


    # ------------------------------------------------------- explanation
    def _trade_summary(self, report: BacktestReport) -> str:
        trades = report.trades or []
        if not trades:
            return "islem yok"
        reasons: Dict[str, int] = {}
        for trade in trades:
            key = str(trade.get("exit_reason", "?"))[:24]
            reasons[key] = reasons.get(key, 0) + 1
        ordered = sorted(reasons.items(), key=lambda item: item[1], reverse=True)[:5]
        best = max(trades, key=lambda t: float(t.get("pnl_pct", 0.0)))
        worst = min(trades, key=lambda t: float(t.get("pnl_pct", 0.0)))
        return (
            f"{len(trades)} islem; cikis nedenleri={dict(ordered)}; "
            f"en iyi={float(best.get('pnl_pct', 0)):.2f}% "
            f"en kotu={float(worst.get('pnl_pct', 0)):.2f}%"
        )

    def _reasons(
        self,
        metrics: Mapping[str, Any],
        oos: Optional[Mapping[str, Any]],
        gates: Mapping[str, bool],
        score: float,
        targets: Mapping[str, float],
        report: BacktestReport,
    ) -> List[str]:
        reasons: List[str] = []
        if report.errors.has_errors:
            reasons.append(f"Kod/hata matrisi: {report.errors.summary[:200]}")
        reasons.append(
            f"net kar %{float(metrics.get('net_profit_pct', 0.0)):.2f} "
            f"(esik %{targets['min_net_profit_pct']:.1f}) | "
            f"win %{float(metrics.get('win_rate', 0.0)):.1f} "
            f"(esik %{targets['min_win_rate']:.1f})"
        )
        reasons.append(
            f"drawdown %{float(metrics.get('max_drawdown_pct', 0.0)):.1f} "
            f"(limit %{targets['max_drawdown_pct']:.1f}) | "
            f"sharpe {float(metrics.get('sharpe', 0.0)):.2f} | "
            f"PF {float(metrics.get('profit_factor', 0.0)):.2f}"
        )
        failed = [name for name, passed in gates.items() if not passed and not name.startswith("oos_")]
        if failed:
            reasons.append(f"Esikte kalan metrikler: {', '.join(failed)}")
        if oos:
            reasons.append(
                f"Out-of-sample net %{float(oos.get('net_profit_pct', 0.0)):.2f}, "
                f"win %{float(oos.get('win_rate', 0.0)):.1f}"
            )
        folds = list(report.fold_metrics or [])
        if folds:
            reasons.append(
                "Walk-forward dilimleri: "
                + ", ".join(f"%{float(m.net_profit_pct):.2f}" for m in folds)
            )
        reasons.append(f"Toplam skor: {score:.1f}/100")
        return reasons


    def _directives(
        self,
        metrics: Mapping[str, Any],
        oos: Optional[Mapping[str, Any]],
        gates: Mapping[str, bool],
        targets: Mapping[str, float],
        report: BacktestReport,
        build: StrategyBuild,
    ) -> List[str]:
        """Coder'a gidecek SOMUT, olculebilir direktifler."""
        directives: List[str] = []
        spec = build.spec
        trades = int(metrics.get("trades", 0) or 0)
        bars = int(metrics.get("bars", 0) or 0)
        win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
        dd = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
        net = float(metrics.get("net_profit_pct", 0.0) or 0.0)
        pf = float(metrics.get("profit_factor", 0.0) or 0.0)

        if gates.get("drawdown") is False:
            new_stop = round(max(0.5, spec.stop_loss_atr * 0.8), 2)
            new_risk = round(max(0.3, spec.risk_per_trade_pct * 0.6), 2)
            directives.append(
                f"stop_loss_atr {spec.stop_loss_atr} -> {new_stop}; "
                f"risk_per_trade_pct {spec.risk_per_trade_pct} -> {new_risk} (drawdown %{dd:.1f})"
            )
            if spec.trailing_stop_atr <= 0:
                directives.append(f"trailing_stop_atr 0 -> {round(new_stop * 0.8, 2)} ekle")
        if gates.get("win_rate") is False:
            directives.append(
                f"win rate %{win_rate:.1f} dusuk: ADX filtresi esigini +2 artir ve "
                "take_profit_atr degerini %15 azalt"
            )
        if gates.get("trades") is False:
            directives.append(
                f"islem sayisi {trades} < min {int(targets['min_trades'])}: giris kosulunu gevset "
                "(kesisim -> karsilastirma) veya filtre esigini %20 dusur"
            )
        if bars and trades > max(10, bars // 6):
            directives.append(
                f"asiri islem ({trades} islem / {bars} bar): cooldown_bars "
                f"{spec.cooldown_bars} -> {spec.cooldown_bars + 4}"
            )
        if gates.get("net_profit") is False and pf < 1.0:
            directives.append(
                f"profit factor {pf:.2f} < 1: giris kalitesini artir (ADX +1), "
                f"TP {spec.take_profit_atr} -> {round(max(0.6, spec.take_profit_atr * 0.85), 2)}"
            )
        if oos and float(oos.get("net_profit_pct", 0.0)) < 0 < net:
            directives.append(
                "out-of-sample negatif: asiri uyum (overfit) riski; parametre sayisini azalt, "
                "trend filtresi ekle ve risk_per_trade_pct degerini 0.75'e cek"
            )
        folds = list(report.fold_metrics or [])
        if folds and sum(1 for m in folds if m.net_profit_pct > 0) == 0:
            directives.append(
                "tum walk-forward dilimleri negatif: strateji ailesini degistir "
                "(trend/mean-reversion) ve trend filtresi ekle"
            )
        if report.errors.has_errors:
            directives.append(f"hata matrisini gider: {report.errors.summary[:160]}")
        if not directives:
            directives.append("esikler gecti: canliya hazir, kucuk pozisyonla izleme moduna al")
        return list(dict.fromkeys(directives))[:8]

    # --------------------------------------------------------- persistence
    def _persist(
        self,
        build: StrategyBuild,
        decision: ReviewDecision,
        report: BacktestReport,
        *,
        cycle: int,
        metrics: Mapping[str, Any],
        market_summary: str,
    ) -> None:
        status_map = {"promote": "promoted", "rework": "rework", "archive": "rejected"}
        status = status_map.get(decision.verdict, "rejected")
        self.memory.update_strategy_result(
            build.version,
            status=status,
            score=decision.score,
            metrics=metrics,
            review=decision,
        )
        self.memory.save_review(decision, cycle=cycle)

        if decision.verdict == "promote":
            regime = ""
            for part in market_summary.split():
                if part.startswith("rejim="):
                    regime = part.split("=", 1)[1].split("/")[0]
                    break
            try:
                self.memory.promote_strategy(
                    build.version,
                    cycle=cycle,
                    regime=regime or "neutral",
                    score=decision.score,
                    symbol=self.settings.exchange.symbol,
                    timeframe=self.settings.exchange.timeframe,
                )
                self.info(
                    "memory_promoted",
                    f"{build.version} kalici hafizaya eklendi (skor {decision.score:.1f})",
                    regime=regime or "neutral",
                    elite=decision.elite,
                )
            except KeyError as exc:  # pragma: no cover - savunma
                self.warn("memory_promote_failed", str(exc))

        # Dersler: basarisiz gate'ler + genel karar ozeti
        metric_keys = {
            "net_profit": "net_profit_pct",
            "win_rate": "win_rate",
            "drawdown": "max_drawdown_pct",
            "sharpe": "sharpe",
            "profit_factor": "profit_factor",
            "trades": "trades",
            "folds_positive": "fold_positive_count",
            "folds_mean_positive": "fold_mean_net_pct",
        }
        for name in [n for n, passed in decision.metric_gates.items() if not passed]:
            if name.startswith("oos_"):
                continue  # OOS gate'leri tek tek ders uretmez (gurultu)
            metric_key = metric_keys.get(name)
            value = metrics.get(metric_key) if metric_key else None
            suffix = f" (deger={value})" if value is not None else ""
            self.memory.add_lesson(
                cycle,
                f"{build.version} ({build.spec.family}) '{name}' esigini gecemedi{suffix}",
                directive=decision.directives[0] if decision.directives else None,
                scope="metric",
                grade="warning",
                source_version=build.version,
            )
        self.memory.add_lesson(
            cycle,
            f"{build.version} karari={decision.verdict} skor={decision.score:.1f} "
            f"net%={float(metrics.get('net_profit_pct', 0.0)):.2f} "
            f"win%={float(metrics.get('win_rate', 0.0)):.1f} "
            f"dd%={float(metrics.get('max_drawdown_pct', 0.0)):.1f}",
            directive=decision.directives[0] if decision.directives else None,
            scope="review",
            grade="info" if decision.verdict == "promote" else "warning",
            source_version=build.version,
        )


