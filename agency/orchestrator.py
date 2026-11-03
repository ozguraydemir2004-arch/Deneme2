"""OTONOM DONGU YONETICISI (Orchestrator).

Bir dongu:
    research -> (coder -> backtest -> critic) x aday x onarim -> hafiza/rapor

Dongu bittiginde elde edilen dersler hafizaya yazilir ve bir sonraki dongu
bu derslerle baslar. ``run()``, ``max_cycles_per_run`` veya durdurma sinyali
gelene kadar calisir (24/7 modunda sonsuz).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from agency.agents.backtester import BacktestAgent
from agency.agents.base import AgentContext
from agency.agents.coder import CoderAgent
from agency.agents.critic import CriticAgent
from agency.agents.planner import PlanContext, StrategyPlanner
from agency.agents.research import ResearchAgent
from agency.config import Settings
from agency.data.market_data import MarketDataProvider
from agency.llm import build_llm
from agency.logging_utils import AgencyLogger, build_logger
from agency.memory.database import Database
from agency.memory.repositories import MemoryStore
from agency.memory.vector_store import FeatureVectorizer
from agency.reporting import CycleReporter
from agency.schemas import CycleResult, ResearchReport, ReviewDecision, StrategyBuild
from agency.strategies.registry import StrategyRegistry
from agency.utils import iso_now, read_json, timestamp_slug, write_json
from agency.versioning import VersionManager


class AgencyOrchestrator:
    """Dort ajani bir dongu icinde calistiran otonom yonetici."""

    def __init__(
        self,
        settings: Settings,
        *,
        console: bool = True,
        max_cycles: Optional[int] = None,
        sleep_seconds: Optional[float] = None,
    ):
        self.settings = settings
        settings.ensure_directories()

        self.run_id = timestamp_slug()
        self.logger: AgencyLogger = build_logger(
            settings.logging.dir,
            level=settings.logging.level,
            jsonl=settings.logging.jsonl,
            console=console,
            run_id=self.run_id,
        )
        self.llm = build_llm(
            settings,
            log=lambda event, payload: self.logger.event(
                event, "LLM olayi", level="DEBUG", data=payload, agent="llm"
            ),
        )
        self.database = Database(settings.memory.database)
        self.memory = MemoryStore(
            self.database,
            vectorizer=FeatureVectorizer(settings.memory.vector_dim),
        )
        self.registry = StrategyRegistry(settings)
        self.provider = MarketDataProvider(settings)
        self.planner = StrategyPlanner(settings)
        self.versioning = VersionManager(settings, self.registry, self.memory)
        self.reporter = CycleReporter(settings, self.memory)

        self.state_path = Path(settings.runtime.state_file)
        self.state: Dict[str, Any] = read_json(self.state_path, default={}) or {}
        self.state.setdefault("runs", [])
        self.state.setdefault("best", {})

        self.max_cycles = int(
            max_cycles if max_cycles is not None else settings.targets.max_cycles_per_run
        )
        self.sleep_seconds = float(
            sleep_seconds if sleep_seconds is not None else settings.targets.cycle_sleep_seconds
        )
        self._stop = threading.Event()
        self._agents: Dict[str, Any] = {}
        self.ctx = self._build_context(cycle=1)
        self._make_agents(cycle=1)

        self.logger.info(
            "orchestrator_init",
            "Ajans sistemi hazir",
            data={
                "run_id": self.run_id,
                "llm": self.llm.describe(),
                "symbol": settings.exchange.symbol,
                "timeframe": settings.exchange.timeframe,
                "data_provider": settings.data.provider,
                "database": str(settings.memory.database),
                "max_cycles": self.max_cycles,
                "sleep_seconds": self.sleep_seconds,
            },
        )

    # ------------------------------------------------------------- context
    def _build_context(self, *, cycle: int) -> AgentContext:
        return AgentContext(
            settings=self.settings,
            memory=self.memory,
            logger=self.logger,
            llm=self.llm,
            registry=self.registry,
            data_provider=self.provider,
            planner=self.planner,
            cycle=cycle,
            run_id=self.run_id,
            state={},
        )

    def _make_agents(self, *, cycle: int) -> None:
        """Ajanlari dongu baglamiyla (yeniden) olusturur."""
        self.ctx = self._build_context(cycle=cycle)
        self._agents = {
            "research": ResearchAgent(self.ctx),
            "coder": CoderAgent(self.ctx),
            "backtest": BacktestAgent(self.ctx),
            "critic": CriticAgent(self.ctx),
        }

    @property
    def research_agent(self) -> ResearchAgent:
        return self._agents["research"]

    @property
    def coder_agent(self) -> CoderAgent:
        return self._agents["coder"]

    @property
    def backtest_agent(self) -> BacktestAgent:
        return self._agents["backtest"]

    @property
    def critic_agent(self) -> CriticAgent:
        return self._agents["critic"]

    # ----------------------------------------------------------------- run
    def request_stop(self, *_args: Any) -> None:
        """SIGINT/SIGTERM veya disaridan durdurma istegi."""
        if not self._stop.is_set():
            self.logger.warning(
                "stop_requested", "Durdurma istegi alindi, dongu tamamlaninca cikilacak"
            )
        self._stop.set()


    def run(self) -> List[CycleResult]:
        """Ana otonom dongu."""
        completed: List[CycleResult] = []
        started = time.perf_counter()
        self.logger.info(
            "run_start",
            "Otonom dongu basladi",
            data={
                "max_cycles": self.max_cycles or "sinirsiz (24/7)",
                "sleep_seconds": self.sleep_seconds,
            },
        )

        while not self._stop.is_set():
            if self.max_cycles and len(completed) >= self.max_cycles:
                break
            cycle = self.memory.next_cycle_number()
            result = self.run_cycle(cycle)
            completed.append(result)

            is_last = bool(self.max_cycles) and len(completed) >= self.max_cycles
            if not is_last and self.sleep_seconds > 0 and not self._stop.is_set():
                self.logger.info(
                    "cycle_sleep", f"Sonraki dongu icin {self.sleep_seconds:.0f}s bekleniyor"
                )
                self._sleep(self.sleep_seconds)

        duration = time.perf_counter() - started
        promoted = [version for result in completed for version in result.promoted]
        self.logger.info(
            "run_finished",
            f"{len(completed)} dongu tamamlandi ({duration:.1f}s)",
            data={
                "cycles": len(completed),
                "promoted": promoted,
                "stats": self.memory.stats(),
            },
        )
        self.state["last_run"] = {
            "run_id": self.run_id,
            "finished_at": iso_now(),
            "duration_seconds": round(duration, 2),
            "cycles": [r.cycle for r in completed],
            "promoted": promoted,
        }
        self._save_state()
        self.reporter.write_index()
        return completed

    def _sleep(self, seconds: float) -> None:
        deadline = time.time() + max(0.0, seconds)
        while time.time() < deadline and not self._stop.is_set():
            time.sleep(min(1.0, max(0.05, deadline - time.time())))

    # ----------------------------------------------------------- cycle run
    def run_cycle(self, cycle: int) -> CycleResult:
        """Tek bir tam dongu calistirir ve sonucu hafizaya yazar."""
        self._make_agents(cycle=cycle)
        started = time.perf_counter()
        result = CycleResult(cycle=cycle, status="running", started_at=iso_now())
        evaluations: List[Dict[str, Any]] = []
        research: Optional[ResearchReport] = None
        report_path: Optional[Path] = None

        self.logger.info(
            "cycle_start", f"=== DONGU {cycle:03d} BASLADI ===", data={"run_id": self.run_id}
        )
        try:
            research = self.research_agent.run(cycle)
            frame = self._frame_from_state()
            result.market = research.market.to_dict()
            result.data_source = research.data_source
            result.candidate_count = len(research.candidates)
            result.research_summary = research.market.regime_summary
            plan_ctx = self._plan_context(cycle, research)

            best_score = -1.0
            outcome = self._run_evaluation_phase(
                cycle=cycle, research=research, frame=frame, plan_ctx=plan_ctx
            )
            evaluations = list(outcome["evaluations"])
            result.versions = list(outcome["versions"])
            result.promoted = list(outcome["promoted"])
            result.archived = list(outcome["archived"])
            if outcome["best_score"] > best_score:
                best_score = outcome["best_score"]
                result.best_version = outcome["best_version"]
                result.best_score = outcome["best_score"]


            result.status = "completed"
        except Exception as exc:  # dongu asla sessizce olmez
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            self.logger.exception(
                "cycle_failed", f"Dongu {cycle} hata ile sonlandi: {exc}"
            )

        result.finished_at = iso_now()
        result.duration_seconds = round(time.perf_counter() - started, 2)
        lessons = self.memory.lessons_for_cycle(cycle, limit=12)
        result.lessons = list(dict.fromkeys(str(item.get("message")) for item in lessons))
        result.next_directives = self.memory.pending_directives(limit=6)

        try:
            report_path = self.reporter.write_cycle(result, research, evaluations)
            result.artifacts["report"] = str(report_path)
        except Exception as exc:  # pragma: no cover - rapor hatasi donguyu durdurmaz
            self.logger.warning("report_failed", f"Rapor yazilamadi: {exc}")

        self.memory.save_cycle(
            result,
            run_id=self.run_id,
            report_path=str(report_path) if report_path else None,
            symbol=self.settings.exchange.symbol,
            timeframe=self.settings.exchange.timeframe,
        )
        self._update_state(result)
        self.reporter.write_index()

        log_level = "info" if result.status == "completed" else "error"
        self.logger.event(
            "cycle_done",
            f"=== DONGU {cycle:03d} BITTI ({result.status}) | "
            f"en iyi={result.best_version or '-'} skor={result.best_score:.1f} | "
            f"terfi={result.promoted or '-'} ===",
            level="INFO" if log_level == "info" else "ERROR",
            data=result.to_dict(),
        )
        return result


    # ------------------------------------------------- candidate evaluation
    def _evaluate_candidate(
        self,
        candidate: Any,
        *,
        cycle: int,
        frame: pd.DataFrame,
        research: ResearchReport,
        plan_ctx: PlanContext,
    ) -> Dict[str, Any]:
        """Adayi kodla -> test et -> degerlendir -> gerekirse onar dongusune sokar."""
        max_attempts = int(self.settings.targets.max_rework_attempts)
        market_summary = research.market.regime_summary
        evaluations: List[Dict[str, Any]] = []
        versions: List[str] = []
        promoted: List[str] = []
        archived: List[str] = []
        best_score = -1.0
        best_version = None

        build: StrategyBuild = self._start_candidate(
            candidate, cycle=cycle, market_summary=market_summary, plan_ctx=plan_ctx
        )
        attempt = 0
        while True:
            report, decision, archive = self._test_and_review(
                build,
                cycle=cycle,
                frame=frame,
                market_summary=market_summary,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            evaluations.append(
                {"build": build, "report": report, "decision": decision, "archive": archive}
            )

            versions.append(build.version)
            if decision.score > best_score:
                best_score = decision.score
                best_version = build.version

            if decision.verdict == "rework" and attempt < max_attempts:
                attempt += 1
                self.logger.info(
                    "rework_loop",
                    f"{build.version} yeniden yazilacak (deneme {attempt}/{max_attempts})",
                    data={"directives": decision.directives[:4], "score": round(decision.score, 1)},
                )
                build = self.coder_agent.repair(
                    build,
                    report=report,
                    decision=decision,
                    cycle=cycle,
                    attempt=attempt,
                    market_summary=market_summary,
                    directives=decision.directives,
                    plan_ctx=plan_ctx,
                )
                continue

            if decision.verdict == "promote":
                promoted.append(build.version)
            else:
                archived.append(build.version)
            break

        return {
            "evaluations": evaluations,
            "versions": versions,
            "promoted": promoted,
            "archived": archived,
            "best_score": best_score,
            "best_version": best_version,
        }

    # ------------------------------------------------------- reusable steps
    def _start_candidate(
        self,
        candidate: Any,
        *,
        cycle: int,
        market_summary: str,
        plan_ctx: PlanContext,
    ) -> StrategyBuild:
        """Aday spec'ini koda cevirir (Coder Agent)."""
        return self.coder_agent.build_from_idea(
            candidate,
            cycle=cycle,
            market_summary=market_summary,
            directives=self.memory.pending_directives(limit=4),
            plan_ctx=plan_ctx,
        )

    def _test_and_review(
        self,
        build: StrategyBuild,
        *,
        cycle: int,
        frame: pd.DataFrame,
        market_summary: str,
        attempt: int,
        max_attempts: int,
    ) -> tuple:
        """Backtest + critic + arsiv adimlarini birlikte yurutur."""
        report = self.backtest_agent.run_suite(build, frame, cycle=cycle)
        decision: ReviewDecision = self.critic_agent.review(
            build,
            report,
            cycle=cycle,
            market_summary=market_summary,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        archive = self.versioning.archive(build, report, decision)
        return report, decision, archive

    def _run_evaluation_phase(
        self,
        *,
        cycle: int,
        research: ResearchReport,
        frame: pd.DataFrame,
        plan_ctx: PlanContext,
    ) -> Dict[str, Any]:
        """Aday degerlendirme fazi (backend tarafindan degistirilebilir).

        ``self.cycle_executor`` atanmissa (orn. LangGraph backend) o cagrilir;
        aksi halde yerlesik (native) dongu kullanilir.
        """
        if getattr(self, "cycle_executor", None) is not None:
            return self.cycle_executor(
                cycle=cycle, research=research, frame=frame, plan_ctx=plan_ctx
            )
        return self._native_evaluation_phase(
            cycle=cycle, research=research, frame=frame, plan_ctx=plan_ctx
        )

    def _native_evaluation_phase(
        self,
        *,
        cycle: int,
        research: ResearchReport,
        frame: pd.DataFrame,
        plan_ctx: PlanContext,
    ) -> Dict[str, Any]:
        """Yerlesik (backend'siz) aday degerlendirme dongusu."""
        evaluations: List[Dict[str, Any]] = []

        versions: List[str] = []
        promoted: List[str] = []
        archived: List[str] = []
        best_score = -1.0
        best_version = None
        for candidate in research.candidates:
            outcome = self._evaluate_candidate(
                candidate,
                cycle=cycle,
                frame=frame,
                research=research,
                plan_ctx=plan_ctx,
            )
            evaluations.extend(outcome["evaluations"])
            versions.extend(outcome["versions"])
            promoted.extend(outcome["promoted"])
            archived.extend(outcome["archived"])
            if outcome["best_score"] > best_score:
                best_score = outcome["best_score"]
                best_version = outcome["best_version"]
        return {
            "evaluations": evaluations,
            "versions": versions,
            "promoted": promoted,
            "archived": archived,
            "best_score": best_score,
            "best_version": best_version,
        }

    # --------------------------------------------------------------- helpers
    def _frame_from_state(self) -> pd.DataFrame:
        frame = self.ctx.state.get("last_frame")
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame
        return self.provider.fetch_ohlcv().frame

    def _plan_context(self, cycle: int, research: ResearchReport) -> PlanContext:
        elite = self.memory.memory_strategies(limit=3, regime=research.market.regime)
        lessons = [row.get("message", "") for row in self.memory.recent_lessons(limit=10)]
        return PlanContext(
            cycle=cycle,
            timeframe=research.market.timeframe,
            regime=research.market.regime,
            atr_pct=research.market.atr_pct,
            rsi=research.market.rsi,
            adx=research.market.adx,
            winner_features=[m.get("features") or {} for m in research.top_matches],
            lessons=lessons,
            directives=self.memory.pending_directives(limit=6),
            elite_specs=[
                {**item.get("spec", {}), "_version": item.get("version")}
                for item in elite
                if item.get("spec")
            ],
        )

    def _update_state(self, result: CycleResult) -> None:
        self.state["last_cycle"] = result.cycle
        self.state["last_status"] = result.status
        self.state["updated_at"] = iso_now()
        best = self.state.get("best") or {}
        if result.best_score > float(best.get("score") or 0.0):
            self.state["best"] = {
                "version": result.best_version,
                "score": round(result.best_score, 2),
                "cycle": result.cycle,
            }
        runs = self.state.setdefault("runs", [])
        runs.append(
            {
                "cycle": result.cycle,
                "status": result.status,
                "best_version": result.best_version,
                "best_score": round(result.best_score, 2),
                "promoted": result.promoted,
                "versions": result.versions,
                "duration_seconds": result.duration_seconds,
                "finished_at": result.finished_at,
            }
        )
        self.state["runs"] = runs[-200:]
        self._save_state()

    def _save_state(self) -> None:
        try:
            write_json(self.state_path, self.state)
        except OSError as exc:  # pragma: no cover
            self.logger.warning("state_save_failed", f"Durum dosyasi yazilamadi: {exc}")

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        stats = self.memory.stats()
        try:
            self.provider.close()
        finally:
            self.memory.close()
        self.logger.info("orchestrator_closed", "Kaynaklar kapatildi", data=stats)

    def __enter__(self) -> "AgencyOrchestrator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


