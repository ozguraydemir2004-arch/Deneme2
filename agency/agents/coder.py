"""STRATEJI VE KOD YAZAR AJANI (Coder Agent).

Gorevi:
    * Research Agent'in aday spec'lerini inceler.
    * Spec'i **derlenebilir** OOP Python stratejisine cevirir (compiler).
    * Critic direktifleri / hata matrisi geldiginde stratejiyi onarip yeni
      SURUM uretir (parent_version ile soy agaci korunur).

Not: LLM yalnizca **spec** uretir; kod her zaman deterministik derleyiciden
gecer. Bu, "LLM bozuk kod yazdi" sinifindaki arizalari yapisal olarak engeller.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from agency.agents.base import BaseAgent
from agency.agents.planner import PlanContext
from agency.agents.prompts import CODER_SYSTEM, coder_prompt
from agency.exceptions import StrategyValidationError
from agency.schemas import BacktestReport, ReviewDecision, StrategyBuild, StrategySpec


class CoderAgent(BaseAgent):
    """Spec -> versiyonlanmis strateji kodu."""

    name = "coder"
    role = "Strateji ve kod yazari"

    # ---------------------------------------------------------------- build
    def build(
        self,
        spec: StrategySpec,
        *,
        cycle: int,
        parent_version: Optional[str] = None,
        attempt: int = 1,
        notes: Sequence[str] = (),
        generated_by: str = "coder_agent",
    ) -> StrategyBuild:
        """Spec'i derler, dogrular, diske yazar ve hafizaya kaydeder."""
        try:
            build = self.registry.create_build(
                spec,
                cycle=cycle,
                parent_version=parent_version,
                attempt=attempt,
                notes=notes,
                generated_by=generated_by,
            )
        except StrategyValidationError as exc:
            self.warn(
                "build_failed",
                f"Spec derlenemedi, deterministik sablona dusuluyor: {exc}",
                errors=(exc.details or {}).get("issues", []),
            )
            fallback = self.planner.sanitize(spec, PlanContext(cycle=cycle))
            build = self.registry.create_build(
                fallback,
                cycle=cycle,
                parent_version=parent_version,
                attempt=attempt,
                notes=list(notes) + ["dogrulama hatasi sonrasi sablon onarimi"],
                generated_by="coder_agent_fallback",
            )

        self.memory.save_strategy_version(build, status="draft")
        self.info(
            "strategy_built",
            f"{build.version} derlendi ({build.spec.family})",
            version=build.version,
            path=build.module_path,
            fingerprint=build.spec.fingerprint(),
            attempt=attempt,
        )
        return build

    # ---------------------------------------------------------- from idea
    def build_from_idea(
        self,
        idea: Mapping[str, Any] | StrategySpec,
        *,
        cycle: int,
        market_summary: str,
        directives: Sequence[str] = (),
        attempt: int = 1,
        parent_version: Optional[str] = None,
        plan_ctx: Optional[PlanContext] = None,
    ) -> StrategyBuild:
        """Arastirma fikrinden (gerekirse LLM ile) spec uretip derler."""
        ctx = plan_ctx or PlanContext(cycle=cycle)
        spec: Optional[StrategySpec] = None

        if self.llm.available:
            idea_payload = idea.to_dict() if isinstance(idea, StrategySpec) else dict(idea)
            elite = self.memory.memory_strategies(limit=3)
            prompt = coder_prompt(
                market_summary=market_summary,
                idea=idea_payload,
                directives=list(directives),
                repair_context=None,
                elite_specs=[item.get("spec", {}) for item in elite],
                attempt=attempt,
            )
            payload = self.ask_json(CODER_SYSTEM, prompt, label="coder_build")
            spec = self.planner.repair_with_llm_payload(payload, ctx)

        if spec is None:
            raw = idea.to_dict() if isinstance(idea, StrategySpec) else dict(idea)
            raw.setdefault("name", f"strategy_c{cycle}")
            raw.setdefault("timeframe", ctx.timeframe)
            raw["origin"] = raw.get("origin", "research")
            spec = self.planner.sanitize(StrategySpec.from_dict(raw), ctx)
            self.info(
                "deterministic_spec",
                "Spec deterministik planlayicidan alindi",
                name=spec.name,
                family=spec.family,
            )

        return self.build(
            spec,
            cycle=cycle,
            attempt=attempt,
            parent_version=parent_version,
            notes=list(spec.notes) + list(directives),
            generated_by="coder_agent" if self.llm.available else "coder_agent_heuristic",
        )


    # -------------------------------------------------------------- repair
    def repair(
        self,
        build: StrategyBuild,
        *,
        report: BacktestReport,
        decision: ReviewDecision,
        cycle: int,
        attempt: int,
        market_summary: str,
        directives: Sequence[str] = (),
        plan_ctx: Optional[PlanContext] = None,
    ) -> StrategyBuild:
        """Backtest/critic geri bildirimine gore yeni surum uretir."""
        ctx = plan_ctx or PlanContext(cycle=cycle)
        metrics = report.metrics.to_dict()
        gate_failures = [name for name, passed in decision.metric_gates.items() if not passed]
        errors = [entry.message for entry in report.errors.entries]
        self.info(
            "repair_start",
            f"{build.version} onariliyor (deneme {attempt})",
            failed_gates=gate_failures,
            errors=errors[:3],
            directives=list(directives)[:5],
        )

        new_spec: Optional[StrategySpec] = None
        if self.llm.available:
            prompt = coder_prompt(
                market_summary=market_summary,
                idea=build.spec.to_dict(),
                directives=list(directives) or decision.directives,
                repair_context={
                    "metrics": metrics,
                    "errors": errors[:5],
                    "verdict": decision.verdict,
                    "gates": decision.metric_gates,
                },
                elite_specs=[
                    item.get("spec", {}) for item in self.memory.memory_strategies(limit=3)
                ],
                attempt=attempt,
            )
            payload = self.ask_json(CODER_SYSTEM, prompt, label="coder_repair")
            new_spec = self.planner.repair_with_llm_payload(payload, ctx)

        if new_spec is None:
            new_spec = self.planner.repair(
                build.spec,
                metrics={**metrics, "bars": report.bars},
                gates=decision.metric_gates,
                directives=list(directives) or decision.directives,
                errors=errors,
                attempt=attempt,
                ctx=ctx,
            )
            self.info(
                "deterministic_repair",
                "Onarim deterministik planlayici ile yapildi",
                changes=new_spec.notes[-6:],
            )

        # Onarim ayni spec'i urettiyse (parametreler sinirda doydu) kesif mutasyonu
        if new_spec.fingerprint() == build.spec.fingerprint():
            new_spec = self.planner.perturb(new_spec, attempt=attempt)
            self.info(
                "repair_perturb",
                "Onarim degisiklik uretmedi; kesif mutasyonu uygulandi",
                changes=new_spec.notes[-4:],
            )

        return self.build(
            new_spec,
            cycle=cycle,
            attempt=attempt,
            parent_version=build.version,
            notes=new_spec.notes[-8:],
            generated_by="coder_agent_repair" if self.llm.available else "coder_agent_heuristic",
        )

    # ---------------------------------------------------------------- info
    def describe_candidate(self, spec: StrategySpec) -> Dict[str, Any]:
        return {
            "name": spec.name,
            "family": spec.family,
            "indicators": [f"{i.kind}({i.resolved_params()})" for i in spec.indicators],
            "entry": [r.describe() for r in spec.entry_long],
            "exit": [r.describe() for r in spec.exit_long],
            "filters": [r.describe() for r in spec.filters],
            "risk": {
                "stop_loss_atr": spec.stop_loss_atr,
                "take_profit_atr": spec.take_profit_atr,
                "risk_per_trade_pct": spec.risk_per_trade_pct,
            },
        }

