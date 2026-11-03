"""Strateji planlayici ve Coder Agent testleri (uretim + onarim dongusu)."""

from __future__ import annotations

from agency.agents.coder import CoderAgent
from agency.agents.planner import FAMILIES, PlanContext, StrategyPlanner
from agency.schemas import BacktestMetrics, BacktestReport, ErrorMatrix, ReviewDecision


def plan_context(**overrides) -> PlanContext:
    base = dict(cycle=3, timeframe="1h", regime="trend_up", atr_pct=1.2, rsi=55.0, adx=24.0)
    base.update(overrides)
    return PlanContext(**base)


def test_plan_produces_valid_distinct_specs(settings) -> None:
    planner = StrategyPlanner(settings)
    specs = planner.plan(plan_context(), count=3)
    assert len(specs) == 3
    assert len({spec.fingerprint() for spec in specs}) == 3
    for spec in specs:
        assert spec.validate() == []
        assert spec.family in FAMILIES


def test_plan_is_deterministic(settings) -> None:
    planner = StrategyPlanner(settings)
    first = [spec.fingerprint() for spec in planner.plan(plan_context(), count=2)]
    second = [spec.fingerprint() for spec in planner.plan(plan_context(), count=2)]
    assert first == second


def test_plan_respects_regime_preference(settings) -> None:
    planner = StrategyPlanner(settings)
    squeeze = planner.plan(plan_context(regime="squeeze"), count=1)
    assert squeeze[0].family in {"volatility_compression", "breakout", "mean_reversion", "hybrid"}


def test_elite_variant_uses_memory(settings, sample_spec) -> None:
    planner = StrategyPlanner(settings)
    ctx = plan_context(elite_specs=[{**sample_spec.to_dict(), "_version": "v009"}])
    specs = planner.plan(ctx, count=1)
    assert specs[0].origin == "memory"
    assert specs[0].family == sample_spec.family
    assert specs[0].parent_version == "v009"
    assert specs[0].validate() == []


def test_repair_changes_losing_spec(settings, sample_spec) -> None:
    planner = StrategyPlanner(settings)
    gates = {name: False for name in ("net_profit", "win_rate", "drawdown", "sharpe",
                                      "profit_factor", "trades")}
    repaired = planner.repair(
        sample_spec,
        metrics={
            "net_profit_pct": -4.0,
            "win_rate": 20.0,
            "trades": 3,
            "bars": 800,
            "max_drawdown_pct": 30.0,
            "sharpe": -1.0,
            "profit_factor": 0.4,
        },
        gates=gates,
        directives=["risk_per_trade_pct 1.5 -> 0.75"],
        errors=["KeyError: atr"],
        attempt=1,
        ctx=plan_context(),
    )
    assert repaired.validate() == []
    assert repaired.fingerprint() != sample_spec.fingerprint()
    assert repaired.risk_per_trade_pct < sample_spec.risk_per_trade_pct
    assert repaired.stop_loss_atr < sample_spec.stop_loss_atr
    assert repaired.name.endswith("_r1")


def test_repair_zero_trade_is_structural(settings, sample_spec) -> None:
    planner = StrategyPlanner(settings)
    repaired = planner.repair(
        sample_spec,
        metrics={
            "net_profit_pct": 0.0,
            "win_rate": 0.0,
            "trades": 0,
            "bars": 800,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
        },
        gates={"trades": False, "net_profit": False},
        attempt=1,
        ctx=plan_context(),
    )
    assert repaired.filters == []
    assert all(rule.op not in {"cross_above", "cross_below"} for rule in repaired.entry_long)
    assert repaired.validate() == []


def test_perturb_always_changes_fingerprint(settings, sample_spec) -> None:
    planner = StrategyPlanner(settings)
    seen = {sample_spec.fingerprint()}
    for attempt in range(1, 5):
        mutated = planner.perturb(sample_spec, attempt=attempt)
        assert mutated.validate() == []
        assert mutated.name.endswith(f"_p{attempt}")
        seen.add(mutated.fingerprint())
    assert len(seen) >= 4


def test_directives_are_parsed(settings, sample_spec) -> None:
    planner = StrategyPlanner(settings)
    updated = planner.apply_directives(
        sample_spec,
        [
            "stop_loss_atr 2.0 -> 1.5",
            "take_profit_atr 4.0 -> 6.0",
            "adx_14 15 -> 20",
            "trend filtresi ekle",
        ],
    )
    assert updated.stop_loss_atr == 1.5
    assert updated.take_profit_atr == 6.0
    adx_rules = [r for r in updated.filters if r.left == "adx_14"]
    assert adx_rules and float(adx_rules[0].right) == 20.0
    assert updated.validate() == []


def test_llm_ideas_are_validated(settings) -> None:
    planner = StrategyPlanner(settings)
    payload = {
        "strategy_ideas": [
            {
                "name": "llm_idea",
                "family": "momentum",
                "timeframe": "1h",
                "indicators": [
                    {
                        "key": "macd_hist",
                        "kind": "macd",
                        "output": "hist",
                        "params": {"fast": 12, "slow": 26, "signal": 9},
                    },
                    {"key": "ema_trend", "kind": "ema", "params": {"period": 80}},
                ],
                "entry_long": [{"left": "macd_hist", "op": "cross_above", "right": 0}],
                "filters": [{"left": "close", "op": ">", "right": "ema_trend"}],
                "stop_loss_atr": 2.0,
                "take_profit_atr": 4.0,
                "risk_per_trade_pct": 1.0,
            },
            {"name": "bozuk", "family": "hybrid", "indicators": []},
        ]
    }
    specs = planner.from_llm_ideas(payload, plan_context())
    assert len(specs) >= 1
    assert specs[0].name == "llm_idea"
    # Eksik/bozuk fikirler ya ayiklanir ya da katalog tabanli onarilir
    for spec in specs:
        assert spec.validate() == []
        assert spec.indicators and spec.entry_long


def test_coder_build_and_repair_cycle(agent_context, sample_spec) -> None:
    coder = CoderAgent(agent_context)
    build = coder.build(sample_spec, cycle=1, notes=["ilk uretim"])
    assert build.version == "v001"
    assert agent_context.memory.get_strategy(build.version)["status"] == "draft"

    report = BacktestReport(
        strategy_version=build.version,
        name=build.name,
        metrics=BacktestMetrics(net_profit_pct=-3.0, win_rate=25.0, trades=4, bars=800),
        bars=800,
        errors=ErrorMatrix(),
    )
    decision = ReviewDecision(
        strategy_version=build.version,
        verdict="rework",
        score=30.0,
        metric_gates={"trades": False, "drawdown": False, "net_profit": False},
        directives=[
            "stop_loss_atr 2.0 -> 1.5",
            "islem sayisi 4 < min 8: giris kosulunu gevset",
        ],
    )
    repaired = coder.repair(
        build,
        report=report,
        decision=decision,
        cycle=1,
        attempt=1,
        market_summary="BTC/USDT 1h rejim=range akis",
        directives=decision.directives,
    )
    assert repaired.version == "v002"
    assert repaired.parent_version == build.version
    assert repaired.spec.fingerprint() != build.spec.fingerprint()
    assert repaired.spec.validate() == []
    assert agent_context.memory.get_strategy(repaired.version) is not None
