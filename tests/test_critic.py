"""Critic & Optimizer ajan testleri (terfi/onarim/arsiv kararlari)."""

from __future__ import annotations

import pytest

from agency.agents.base import AgentContext
from agency.agents.critic import CriticAgent
from agency.agents.planner import StrategyPlanner
from agency.data.market_data import MarketDataProvider
from agency.llm import build_llm
from agency.memory.repositories import MemoryStore
from agency.schemas import BacktestMetrics, BacktestReport, ReviewDecision
from agency.strategies.registry import StrategyRegistry


@pytest.fixture()
def context(settings, memory: MemoryStore, registry: StrategyRegistry) -> AgentContext:
    from agency.logging_utils import build_logger

    logger = build_logger(settings.logging.dir, console=False, run_id="test")
    return AgentContext(
        settings=settings,
        memory=memory,
        logger=logger,
        llm=build_llm(settings),
        registry=registry,
        data_provider=MarketDataProvider(settings),
        planner=StrategyPlanner(settings),
        cycle=1,
        run_id="test",
    )


def metrics(**overrides) -> BacktestMetrics:
    base = dict(
        net_profit_pct=6.0,
        win_rate=55.0,
        max_drawdown_pct=5.0,
        sharpe=1.4,
        profit_factor=1.8,
        trades=25,
        bars=800,
        buy_hold_pct=1.0,
    )
    base.update(overrides)
    return BacktestMetrics(**base)


def report_with(train: BacktestMetrics, folds=None, oos=None) -> BacktestReport:
    return BacktestReport(
        strategy_version="v001",
        name="test",
        metrics=train,
        fold_metrics=folds or [],
        oos_metrics=oos,
        bars=800,
    )


def test_promotes_strong_strategy(context: AgentContext, sample_spec) -> None:
    build = context.registry.create_build(sample_spec, cycle=1)
    context.memory.save_strategy_version(build)
    report = report_with(
        metrics(),
        folds=[metrics(net_profit_pct=2.0), metrics(net_profit_pct=1.0), metrics(net_profit_pct=-0.5)],
        oos=metrics(net_profit_pct=1.2, trades=10, win_rate=50.0),
    )
    critic = CriticAgent(context)
    decision = critic.review(
        build, report, cycle=1, market_summary="BTC/USDT 1h rejim=range/weak", attempt=0, max_attempts=3
    )
    assert decision.verdict == "promote"
    assert decision.score >= 50
    stored = context.memory.get_strategy(build.version)
    assert stored["status"] == "promoted"
    assert context.memory.memory_strategies(limit=5)[0]["version"] == build.version


def test_rejects_overfit_strategy(context: AgentContext, sample_spec) -> None:
    build = context.registry.create_build(sample_spec, cycle=1)
    context.memory.save_strategy_version(build)
    report = report_with(
        metrics(),
        folds=[metrics(net_profit_pct=-1.0), metrics(net_profit_pct=-2.0), metrics(net_profit_pct=-3.0)],
        oos=metrics(net_profit_pct=-4.0, trades=12),
    )
    decision = CriticAgent(context).review(
        build, report, cycle=1, market_summary="rejim=trend_down", attempt=0, max_attempts=3
    )
    assert decision.verdict == "rework"
    assert decision.oos_confirmed is False
    assert any("walk-forward" in d.lower() or "out-of-sample" in d.lower() for d in decision.directives)


def test_archives_when_attempts_exhausted(context: AgentContext, sample_spec) -> None:
    build = context.registry.create_build(sample_spec, cycle=1)
    context.memory.save_strategy_version(build)
    report = report_with(metrics(net_profit_pct=-5.0, win_rate=20.0), oos=metrics(net_profit_pct=-4.0))
    decision = CriticAgent(context).review(
        build, report, cycle=1, market_summary="rejim=range", attempt=3, max_attempts=3
    )
    assert decision.verdict == "archive"
    assert context.memory.get_strategy(build.version)["status"] == "rejected"


def test_error_matrix_forces_rework(context: AgentContext, sample_spec) -> None:
    build = context.registry.create_build(sample_spec, cycle=1)
    context.memory.save_strategy_version(build)
    report = report_with(metrics(net_profit_pct=9.0))
    report.errors.add("run", "KeyError", "atr kolonu yok")
    decision = CriticAgent(context).review(
        build, report, cycle=1, market_summary="rejim=range", attempt=0, max_attempts=2
    )
    assert decision.verdict == "rework"
    assert any("hata matrisi" in d.lower() for d in decision.directives)


def test_lessons_recorded(context: AgentContext, sample_spec) -> None:
    build = context.registry.create_build(sample_spec, cycle=2)
    context.memory.save_strategy_version(build)
    report = report_with(metrics(net_profit_pct=-1.0, win_rate=30.0))
    CriticAgent(context).review(
        build, report, cycle=2, market_summary="rejim=range", attempt=2, max_attempts=2
    )
    lessons = context.memory.lessons_for_cycle(2)
    assert lessons
    assert any(item["scope"] == "review" for item in lessons)


def test_decision_serialization_roundtrip() -> None:
    decision = ReviewDecision(strategy_version="v002", verdict="rework", score=44.4)
    assert ReviewDecision.from_dict(decision.to_dict()).verdict == "rework"
