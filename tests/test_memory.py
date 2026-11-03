"""Hafiza katmani (SQLite + vektor) testleri."""

from __future__ import annotations

from agency.memory.database import Database
from agency.memory.repositories import MemoryStore
from agency.memory.vector_store import FeatureVectorizer
from agency.schemas import (
    BacktestMetrics,
    BacktestReport,
    CycleResult,
    MarketContext,
    ResearchReport,
    ReviewDecision,
    StrategyBuild,
)
from agency.strategies.registry import StrategyRegistry


def build_object(registry: StrategyRegistry, spec, cycle: int = 1) -> StrategyBuild:
    return registry.create_build(spec, cycle=cycle)


def test_schema_is_idempotent(settings) -> None:
    first = Database(settings.memory.database)
    first.connection
    first.close()
    second = Database(settings.memory.database)
    counts = second.table_counts()
    second.close()
    assert counts["cycles"] == 0
    assert "strategy_versions" in counts


def test_cycle_roundtrip(memory: MemoryStore) -> None:
    result = CycleResult(
        cycle=1,
        status="completed",
        best_version="v001",
        best_score=61.5,
        promoted=["v001"],
        lessons=["ders 1"],
        next_directives=["stop_loss_atr 2.0 -> 1.5"],
    )
    memory.save_cycle(result, run_id="run-1", symbol="BTC/USDT", timeframe="1h")
    stored = memory.get_cycle(1)
    assert stored["status"] == "completed"
    assert stored["promoted"] == ["v001"]
    assert stored["next_directives"] == ["stop_loss_atr 2.0 -> 1.5"]
    assert memory.next_cycle_number() == 2
    assert memory.last_cycle()["cycle"] == 1
    assert memory.list_cycles(limit=5)[0]["best_version"] == "v001"


def test_research_and_news(memory: MemoryStore, sample_spec) -> None:
    market = MarketContext(
        symbol="BTC/USDT",
        timeframe="1h",
        data_source="synthetic",
        bars=900,
        last_price=100.0,
        regime="range",
    )
    report = ResearchReport(cycle=1, market=market, candidates=[sample_spec])
    memory.save_research(report)
    latest = memory.latest_research()
    assert latest["market"]["regime"] == "range"

    saved = memory.save_news(
        1,
        [
            {"title": "ETF approval boosts bitcoin", "sentiment": "bullish", "keywords": ["etf"]},
            {"title": "Exchange hacked", "sentiment": "bearish", "keywords": []},
        ],
    )
    assert saved == 2
    assert len(memory.recent_news(limit=5)) == 2


def test_snapshot_similarity_search(memory: MemoryStore) -> None:
    base = {"rsi": 30.0, "adx": 15.0, "atr_pct": 1.0, "bb_width": 2.0}
    rows = [
        {"ts": "t1", "bar_index": 1, "close": 100.0, "forward_return_pct": 2.0,
         "label": "winner", "features": base},
        {"ts": "t2", "bar_index": 2, "close": 101.0, "forward_return_pct": -2.0,
         "label": "loser", "features": {**base, "rsi": 75.0, "adx": 40.0}},
    ]
    assert memory.save_snapshots(1, "BTC/USDT", "1h", rows) == 2
    query = memory.vectorizer.embed_dict(base)
    matches = memory.similar_snapshots(query, top_k=2, label="winner")
    assert matches and matches[0]["label"] == "winner"
    assert matches[0]["similarity"] > 0.9

    stats = memory.snapshot_stats()
    assert stats["total"] == 2 and stats["winners"] == 1


def test_strategy_lifecycle(
    memory: MemoryStore, registry: StrategyRegistry, sample_spec
) -> None:
    build = build_object(registry, sample_spec)
    memory.save_strategy_version(build, status="draft")
    stored = memory.get_strategy(build.version)
    assert stored["status"] == "draft"
    assert stored["spec"]["name"] == sample_spec.name

    decision = ReviewDecision(strategy_version=build.version, verdict="promote", score=81.0)
    memory.update_strategy_result(
        build.version,
        status="promoted",
        score=81.0,
        metrics={"net_profit_pct": 5.0},
        review=decision,
    )
    memory.save_review(decision, cycle=1)
    updated = memory.get_strategy(build.version)
    assert updated["status"] == "promoted" and updated["score"] == 81.0
    assert memory.best_strategy()["version"] == build.version
    assert memory.review_history(build.version)[0]["verdict"] == "promote"

    memory.promote_strategy(
        build.version,
        cycle=1,
        regime="range",
        score=81.0,
        symbol="BTC/USDT",
        timeframe="1h",
    )
    memory.mark_memory_reused(build.version)
    elite = memory.memory_strategies(limit=5)
    assert elite and elite[0]["version"] == build.version
    assert elite[0]["times_reused"] == 1
    assert elite[0]["spec"]["family"] == sample_spec.family


def test_backtest_and_lessons(
    memory: MemoryStore, registry: StrategyRegistry, sample_spec
) -> None:
    build = build_object(registry, sample_spec)
    memory.save_strategy_version(build)
    report = BacktestReport(
        strategy_version=build.version,
        name=build.name,
        metrics=BacktestMetrics(net_profit_pct=3.0),
    )
    memory.save_backtest(report, cycle=1, report_path="x.json")
    history = memory.backtest_history(build.version)
    assert history[0]["metrics"]["net_profit_pct"] == 3.0

    memory.add_lesson(1, "drawdown asimi", directive="stop_loss_atr 2.0 -> 1.5")
    assert memory.recent_lessons()[0]["directive"] == "stop_loss_atr 2.0 -> 1.5"
    assert memory.pending_directives(limit=1) == ["stop_loss_atr 2.0 -> 1.5"]
    assert memory.lessons_for_cycle(1)[0]["message"] == "drawdown asimi"

    stats = memory.stats()
    assert stats["versions"] == 1 and stats["lessons"] == 1


def test_vectorizer_is_deterministic() -> None:
    vectorizer = FeatureVectorizer(dim=32)
    features = {"rsi": 44.0, "adx": 21.0, "atr_pct": 1.2, "rel_volume": 1.4}
    first = vectorizer.embed_dict(features)
    second = vectorizer.embed_dict(features)
    assert first == second
    similarity = vectorizer.cosine(first, second)
    assert similarity > 0.999
    assert len(first) == 32
