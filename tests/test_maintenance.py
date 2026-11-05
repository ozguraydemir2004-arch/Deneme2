"""Yeniden dogrulama (decay check) testleri."""

from __future__ import annotations

from pathlib import Path

from agency.maintenance import StrategyMaintenance
from agency.memory.repositories import MemoryStore
from agency.strategies.registry import StrategyRegistry


def promote(sample_spec, memory: MemoryStore, registry: StrategyRegistry):
    build = registry.create_build(sample_spec, cycle=1)
    memory.save_strategy_version(build, status="promoted", score=90.0)
    memory.update_strategy_result(
        build.version, status="promoted", score=90.0, metrics={"net_profit_pct": 6.0}
    )
    memory.promote_strategy(
        build.version,
        cycle=1,
        regime="range",
        score=90.0,
        symbol="BTC/USDT",
        timeframe="1h",
    )
    return build


def loosen(settings) -> None:
    settings.targets.min_net_profit_pct = -1000.0
    settings.targets.min_win_rate = 0.0
    settings.targets.max_drawdown_pct = 100.0
    settings.targets.min_sharpe = -1000.0
    settings.targets.min_profit_factor = 0.0
    settings.backtest.min_trades = 2


def test_revalidate_writes_report(settings, memory, registry, sample_spec) -> None:
    loosen(settings)
    build = promote(sample_spec, memory, registry)

    maintenance = StrategyMaintenance(settings, memory=memory, registry=registry)
    try:
        results = maintenance.revalidate(limit=5)
    finally:
        maintenance.close()

    assert len(results) == 1
    assert results[0].version == build.version
    assert results[0].error is None
    assert results[0].ok is True
    assert results[0].metrics.bars > 0

    report = Path(settings.runtime.reports_dir) / "REVALIDATION.md"
    assert report.is_file()
    assert "Yeniden Dogrulama" in report.read_text(encoding="utf-8")
    # Demote kapali oldugu icin hafiza korunur
    assert [row["version"] for row in memory.memory_strategies(limit=5)] == [build.version]


def test_revalidate_demotes_decayed_strategy(settings, memory, registry, sample_spec) -> None:
    # Ulasilamaz esikler -> her kosu basarisiz
    settings.targets.min_net_profit_pct = 1_000_000.0
    settings.backtest.min_trades = 2
    build = promote(sample_spec, memory, registry)

    maintenance = StrategyMaintenance(settings, memory=memory, registry=registry)
    try:
        results = maintenance.revalidate(limit=5, demote=True)
    finally:
        maintenance.close()

    assert results[0].ok is False
    assert results[0].demoted is True
    assert memory.memory_strategies(limit=5) == []
    assert memory.get_strategy(build.version)["status"] == "demoted"
    lessons = [row for row in memory.recent_lessons(limit=5) if row["scope"] == "maintenance"]
    assert lessons and "kalici hafizadan cikarildi" in lessons[0]["message"]


def test_revalidate_without_promotions(settings, memory, registry) -> None:
    maintenance = StrategyMaintenance(settings, memory=memory, registry=registry)
    try:
        results = maintenance.revalidate(limit=5, demote=True)
    finally:
        maintenance.close()
    assert results == []
    report = Path(settings.runtime.reports_dir) / "REVALIDATION.md"
    assert report.is_file()
    assert "terfi etmis strateji yok" in report.read_text(encoding="utf-8")


def test_revalidate_handles_broken_code_path(settings, memory, registry, sample_spec) -> None:
    loosen(settings)
    build = promote(sample_spec, memory, registry)
    strategy = memory.get_strategy(build.version)
    memory.update_strategy_result(
        build.version,
        status="promoted",
        score=90.0,
        metrics=strategy.get("metrics") or {},
    )
    # Kod yolunu boz: denetim cokmemeli, HATA olarak raporlanmali
    memory.conn.execute(
        "UPDATE strategy_versions SET code_path=? WHERE version=?",
        (str(Path(settings.versioning.library_dir) / "yok.py"), build.version),
    )
    memory.conn.commit()

    maintenance = StrategyMaintenance(settings, memory=memory, registry=registry)
    try:
        results = maintenance.revalidate(limit=5)
    finally:
        maintenance.close()

    assert results and results[0].ok is False
    assert results[0].error is not None
