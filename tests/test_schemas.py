"""Veri sozlesmeleri (schemas) testleri."""

from __future__ import annotations

import pytest

from agency.exceptions import StrategyValidationError
from agency.schemas import (
    BacktestMetrics,
    BacktestReport,
    CycleResult,
    ErrorMatrix,
    IndicatorSpec,
    MarketContext,
    ResearchReport,
    ReviewDecision,
    Rule,
    StrategySpec,
)


def test_valid_spec_passes(sample_spec: StrategySpec) -> None:
    assert sample_spec.validate() == []
    assert sample_spec.ensure_valid() is sample_spec


def test_invalid_operator_rejected(sample_spec: StrategySpec) -> None:
    sample_spec.entry_long[0].op = "~>"
    errors = sample_spec.validate()
    assert any("operator" in error for error in errors)


def test_unknown_operand_rejected(sample_spec: StrategySpec) -> None:
    sample_spec.entry_long[0].right = "not_an_indicator"
    errors = sample_spec.validate()
    assert any("bilinmeyen sag operand" in error for error in errors)


def test_indicator_param_range_enforced(sample_spec: StrategySpec) -> None:
    sample_spec.indicators[0].params = {"period": 5000}
    errors = sample_spec.validate()
    assert any("aralik disi" in error for error in errors)


def test_duplicate_indicator_keys_rejected(sample_spec: StrategySpec) -> None:
    sample_spec.indicators.append(
        IndicatorSpec(key="ema_fast", kind="ema", params={"period": 9})
    )
    errors = sample_spec.validate()
    assert any("yinelenen" in error for error in errors)


def test_previous_bar_operand_allowed(sample_spec: StrategySpec) -> None:
    sample_spec.indicators.append(
        IndicatorSpec(
            key="bb_width", kind="bbands", output="width", params={"period": 20, "std": 2.0}
        )
    )
    sample_spec.filters.append(Rule(left="bb_width_prev", op="<", right=4.0))
    errors = sample_spec.validate()
    assert not any("bilinmeyen" in error for error in errors)


def test_ensure_valid_raises_with_details(sample_spec: StrategySpec) -> None:
    sample_spec.entry_long = []
    with pytest.raises(StrategyValidationError) as excinfo:
        sample_spec.ensure_valid()
    assert "entry_long" in str(excinfo.value)
    assert excinfo.value.details["errors"]


def test_roundtrip_preserves_canonical(sample_spec: StrategySpec) -> None:
    payload = sample_spec.to_dict()
    restored = StrategySpec.from_dict(payload)
    assert restored.canonical() == sample_spec.canonical()
    assert restored.fingerprint() == sample_spec.fingerprint()


def test_fingerprint_changes_with_risk(sample_spec: StrategySpec) -> None:
    before = sample_spec.fingerprint()
    sample_spec.stop_loss_atr = 3.5
    assert sample_spec.fingerprint() != before


def test_with_mutation_returns_new_spec(sample_spec: StrategySpec) -> None:
    other = sample_spec.with_mutation(take_profit_atr=6.0)
    assert other.take_profit_atr == 6.0
    assert sample_spec.take_profit_atr == 4.0


def test_describe_contains_key_parts(sample_spec: StrategySpec) -> None:
    text = sample_spec.describe()
    assert "ema_fast" in text and "trend_following" in text


def test_indicator_resolved_params_types() -> None:
    indicator = IndicatorSpec(key="bb", kind="bbands", output="width", params={"period": 20})
    resolved = indicator.resolved_params()
    assert resolved["period"] == 20 and resolved["std"] == 2.0


def test_backtest_report_roundtrip() -> None:
    report = BacktestReport(
        strategy_version="v001",
        name="x",
        status="ok",
        metrics=BacktestMetrics(net_profit_pct=5.0, trades=10, win_rate=50.0),
        oos_metrics=BacktestMetrics(net_profit_pct=1.0),
        fold_metrics=[BacktestMetrics(net_profit_pct=0.5), BacktestMetrics(net_profit_pct=-0.2)],
    )
    restored = BacktestReport.from_dict(report.to_dict())
    assert restored.metrics.net_profit_pct == 5.0
    assert len(restored.fold_metrics) == 2
    assert restored.oos_metrics.net_profit_pct == 1.0


def test_error_matrix_summary() -> None:
    matrix = ErrorMatrix()
    assert not matrix.has_errors and matrix.summary == "hata yok"
    matrix.add("run", "KeyError", "atr kolonu yok", file="x.py", line=12)
    assert matrix.has_errors
    assert "KeyError" in matrix.summary


def test_review_and_cycle_roundtrip() -> None:
    decision = ReviewDecision(strategy_version="v001", verdict="promote", score=77.7)
    assert ReviewDecision.from_dict(decision.to_dict()).score == 77.7

    result = CycleResult(cycle=3, status="completed", promoted=["v001"])
    restored = CycleResult.from_dict(result.to_dict())
    assert restored.cycle == 3 and restored.promoted == ["v001"]


def test_research_report_roundtrip(sample_spec: StrategySpec) -> None:
    market = MarketContext(
        symbol="BTC/USDT",
        timeframe="1h",
        data_source="synthetic",
        bars=900,
        last_price=50000.0,
        regime="range",
    )
    report = ResearchReport(cycle=1, market=market, candidates=[sample_spec], snapshot_count=5)
    restored = ResearchReport.from_dict(report.to_dict())
    assert restored.market.regime == "range"
    assert restored.candidates[0].fingerprint() == sample_spec.fingerprint()
