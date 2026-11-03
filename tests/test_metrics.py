"""Metrik ve skorlama testleri."""

from __future__ import annotations

import pytest

from agency.backtest.metrics import (
    compute_metrics,
    equity_sample,
    evaluate_gates,
    max_drawdown_pct,
    score_metrics,
    sharpe_ratio,
)
from agency.strategies.base import TradeRecord

WEIGHTS = {
    "net_profit": 0.35,
    "win_rate": 0.20,
    "drawdown": 0.20,
    "sharpe": 0.15,
    "profit_factor": 0.10,
}
TARGETS = {
    "min_net_profit_pct": 2.0,
    "min_win_rate": 40.0,
    "max_drawdown_pct": 20.0,
    "min_sharpe": 0.25,
    "min_profit_factor": 1.10,
    "min_trades": 8,
}


def trade(pnl_pct: float, pnl_quote: float, bars: int = 5) -> TradeRecord:
    return TradeRecord(
        entry_ts="2024-01-01",
        exit_ts="2024-01-02",
        entry_price=100.0,
        exit_price=100.0 * (1 + pnl_pct / 100),
        qty=1.0,
        pnl_quote=pnl_quote,
        pnl_pct=pnl_pct,
        fees_quote=0.1,
        exit_reason="test",
        bars_held=bars,
    )


def test_max_drawdown() -> None:
    assert max_drawdown_pct([100, 120, 90, 110]) == pytest.approx(25.0)
    assert max_drawdown_pct([100, 110, 120]) == 0.0
    assert max_drawdown_pct([]) == 0.0


def test_sharpe_sign() -> None:
    rising = [100 * (1.001 ** i) for i in range(200)]
    falling = [100 * (0.999 ** i) for i in range(200)]
    assert sharpe_ratio(rising) > 0
    assert sharpe_ratio(falling) < 0
    assert sharpe_ratio([100.0, 100.0, 100.0]) == 0.0


def test_compute_metrics_values() -> None:
    trades = [trade(5.0, 50.0), trade(-2.0, -20.0), trade(3.0, 30.0), trade(-1.0, -10.0)]
    equity = [10_000 + i for i in range(100)]
    metrics = compute_metrics(
        trades,
        equity,
        initial_capital=10_000.0,
        bars=100,
        exposure_bars=50,
        buy_hold_pct=1.5,
        total_fees=1.2,
    )
    assert metrics.trades == 4
    assert metrics.winning_trades == 2 and metrics.losing_trades == 2
    assert metrics.win_rate == pytest.approx(50.0)
    assert metrics.profit_factor == pytest.approx(80.0 / 30.0, rel=1e-3)
    assert metrics.exposure_pct == pytest.approx(50.0)
    assert metrics.buy_hold_pct == 1.5


def test_profit_factor_capped_when_no_losses() -> None:
    metrics = compute_metrics(
        [trade(3.0, 30.0)],
        [10_000, 10_030],
        initial_capital=10_000.0,
        bars=2,
        exposure_bars=1,
        buy_hold_pct=0.0,
        total_fees=0.0,
    )
    assert metrics.profit_factor > 1.0


def test_evaluate_gates() -> None:
    good = {
        "net_profit_pct": 5.0,
        "win_rate": 55.0,
        "max_drawdown_pct": 8.0,
        "sharpe": 1.2,
        "profit_factor": 1.8,
        "trades": 20,
    }
    gates = evaluate_gates(good, TARGETS)
    assert all(gates.values())

    bad = dict(good, max_drawdown_pct=45.0, trades=2)
    gates = evaluate_gates(bad, TARGETS)
    assert gates["drawdown"] is False and gates["trades"] is False


def test_score_ordering() -> None:
    strong = {
        "net_profit_pct": 6.0,
        "win_rate": 60.0,
        "max_drawdown_pct": 4.0,
        "sharpe": 1.5,
        "profit_factor": 2.0,
        "trades": 25,
    }
    weak = {
        "net_profit_pct": -1.0,
        "win_rate": 25.0,
        "max_drawdown_pct": 18.0,
        "sharpe": -1.0,
        "profit_factor": 0.5,
        "trades": 25,
    }
    strong_score, components = score_metrics(strong, weights=WEIGHTS, targets=TARGETS)
    weak_score, _ = score_metrics(weak, weights=WEIGHTS, targets=TARGETS)
    assert strong_score > weak_score
    assert 0.0 <= strong_score <= 100.0
    assert set(components) == {"net_profit", "win_rate", "drawdown", "sharpe", "profit_factor"}


def test_score_penalises_few_trades() -> None:
    metrics = {
        "net_profit_pct": 6.0,
        "win_rate": 60.0,
        "max_drawdown_pct": 4.0,
        "sharpe": 1.5,
        "profit_factor": 2.0,
        "trades": 2,
    }
    score, _ = score_metrics(metrics, weights=WEIGHTS, targets=TARGETS)
    metrics["trades"] = 25
    score_many, _ = score_metrics(metrics, weights=WEIGHTS, targets=TARGETS)
    assert score < score_many


def test_equity_sample_downsampling() -> None:
    curve = [10_000 + i for i in range(1000)]
    stamps = [f"t{i}" for i in range(1000)]
    sample = equity_sample(curve, stamps, max_points=50)
    assert 40 <= len(sample) <= 60
    assert sample[0][1] == 10_000
    assert sample[-1][1] == 10_999
