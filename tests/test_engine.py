"""Backtest motoru testleri (deterministik senaryolar)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agency.backtest.engine import BacktestEngine, EngineConfig, _normalize_decision
from agency.exceptions import BacktestError
from agency.strategies.base import BaseStrategy, RiskConfig


class StubStrategy(BaseStrategy):
    """Test icin kontrol edilebilir strateji."""

    name = "stub"
    family = "hybrid"
    risk_defaults = {
        "stop_loss_atr": 2.0,
        "take_profit_atr": 1.0,
        "trailing_stop_atr": 0.0,
        "risk_per_trade_pct": 50.0,
        "cooldown_bars": 0,
        "max_holding_bars": 0,
        "allow_short": False,
    }
    entry_index = 100

    @classmethod
    def default_params(cls):
        return {}

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["atr"] = 1.0
        out["entry"] = 0
        out.loc[out.index == self.entry_index, "entry"] = 1
        return out

    def generate_signal(self, row, prev, position):
        if position is None and int(row.get("entry", 0)) == 1:
            return {"action": "buy", "reason": "stub entry"}
        return {"action": "hold", "reason": "stub"}


def make_frame(prices, highs=None, lows=None) -> pd.DataFrame:
    prices = np.asarray(prices, dtype=float)
    open_ = np.concatenate([[prices[0]], prices[:-1]])
    high = np.asarray(highs, dtype=float) if highs is not None else np.maximum(open_, prices)
    low = np.asarray(lows, dtype=float) if lows is not None else np.minimum(open_, prices)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(prices), freq="1h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": prices,
            "volume": np.full(len(prices), 10.0),
        }
    )


def config(**overrides) -> EngineConfig:
    base = dict(initial_capital=10_000.0, fee_bps=10.0, slippage_bps=0.0, min_bars=120)
    base.update(overrides)
    return EngineConfig(**base)


def test_take_profit_exit() -> None:
    prices = [100.0] * 300
    prices[101] = 106.0  # TP 1 ATR(1.0) = 101 -> tetiklenir
    frame = make_frame(prices)
    result = BacktestEngine(config()).run(StubStrategy(), frame)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason in {"take_profit", "end_of_data"}
    assert trade.pnl_quote > 0
    assert result.metrics.trades == 1


def test_fees_reduce_profit() -> None:
    prices = [100.0] * 300
    frame = make_frame(prices)
    result = BacktestEngine(config(fee_bps=20.0, initial_capital=10_000.0)).run(
        StubStrategy(), frame
    )
    assert result.trades
    assert result.metrics.total_fees_quote > 0
    assert result.metrics.net_profit_pct < 0  # ayni fiyattan girip cikinca maliyet


def test_trailing_stop_does_not_trigger_on_same_bar() -> None:
    """Look-ahead regresyonu: trailing stop giris barinda tetiklenmemeli."""
    strategy = StubStrategy()
    strategy.risk = RiskConfig(
        stop_loss_atr=2.0,
        take_profit_atr=1.0,
        trailing_stop_atr=1.0,
        risk_per_trade_pct=50.0,
        max_holding_bars=0,
    )
    prices = [100.0] * 300
    highs = [100.0] * 300
    lows = [100.0] * 300
    highs[100] = 130.0  # giris barinda devasa high
    lows[100] = 95.0    # giris barinda devasa low
    highs[101] = 100.5
    lows[101] = 100.4
    frame = make_frame(prices, highs, lows)
    result = BacktestEngine(config()).run(strategy, frame)
    assert result.trades
    first = result.trades[0]
    assert first.bars_held >= 1
    assert first.exit_reason != "stop_loss" or first.exit_ts != first.entry_ts


def test_max_holding_forces_exit() -> None:
    strategy = StubStrategy()
    strategy.risk = RiskConfig(
        stop_loss_atr=2.0,
        take_profit_atr=0.0,
        trailing_stop_atr=0.0,
        risk_per_trade_pct=50.0,
        max_holding_bars=5,
    )
    frame = make_frame([100.0] * 300)
    result = BacktestEngine(config()).run(strategy, frame)
    assert result.trades[0].exit_reason == "max_holding"
    assert result.trades[0].bars_held == 5


def test_stop_loss_exit() -> None:
    strategy = StubStrategy()
    strategy.risk = RiskConfig(
        stop_loss_atr=2.0,
        take_profit_atr=0.0,
        trailing_stop_atr=0.0,
        risk_per_trade_pct=50.0,
        max_holding_bars=0,
    )
    prices = [100.0] * 300
    prices[101] = 97.0
    frame = make_frame(prices)
    result = BacktestEngine(config()).run(strategy, frame)
    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].pnl_quote < 0


def test_no_more_than_one_position_and_cash_conserved() -> None:
    frame = make_frame([100.0] * 300)
    result = BacktestEngine(config()).run(StubStrategy(), frame)
    assert len(result.trades) == 1
    assert result.equity_curve
    assert max(result.equity_curve) < 12_000  # kaldirac yok


def test_insufficient_data_raises() -> None:
    frame = make_frame([100.0] * 50)
    with pytest.raises(BacktestError):
        BacktestEngine(config()).run(StubStrategy(), frame)


def test_missing_atr_raises() -> None:
    class NoAtr(StubStrategy):
        def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame.copy()

    with pytest.raises(BacktestError):
        BacktestEngine(config()).run(NoAtr(), make_frame([100.0] * 300))


def test_normalize_decision_variants() -> None:
    assert _normalize_decision({"action": "BUY"}).action == "buy"
    assert _normalize_decision("sell").action == "sell"
    assert _normalize_decision({"action": "long"}).action == "hold"
    assert _normalize_decision(None).action == "hold"


def test_metrics_sane_for_flat_market() -> None:
    frame = make_frame([100.0] * 300)
    result = BacktestEngine(config()).run(StubStrategy(), frame)
    metrics = result.metrics
    assert metrics.bars > 0
    assert 0.0 <= metrics.win_rate <= 100.0
    assert metrics.max_drawdown_pct >= 0.0
    assert metrics.buy_hold_pct == pytest.approx(0.0, abs=1e-9)
