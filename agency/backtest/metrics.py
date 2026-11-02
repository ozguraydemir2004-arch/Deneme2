"""Backtest performans metrikleri.

Tum metrikler tek bir yerden hesaplanir; Backtest Agent ve Critic Agent ayni
sayilari gorur (tek dogruluk kaynagi).
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

import numpy as np

from agency.schemas import BacktestMetrics
from agency.strategies.base import TradeRecord

#: Sonsuz profit factor'u kirpmak icin ust sinir
PROFIT_FACTOR_CAP = 99.0


def max_drawdown_pct(equity_curve: Sequence[float]) -> float:
    """Zirve-dip bazli maksimum dusus (yuzde, pozitif sayi)."""
    peak = -math.inf
    worst = 0.0
    for value in equity_curve:
        if value is None or not math.isfinite(float(value)):
            continue
        value = float(value)
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak * 100.0)
    return worst


def _bar_returns(equity_curve: Sequence[float]) -> np.ndarray:
    values = np.asarray([float(v) for v in equity_curve if v is not None], dtype=np.float64)
    if values.size < 3:
        return np.zeros(0)
    previous = values[:-1]
    current = values[1:]
    valid = previous > 0
    return np.where(valid, current / np.where(valid, previous, 1.0) - 1.0, 0.0)


def sharpe_ratio(equity_curve: Sequence[float], *, bars_per_year: float = 8760.0) -> float:
    returns = _bar_returns(equity_curve)
    if returns.size < 3:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(max(1.0, bars_per_year)))


def sortino_ratio(equity_curve: Sequence[float], *, bars_per_year: float = 8760.0) -> float:
    returns = _bar_returns(equity_curve)
    if returns.size < 3:
        return 0.0
    downside = returns[returns < 0]
    if downside.size == 0:
        return 0.0
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else abs(float(downside.mean()))
    if downside_std <= 0:
        return 0.0
    return float(np.mean(returns) / downside_std * math.sqrt(max(1.0, bars_per_year)))


def compute_metrics(
    trades: Sequence[TradeRecord],
    equity_curve: Sequence[float],
    *,
    initial_capital: float,
    bars: int,
    exposure_bars: int,
    buy_hold_pct: float,
    total_fees: float,
    bars_per_year: float = 8760.0,
) -> BacktestMetrics:
    """Islem listesi ve equity egrisinden tam metrik seti uretir."""
    final_equity = float(equity_curve[-1]) if equity_curve else float(initial_capital)
    pnl_values = [float(t.pnl_pct) for t in trades]
    wins = [t for t in trades if t.pnl_quote > 0]
    losses = [t for t in trades if t.pnl_quote <= 0]

    gross_profit = sum(float(t.pnl_quote) for t in wins)
    gross_loss = abs(sum(float(t.pnl_quote) for t in losses))
    if gross_loss > 0:
        profit_factor = min(gross_profit / gross_loss, PROFIT_FACTOR_CAP)
    elif gross_profit > 0:
        profit_factor = PROFIT_FACTOR_CAP
    else:
        profit_factor = 0.0

    trade_count = len(trades)
    win_rate = (len(wins) / trade_count * 100.0) if trade_count else 0.0
    avg_trade = float(np.mean(pnl_values)) if pnl_values else 0.0
    avg_win = float(np.mean([float(t.pnl_pct) for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([float(t.pnl_pct) for t in losses])) if losses else 0.0
    avg_holding = float(np.mean([t.bars_held for t in trades])) if trades else 0.0

    return BacktestMetrics(
        net_profit_pct=(final_equity / initial_capital - 1.0) * 100.0 if initial_capital else 0.0,
        net_profit_quote=final_equity - initial_capital,
        final_equity=final_equity,
        win_rate=win_rate,
        max_drawdown_pct=max_drawdown_pct(equity_curve),
        sharpe=sharpe_ratio(equity_curve, bars_per_year=bars_per_year),
        sortino=sortino_ratio(equity_curve, bars_per_year=bars_per_year),
        profit_factor=profit_factor,
        trades=trade_count,
        winning_trades=len(wins),
        losing_trades=len(losses),
        avg_trade_pct=avg_trade,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        expectancy_pct=avg_trade,
        total_fees_quote=float(total_fees),
        exposure_pct=(exposure_bars / bars * 100.0) if bars else 0.0,
        avg_holding_bars=avg_holding,
        buy_hold_pct=buy_hold_pct,
        bars=int(bars),
    )


def evaluate_gates(metrics: Mapping[str, float], targets: Mapping[str, float]) -> Dict[str, bool]:
    """Hedef esiklerine karsi gecis/kalis tablosu."""
    return {
        "net_profit": float(metrics.get("net_profit_pct", 0.0)) >= float(targets["min_net_profit_pct"]),
        "win_rate": float(metrics.get("win_rate", 0.0)) >= float(targets["min_win_rate"]),
        "drawdown": float(metrics.get("max_drawdown_pct", 100.0)) <= float(targets["max_drawdown_pct"]),
        "sharpe": float(metrics.get("sharpe", 0.0)) >= float(targets["min_sharpe"]),
        "profit_factor": float(metrics.get("profit_factor", 0.0)) >= float(targets["min_profit_factor"]),
        "trades": int(metrics.get("trades", 0)) >= int(targets.get("min_trades", 1)),
    }


def score_metrics(
    metrics: Mapping[str, float],
    *,
    weights: Mapping[str, float],
    targets: Mapping[str, float],
) -> tuple[float, Dict[str, float]]:
    """0-100 arasi normalize skor + bilesen dokumu.

    Her bilesen hedefe gore normalize edilir; esik asilirsa 1.0'da doygunlasir.
    """
    net = float(metrics.get("net_profit_pct", 0.0))
    win = float(metrics.get("win_rate", 0.0))
    dd = float(metrics.get("max_drawdown_pct", 100.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))

    net_component = min(max(net / max(float(targets["min_net_profit_pct"]), 1e-9), 0.0), 2.0) / 2.0
    win_component = min(max(win / max(float(targets["min_win_rate"]), 1e-9), 0.0), 1.5) / 1.5
    dd_target = max(float(targets["max_drawdown_pct"]), 1e-9)
    dd_component = 1.0 - min(dd / dd_target, 1.5) / 1.5
    sharpe_component = min(max(sharpe / max(float(targets["min_sharpe"]), 1e-9), 0.0), 2.0) / 2.0
    pf_component = (
        min(max((pf - 0.8) / max(float(targets["min_profit_factor"]) - 0.8, 1e-9), 0.0), 1.5) / 1.5
    )

    components = {
        "net_profit": net_component * 100.0 * float(weights["net_profit"]),
        "win_rate": win_component * 100.0 * float(weights["win_rate"]),
        "drawdown": dd_component * 100.0 * float(weights["drawdown"]),
        "sharpe": sharpe_component * 100.0 * float(weights["sharpe"]),
        "profit_factor": pf_component * 100.0 * float(weights["profit_factor"]),
    }
    total = sum(components.values())
    if int(metrics.get("trades", 0)) < int(targets.get("min_trades", 1)):
        total *= 0.6  # yetersiz orneklem cezasi
    return float(min(max(total, 0.0), 100.0)), components


def equity_sample(
    equity_curve: Sequence[float],
    timestamps: Sequence[str],
    max_points: int = 300,
) -> List[List[object]]:
    """Equity egrisini rapor icin seyreltir."""
    size = len(equity_curve)
    if size == 0:
        return []
    step = 1 if size <= max_points else max(1, size // max_points)
    sample: List[List[object]] = []
    for index in range(0, size, step):
        sample.append(
            [
                str(timestamps[index]) if index < len(timestamps) else str(index),
                round(float(equity_curve[index]), 2),
            ]
        )
    if sample and sample[-1][1] != round(float(equity_curve[-1]), 2):
        sample.append(
            [
                str(timestamps[-1]) if timestamps else str(size - 1),
                round(float(equity_curve[-1]), 2),
            ]
        )
    return sample

