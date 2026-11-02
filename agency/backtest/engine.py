"""Olay-tabanli (bar bazli) backtest motoru.

Ozellikler:
    * Long-only spot simulasyonu (kaldirac yok), istege bagli short
    * Komisyon (bps) + kayma (slippage bps)
    * ATR tabanli stop-loss / take-profit / trailing stop
    * Zaman stopu (max_holding_bars) ve bekleme suresi (cooldown_bars)
    * Risk yuzdesine gore pozisyon boyutlandirma
    * MAE / MFE takibi ve equity egrisi

Motor, stratejinin yalnizca :class:`BaseStrategy` sozlesmesine uymasini bekler;
uretilen kodun ic detaylarini bilmez.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from agency.backtest.metrics import compute_metrics, equity_sample
from agency.exceptions import BacktestError
from agency.schemas import BacktestMetrics
from agency.strategies.base import BaseStrategy, Position, SignalDecision, TradeRecord, safe_float

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass
class EngineConfig:
    """Simulasyon parametreleri."""

    initial_capital: float = 10000.0
    fee_bps: float = 7.5
    slippage_bps: float = 2.0
    allow_short: bool = False
    bars_per_year: float = 8760.0
    min_bars: int = 120

    @property
    def fee_rate(self) -> float:
        return float(self.fee_bps) / 10000.0

    @property
    def slippage_rate(self) -> float:
        return float(self.slippage_bps) / 10000.0

    @classmethod
    def from_settings(cls, settings: Any, bars_per_year: float = 8760.0) -> "EngineConfig":
        bt = settings.backtest
        return cls(
            initial_capital=float(bt.initial_capital),
            fee_bps=float(bt.fee_bps),
            slippage_bps=float(bt.slippage_bps),
            allow_short=bool(bt.allow_short),
            bars_per_year=bars_per_year,
        )


@dataclass
class BacktestResult:
    """Motorun ham cikisi (BacktestReport'a donusturulur)."""

    metrics: BacktestMetrics
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    bars: int = 0
    exposure_bars: int = 0
    buy_hold_pct: float = 0.0
    total_fees: float = 0.0
    period_start: Optional[str] = None
    period_end: Optional[str] = None

    def equity_sample(self, max_points: int = 300) -> List[List[object]]:
        return equity_sample(self.equity_curve, self.timestamps, max_points=max_points)

    def trade_dicts(self, limit: int = 200) -> List[Dict[str, Any]]:
        return [trade.to_dict() for trade in self.trades[-limit:]]


def _normalize_decision(decision: Any) -> SignalDecision:
    """Strateji ciktisini tek bicime indirger."""
    if isinstance(decision, SignalDecision):
        return decision
    if isinstance(decision, str):
        action = decision.strip().lower()
        return SignalDecision(action if action in {"buy", "sell", "hold"} else "hold", "string")
    if isinstance(decision, Mapping):
        action = str(decision.get("action", "hold")).strip().lower()
        if action not in {"buy", "sell", "hold"}:
            action = "hold"
        return SignalDecision(action, str(decision.get("reason", "")))
    return SignalDecision("hold", "gecersiz cikti")


class BacktestEngine:
    """Bar bazli simulator."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

    # ------------------------------------------------------------------ util
    @staticmethod
    def _warmup_bars(strategy: BaseStrategy, total: int) -> int:
        periods = []
        for key, value in (strategy.params or {}).items():
            if key.endswith("_period") or key.endswith("_slow") or key.endswith("_signal"):
                try:
                    periods.append(int(float(value)))
                except (TypeError, ValueError):
                    continue
        base = max([60] + [p + 5 for p in periods])
        return int(min(max(base, 30), max(30, total // 4)))

    def _validate_input(self, strategy: BaseStrategy, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            raise BacktestError("Backtest icin veri yok")
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise BacktestError(f"Eksik OHLCV kolonlari: {missing}")
        if len(frame) < self.config.min_bars:
            raise BacktestError(f"Yetersiz veri: {len(frame)} bar (min {self.config.min_bars})")
        if not callable(getattr(strategy, "generate_signal", None)):
            raise BacktestError("Strateji generate_signal metodunu saglamiyor")
        if not callable(getattr(strategy, "prepare", None)):
            raise BacktestError("Strateji prepare metodunu saglamiyor")

    # ------------------------------------------------------------------- run
    def run(self, strategy: BaseStrategy, frame: pd.DataFrame) -> BacktestResult:
        """Stratejiyi verilen OHLCV uzerinde simule eder."""
        self._validate_input(strategy, frame)
        prepared = strategy.prepare(frame.copy())
        if not isinstance(prepared, pd.DataFrame):
            raise BacktestError("prepare() DataFrame dondurmeli")
        if "atr" not in prepared.columns:
            raise BacktestError("prepare() 'atr' kolonunu uretmedi (stop-loss icin zorunlu)")
        prepared = prepared.reset_index(drop=True)

        records: List[Dict[str, Any]] = prepared.to_dict("records")
        timestamps = [str(v) for v in prepared["timestamp"].tolist()]
        close = prepared["close"].to_numpy(dtype=float)
        high = prepared["high"].to_numpy(dtype=float)
        low = prepared["low"].to_numpy(dtype=float)
        open_prices = prepared["open"].to_numpy(dtype=float)

        total = len(records)
        warmup = self._warmup_bars(strategy, total)
        fee_rate = self.config.fee_rate
        slip = self.config.slippage_rate

        cash = float(self.config.initial_capital)
        qty = 0.0
        position: Optional[Position] = None
        trades: List[TradeRecord] = []
        equity_curve: List[float] = []
        exposure_bars = 0
        total_fees = 0.0
        cooldown_until = -1

        def close_position(index: int, price: float, reason: str) -> None:
            nonlocal cash, qty, position, total_fees, cooldown_until
            if position is None or qty <= 0:
                return
            exit_price = price * (1.0 - slip)
            gross = qty * exit_price
            fee = gross * fee_rate
            cash += gross - fee
            total_fees += fee
            entry_notional = position.entry_price * qty
            pnl_quote = (exit_price - position.entry_price) * qty - fee - position.fees_paid
            pnl_pct = (pnl_quote / entry_notional * 100.0) if entry_notional else 0.0
            trades.append(
                TradeRecord(
                    entry_ts=position.entry_ts,
                    exit_ts=timestamps[index],
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    qty=qty,
                    pnl_quote=pnl_quote,
                    pnl_pct=pnl_pct,
                    fees_quote=fee + position.fees_paid,
                    exit_reason=reason,
                    bars_held=max(0, index - position.entry_index),
                    mae_pct=(position.lowest_price / position.entry_price - 1.0) * 100.0,
                    mfe_pct=(position.highest_price / position.entry_price - 1.0) * 100.0,
                )
            )
            cooldown_until = index + int(strategy.risk.cooldown_bars)
            qty = 0.0
            position = None

        for index in range(warmup, total):
            row = records[index]
            prev = records[index - 1] if index > 0 else row
            price = float(close[index])
            if not math.isfinite(price) or price <= 0:
                equity_curve.append(cash + qty * max(price, 0.0))
                continue

            # --- acik pozisyon yonetimi -----------------------------------
            # ONEMLI: Once bu barin acilisinda GECERLI olan stop/TP ile cikis
            # kontrolu yapilir. Trailing stop ancak bundan SONRA, bu barin
            # high/low'u ile guncellenir (aksi halde ayni barin high'i ile stop
            # yukseltilip ayni barin low'u ile tetiklenir = ileriye bakma).
            if position is not None:
                exit_reason = None
                exit_price = price
                if float(low[index]) <= position.stop_price:
                    exit_reason = "stop_loss"
                    exit_price = min(position.stop_price, float(open_prices[index]))
                elif position.take_profit_price > 0 and float(high[index]) >= position.take_profit_price:
                    exit_reason = "take_profit"
                    exit_price = max(position.take_profit_price, float(open_prices[index]))
                elif (
                    strategy.risk.max_holding_bars
                    and (index - position.entry_index) >= strategy.risk.max_holding_bars
                ):
                    exit_reason = "max_holding"
                    exit_price = price
                if exit_reason:
                    close_position(index, exit_price, exit_reason)

            if position is not None:
                position.bars_held = index - position.entry_index
                position.highest_price = max(position.highest_price, float(high[index]))
                position.lowest_price = min(position.lowest_price, float(low[index]))
                if strategy.risk.trailing_stop_atr > 0:
                    atr_value = safe_float(row.get("atr"), 0.0)
                    if atr_value > 0:
                        trailed = position.highest_price - atr_value * strategy.risk.trailing_stop_atr
                        position.stop_price = max(position.stop_price, trailed)

            # --- sinyal uretimi -------------------------------------------
            try:
                decision = _normalize_decision(strategy.generate_signal(row, prev, position))
            except Exception as exc:
                raise BacktestError(
                    f"generate_signal hata verdi (bar {index}): {type(exc).__name__}: {exc}",
                    details={"bar_index": index, "ts": timestamps[index]},
                ) from exc

            if position is None and index > cooldown_until and decision.action == "buy" and cash > 0:
                entry_price = price * (1.0 + slip)
                stop_distance = strategy.stop_distance(row, entry_price)
                size = strategy.position_size(cash, entry_price, stop_distance)
                size = min(size, cash / (entry_price * (1.0 + fee_rate)))
                if size > 1e-9:
                    notional = size * entry_price
                    fee = notional * fee_rate
                    if notional + fee <= cash:
                        cash -= notional + fee
                        total_fees += fee
                        qty = size
                        tp_distance = strategy.take_profit_distance(row, entry_price)
                        position = Position(
                            entry_index=index,
                            entry_ts=timestamps[index],
                            entry_price=entry_price,
                            qty=size,
                            stop_price=max(0.0, entry_price - stop_distance),
                            take_profit_price=entry_price + tp_distance if tp_distance > 0 else 0.0,
                            highest_price=entry_price,
                            lowest_price=entry_price,
                            fees_paid=fee,
                            reason=decision.reason,
                        )
                        strategy.on_position_opened(position)
            elif position is not None and decision.action == "sell":
                close_position(index, price, f"signal:{decision.reason[:60] or 'exit'}")

            equity_curve.append(cash + qty * price)
            if qty > 0:
                exposure_bars += 1

        if position is not None:
            close_position(total - 1, float(close[-1]), "end_of_data")
            if equity_curve:
                equity_curve[-1] = cash

        buy_hold_pct = 0.0
        if total > warmup and close[warmup] > 0:
            buy_hold_pct = (close[-1] / close[warmup] - 1.0) * 100.0

        metrics = compute_metrics(
            trades,
            equity_curve,
            initial_capital=float(self.config.initial_capital),
            bars=max(1, total - warmup),
            exposure_bars=exposure_bars,
            buy_hold_pct=buy_hold_pct,
            total_fees=total_fees,
            bars_per_year=self.config.bars_per_year,
        )

        return BacktestResult(
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            timestamps=timestamps[warmup:],
            bars=max(0, total - warmup),
            exposure_bars=exposure_bars,
            buy_hold_pct=buy_hold_pct,
            total_fees=total_fees,
            period_start=timestamps[warmup] if total > warmup else None,
            period_end=timestamps[-1] if timestamps else None,
        )


