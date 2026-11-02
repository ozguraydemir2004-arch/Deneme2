"""Strateji temel siniflari (OOP sozlesmesi).

Uretilen her strateji bu moduldeki :class:`BaseStrategy` sinifindan turer.
Backtest motoru yalnizca bu sozlesmeye guvenir:

    prepare(df)                     -> indikator kolonlari eklenmis DataFrame
    generate_signal(row, prev, pos) -> {"action": "buy"|"sell"|"hold"}
    risk_defaults                   -> SL/TP/boyut/limit ayarlari

Boylece LLM veya derleyici tarafindan uretilen kod, motorun geri kalanindan
izole edilir ve dogrulanabilir kalir.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import pandas as pd

ACTIONS = {"buy", "sell", "hold"}


@dataclass
class RiskConfig:
    """Strateji bazli risk parametreleri."""

    stop_loss_atr: float = 2.0
    take_profit_atr: float = 3.0
    trailing_stop_atr: float = 0.0
    risk_per_trade_pct: float = 1.5
    cooldown_bars: int = 0
    max_holding_bars: int = 0
    allow_short: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stop_loss_atr": float(self.stop_loss_atr),
            "take_profit_atr": float(self.take_profit_atr),
            "trailing_stop_atr": float(self.trailing_stop_atr),
            "risk_per_trade_pct": float(self.risk_per_trade_pct),
            "cooldown_bars": int(self.cooldown_bars),
            "max_holding_bars": int(self.max_holding_bars),
            "allow_short": bool(self.allow_short),
        }


@dataclass
class Position:
    """Acik pozisyon durumu (motor tarafindan yonetilir)."""

    entry_index: int
    entry_ts: str
    entry_price: float
    qty: float
    stop_price: float
    take_profit_price: float
    highest_price: float
    lowest_price: float
    bars_held: int = 0
    fees_paid: float = 0.0
    reason: str = ""

    def unrealized_pct(self, price: float) -> float:
        if not self.entry_price:
            return 0.0
        return (price / self.entry_price - 1.0) * 100.0


@dataclass
class TradeRecord:
    """Kapanmis islem kaydi."""

    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    qty: float
    pnl_quote: float
    pnl_pct: float
    fees_quote: float
    exit_reason: str
    bars_held: int
    mae_pct: float = 0.0
    mfe_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "entry_price": round(float(self.entry_price), 6),
            "exit_price": round(float(self.exit_price), 6),
            "qty": round(float(self.qty), 8),
            "pnl_quote": round(float(self.pnl_quote), 4),
            "pnl_pct": round(float(self.pnl_pct), 4),
            "fees_quote": round(float(self.fees_quote), 4),
            "exit_reason": self.exit_reason,
            "bars_held": int(self.bars_held),
            "mae_pct": round(float(self.mae_pct), 4),
            "mfe_pct": round(float(self.mfe_pct), 4),
        }


@dataclass
class SignalDecision:
    """Stratejinin urettigi karar (dict yerine tip guvenli alternatif)."""

    action: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "reason": self.reason}


def safe_float(value: Any, default: float = math.nan) -> float:
    """Series/dict hucresini guvenli float'a cevirir."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return number


def clamp_stop_distance(distance: float, price: float) -> float:
    """Stop mesafesini makul araliga sikistirir (%0.1 - %25)."""
    if price <= 0:
        return max(distance, 0.0)
    return max(price * 0.001, min(distance, price * 0.25))


class BaseStrategy(ABC):
    """Tum stratejilerin ortak davranisi."""

    name: str = "base_strategy"
    family: str = "hybrid"
    version: str = "v000"
    timeframe: str = "1h"
    thesis: str = ""
    risk_defaults: Dict[str, Any] = {
        "stop_loss_atr": 2.0,
        "take_profit_atr": 3.0,
        "trailing_stop_atr": 0.0,
        "risk_per_trade_pct": 1.5,
        "cooldown_bars": 0,
        "max_holding_bars": 0,
        "allow_short": False,
    }

    def __init__(self, params: Optional[Mapping[str, Any]] = None):
        defaults = dict(self.default_params())
        defaults.update({k: v for k, v in (params or {}).items() if k in defaults})
        self.params: Dict[str, Any] = defaults
        self.risk = RiskConfig(
            stop_loss_atr=float(self.risk_defaults.get("stop_loss_atr", 2.0)),
            take_profit_atr=float(self.risk_defaults.get("take_profit_atr", 3.0)),
            trailing_stop_atr=float(self.risk_defaults.get("trailing_stop_atr", 0.0)),
            risk_per_trade_pct=float(self.risk_defaults.get("risk_per_trade_pct", 1.5)),
            cooldown_bars=int(self.risk_defaults.get("cooldown_bars", 0)),
            max_holding_bars=int(self.risk_defaults.get("max_holding_bars", 0)),
            allow_short=bool(self.risk_defaults.get("allow_short", False)),
        )

    # -- soyut arayuz -------------------------------------------------------
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Stratejinin varsayilan parametreleri (alt sinif doldurur)."""
        return {}

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Indikator kolonlarini ekler. Varsayilan: dokunmadan kopyalar."""
        return frame.copy()

    @abstractmethod
    def generate_signal(
        self,
        row: Mapping[str, Any],
        prev: Mapping[str, Any],
        position: Optional[Position],
    ) -> Any:
        """Karar uretir: ``{"action": "buy"|"sell"|"hold", "reason": str}``."""
        raise NotImplementedError

    # -- risk / boyutlandirma ----------------------------------------------
    def stop_distance(self, row: Mapping[str, Any], price: float) -> float:
        atr_value = safe_float(row.get("atr"), 0.0)
        distance = atr_value * float(self.risk.stop_loss_atr)
        if not math.isfinite(distance) or distance <= 0:
            distance = price * 0.01  # ATR yoksa %1 taban
        return float(clamp_stop_distance(distance, price))

    def take_profit_distance(self, row: Mapping[str, Any], price: float) -> float:
        atr_value = safe_float(row.get("atr"), 0.0)
        distance = atr_value * float(self.risk.take_profit_atr)
        if not math.isfinite(distance) or distance < 0:
            distance = 0.0
        return float(distance)

    def position_size(self, equity: float, price: float, stop_distance: float) -> float:
        """Risk yuzdesine gore miktar; kaldirac yok (spot)."""
        if price <= 0 or equity <= 0:
            return 0.0
        risk_quote = equity * float(self.risk.risk_per_trade_pct) / 100.0
        qty = risk_quote / stop_distance if stop_distance > 0 else risk_quote / price
        max_qty = equity / price
        return float(max(0.0, min(qty, max_qty)))

    # -- bilgi / kancalar ---------------------------------------------------
    @classmethod
    def describe(cls) -> str:
        return f"{cls.name} [{cls.family}] {cls.version} - {cls.thesis or 'aciklama yok'}"

    @classmethod
    def spec_params(cls) -> Dict[str, Any]:
        return dict(cls.default_params())

    def on_trade_closed(self, trade: TradeRecord) -> None:
        """Alt siniflar performans geri bildirimi almak isterse override eder."""
        return None

    def on_position_opened(self, position: Position) -> None:
        """Pozisyon acildiginda cagrilir."""
        return None

