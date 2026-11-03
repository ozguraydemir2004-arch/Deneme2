"""Veri madenciligi: ozellik cikarimi, rejim siniflandirma, basari snapshot'lari.

Bu modul Research Agent'in "analitik kasidir". LLM'e bagli degildir; tamamen
deterministik calisir ve her dongude ayni veri icin ayni sonucu uretir.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from agency.data.indicators import adx, atr, bollinger, ema, macd, rsi, sma, stochastic
from agency.schemas import MarketContext

#: Snapshot'lara yazilan sayisal ozellikler (vektor hafizada da kullanilir)
FEATURE_KEYS = (
    "rsi",
    "rsi_delta",
    "adx",
    "plus_di",
    "minus_di",
    "atr_pct",
    "volatility_ratio",
    "rel_volume",
    "macd_hist_pct",
    "roc",
    "bb_width",
    "bb_pctb",
    "stoch_k",
    "ema_spread_pct",
    "dist_ema_slow_pct",
    "close_pos_20",
    "close_pos_50",
    "trend_score",
    "slope_20_pct",
    "drawdown_from_high_pct",
    "runup_from_low_pct",
)


def compute_features(frame: pd.DataFrame) -> pd.DataFrame:
    """OHLCV'den indikator/ozellik matrisi uretir."""
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    out = pd.DataFrame(index=frame.index)
    rsi_series = rsi(close, 14)
    out["rsi"] = rsi_series
    out["rsi_delta"] = rsi_series.diff()

    adx_values = adx(high, low, close, 14)
    out["adx"] = adx_values["adx"]
    out["plus_di"] = adx_values["plus_di"]
    out["minus_di"] = adx_values["minus_di"]

    atr_series = atr(high, low, close, 14)
    atr_pct = atr_series / close * 100.0
    out["atr_pct"] = atr_pct
    out["volatility_ratio"] = atr_pct / atr_pct.rolling(96, min_periods=24).median()

    out["rel_volume"] = volume / volume.rolling(20, min_periods=5).mean()

    macd_values = macd(close, 12, 26, 9)
    out["macd_hist_pct"] = macd_values["hist"] / close * 100.0
    out["roc"] = close.pct_change(10) * 100.0

    bands = bollinger(close, 20, 2.0)
    out["bb_width"] = bands["width"]
    out["bb_pctb"] = bands["pctb"]
    out["stoch_k"] = stochastic(high, low, close, 14, 3)["k"]

    ema_fast = ema(close, 20)
    ema_slow = ema(close, 50)
    out["ema_spread_pct"] = (ema_fast - ema_slow) / close * 100.0
    out["dist_ema_slow_pct"] = (close - ema_slow) / close * 100.0

    low_20 = low.rolling(20, min_periods=10).min()
    high_20 = high.rolling(20, min_periods=10).max()
    out["close_pos_20"] = (close - low_20) / (high_20 - low_20).replace(0.0, np.nan)
    low_50 = low.rolling(50, min_periods=20).min()
    high_50 = high.rolling(50, min_periods=20).max()
    out["close_pos_50"] = (close - low_50) / (high_50 - low_50).replace(0.0, np.nan)

    sma_20 = sma(close, 20)
    out["trend_score"] = (
        (close > ema_slow).astype(float)
        + (ema_fast > ema_slow).astype(float)
        + (close > sma_20).astype(float)
        + (out["plus_di"] > out["minus_di"]).astype(float)
    ) / 4.0 - 0.5

    out["slope_20_pct"] = sma_20.pct_change(5) * 100.0
    out["drawdown_from_high_pct"] = (high_50 - close) / close * 100.0
    out["runup_from_low_pct"] = (close - low_50) / close * 100.0

    return out.replace([np.inf, -np.inf], np.nan)


def classify_regime(features: Mapping[str, float]) -> Tuple[str, str, str]:
    """(regime, trend_strength, volatility_regime) dondurur."""
    adx_value = float(features.get("adx") or 0.0)
    rsi_value = float(features.get("rsi") or 50.0)
    ema_spread = float(features.get("ema_spread_pct") or 0.0)
    trend_score = float(features.get("trend_score") or 0.0)
    vol_ratio = float(features.get("volatility_ratio") or 1.0)
    bb_width = float(features.get("bb_width") or 0.0)

    if adx_value >= 25.0:
        trend_strength = "strong"
    elif adx_value >= 18.0:
        trend_strength = "moderate"
    else:
        trend_strength = "weak"

    if vol_ratio >= 1.35:
        volatility_regime = "high"
    elif vol_ratio <= 0.7:
        volatility_regime = "low"
    else:
        volatility_regime = "normal"

    if adx_value >= 20.0 and ema_spread > 0 and trend_score > 0:
        regime = "trend_up"
    elif adx_value >= 20.0 and ema_spread < 0 and trend_score < 0:
        regime = "trend_down"
    elif volatility_regime == "high":
        regime = "volatile"
    elif bb_width and bb_width < 2.0:
        regime = "squeeze"
    elif rsi_value <= 35.0:
        regime = "oversold_range"
    elif rsi_value >= 65.0:
        regime = "overbought_range"
    else:
        regime = "range"

    return regime, trend_strength, volatility_regime


def feature_dict(row: Any, feature_keys: Tuple[str, ...] = FEATURE_KEYS) -> Dict[str, float]:
    """Bir satiri (Series/dict) yuvarlanmis ozellik sozlugune cevirir."""
    payload: Dict[str, float] = {}
    for key in feature_keys:
        try:
            value = row[key]
        except (KeyError, IndexError, TypeError):
            continue
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(number) or np.isinf(number):
            continue
        payload[key] = round(number, 5)
    return payload


def build_market_context(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    source: str,
    order_book: Optional[Mapping[str, Any]] = None,
) -> MarketContext:
    """Son bar uzerinden piyasa rejimi ozeti uretir."""
    if frame.empty or features.empty:
        raise ValueError("Bos veri ile piyasa baglami olusturulamaz")

    last_features = feature_dict(features.iloc[-1])
    regime, trend_strength, vol_regime = classify_regime(last_features)

    close_series = frame["close"].astype(float)
    ema_fast_value = float(ema(close_series, 20).iloc[-1])
    ema_slow_value = float(ema(close_series, 50).iloc[-1])

    lookback = min(len(frame) - 1, 100)
    reference_price = (
        float(frame["close"].iloc[-1 - lookback]) if lookback > 0 else float(frame["close"].iloc[0])
    )
    last_price = float(frame["close"].iloc[-1])
    price_change = (last_price / reference_price - 1.0) * 100.0 if reference_price else 0.0

    start = frame["timestamp"].iloc[0]
    end = frame["timestamp"].iloc[-1]
    order_book = order_book or {}

    return MarketContext(
        symbol=symbol,
        timeframe=timeframe,
        data_source=source,
        bars=int(len(frame)),
        last_price=last_price,
        price_change_pct=price_change,
        atr_pct=last_features.get("atr_pct", 0.0),
        rsi=last_features.get("rsi", 50.0),
        adx=last_features.get("adx", 0.0),
        ema_fast=ema_fast_value,
        ema_slow=ema_slow_value,
        volume_ratio=last_features.get("rel_volume", 1.0),
        regime=regime,
        trend_strength=trend_strength,
        volatility_regime=vol_regime,
        start=str(start),
        end=str(end),
    )


def forward_returns(frame: pd.DataFrame, horizon: int) -> pd.Series:
    close = frame["close"].astype(float)
    return (close.shift(-horizon) / close - 1.0) * 100.0


def mine_snapshots(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    horizon: int = 12,
    winner_threshold_pct: float = 1.5,
    loser_threshold_pct: Optional[float] = None,
    max_snapshots: int = 400,
    warmup_bars: int = 60,
    vectorizer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Basari (ve ornek kayip) anlarini snapshot olarak cikarir.

    Yaklasim:
        1. Her bar icin ``horizon`` barlik ileri getiri hesaplanir.
        2. Getiri >= esik -> ``winner``, <= -esik/2 -> ``loser``.
        3. Ayni hareketten birden fazla snapshot alinmamasi icin en az
           ``horizon // 2`` bar arayla secim yapilir.
        4. En yuksek getiriden baslanarak ``max_snapshots`` kadar kayit alinir.
    """
    loser_threshold_pct = (
        float(loser_threshold_pct) if loser_threshold_pct is not None else float(winner_threshold_pct) / 2.0
    )
    returns = forward_returns(frame, horizon)
    min_gap = max(1, int(horizon) // 2)
    snapshots: List[Dict[str, Any]] = []
    taken: List[int] = []

    for label, threshold, reverse in (
        ("winner", float(winner_threshold_pct), True),
        ("loser", -abs(loser_threshold_pct), False),
    ):
        if label == "winner":
            mask = returns >= threshold
        else:
            mask = returns <= threshold
        indices = [int(i) for i in np.flatnonzero(mask.to_numpy()) if i >= warmup_bars]
        if not indices:
            continue
        ordered = sorted(indices, key=lambda i: returns.iloc[i], reverse=reverse)

        label_taken = 0
        for idx in ordered:
            if any(abs(idx - other) < min_gap for other in taken):
                continue
            feature_row = features.iloc[idx]
            payload = feature_dict(feature_row)
            if len(payload) < 6:  # yeterli indikator verisi yok
                continue
            entry = {
                "ts": str(frame["timestamp"].iloc[idx]),
                "bar_index": idx,
                "close": float(frame["close"].iloc[idx]),
                "forward_return_pct": round(float(returns.iloc[idx]), 4),
                "label": label,
                "features": payload,
            }
            if vectorizer is not None:
                entry["vector"] = vectorizer.embed_dict(payload)
            snapshots.append(entry)
            taken.append(idx)
            label_taken += 1
            if label_taken >= max_snapshots:
                break

    snapshots.sort(key=lambda item: item["bar_index"])
    return snapshots


def summarize_snapshot(snapshot: Mapping[str, Any]) -> str:
    """Snapshot'i log/rapor icin kisa metne cevirir."""
    features = snapshot.get("features") or {}
    return (
        f"{snapshot.get('ts')} {snapshot.get('label')} "
        f"fwd={snapshot.get('forward_return_pct')}% "
        f"rsi={features.get('rsi')} adx={features.get('adx')} "
        f"atr%={features.get('atr_pct')} bbw={features.get('bb_width')}"
    )

