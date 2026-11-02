"""Indikator kutuphanesi + indikator katalogu.

Bu modul iki is yapar:

1. ``pandas``/``numpy`` ile saf, vektorize indikator hesaplari saglar.
2. :data:`INDICATOR_KINDS` katalogu ile hangi indikatorun hangi parametreleri
   aldigini ve hangi kolonlari urettigini tanimlar. Strateji derleyicisi
   (``agency.strategies.compiler``) yalnizca bu katalogdaki kombinasyonlari
   uretir; boylece LLM'den gelen serbest metin asla gecersiz kod uretemez.

Tum fonksiyonlar ``pd.Series`` girdi alir, ayni uzunlukta ``pd.Series``
dondurur ve eksik degerleri ``float`` olarak birakir (backfill yapmaz).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

__all__ = [
    "sma",
    "ema",
    "rsi",
    "atr",
    "adx",
    "macd",
    "bollinger",
    "roc",
    "stochastic",
    "rolling_vwap",
    "donchian",
    "volume_sma",
    "INDICATOR_KINDS",
    "RAW_OPERANDS",
    "catalog_for_prompt",
]


# ---------------------------------------------------------------------------
# Temel indikatorler
# ---------------------------------------------------------------------------
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=int(period), min_periods=int(period)).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=int(period), adjust=False, min_periods=int(period)).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    period = int(period)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    period = int(period)
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> Dict[str, pd.Series]:
    period = int(period)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )
    atr_values = atr(high, low, close, period).replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_values
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_values
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_values = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return {"adx": adx_values, "plus_di": plus_di, "minus_di": minus_di}


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    signal_line = line.ewm(span=int(signal), adjust=False, min_periods=int(signal)).mean()
    return {"macd": line, "signal": signal_line, "hist": line - signal_line}


def bollinger(close: pd.Series, period: int = 20, std: float = 2.0) -> Dict[str, pd.Series]:
    period = int(period)
    middle = sma(close, period)
    deviation = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + float(std) * deviation
    lower = middle - float(std) * deviation
    width = (upper - lower) / middle.replace(0.0, np.nan) * 100.0
    pctb = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "width": width,
        "pctb": pctb,
    }


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    period = int(period)
    return close.pct_change(periods=period) * 100.0


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    smooth: int = 3,
) -> Dict[str, pd.Series]:
    period = int(period)
    lowest = low.rolling(window=period, min_periods=period).min()
    highest = high.rolling(window=period, min_periods=period).max()
    k = (close - lowest) / (highest - lowest).replace(0.0, np.nan) * 100.0
    d = k.rolling(window=int(smooth), min_periods=int(smooth)).mean()
    return {"k": k, "d": d}


def rolling_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3.0
    window = int(period)
    pv = (typical * volume).rolling(window=window, min_periods=window).sum()
    vol = volume.rolling(window=window, min_periods=window).sum().replace(0.0, np.nan)
    return pv / vol


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
    """Onceki N barin en yuksek/en dusuk seviyesi (kanal).

    ONEMLI: Kanal, **icinde bulunulan bar haric** hesaplanir (``shift(1)``).
    Aksi halde ``close > upper`` kosulu matematiksel olarak imkansiz olur
    (bar'in kendi high degeri kanalin icine girer) - klasik ileriye bakma
    (look-ahead) hatasi.
    """
    window = int(period)
    upper = high.rolling(window=window, min_periods=window).max().shift(1)
    lower = low.rolling(window=window, min_periods=window).min().shift(1)
    mid = (upper + lower) / 2.0
    return {"upper": upper, "lower": lower, "mid": mid}


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    return sma(volume, period)


# ---------------------------------------------------------------------------
# Indikator katalogu
# ---------------------------------------------------------------------------
# Her kayit:
#   params  -> {isim: {"min":.., "max":.., "default":.., "type": "int"|"float"}}
#   outputs -> uretilebilen cikti adlari; strateji spec'i "output" ile secer
#   needs   -> ham veri kolonlari
#   family  -> trend | momentum | volatility | volume | structure
#   desc    -> LLM prompt'larina giden kisa aciklama
INDICATOR_KINDS: Dict[str, Dict[str, Any]] = {
    "ema": {
        "params": {"period": {"type": "int", "min": 2, "max": 400, "default": 20}},
        "outputs": ["value"],
        "needs": ["close"],
        "family": "trend",
        "desc": "Ustel hareketli ortalama (fiyat).",
    },
    "sma": {
        "params": {"period": {"type": "int", "min": 2, "max": 400, "default": 50}},
        "outputs": ["value"],
        "needs": ["close"],
        "family": "trend",
        "desc": "Basit hareketli ortalama (fiyat).",
    },
    "rsi": {
        "params": {"period": {"type": "int", "min": 2, "max": 100, "default": 14}},
        "outputs": ["value"],
        "needs": ["close"],
        "family": "momentum",
        "desc": "Goreli guc endeksi (0-100). 30 alti asiri satim, 70 ustu asiri alim.",
    },
    "atr": {
        "params": {"period": {"type": "int", "min": 2, "max": 100, "default": 14}},
        "outputs": ["value"],
        "needs": ["high", "low", "close"],
        "family": "volatility",
        "desc": "Ortalama gercek aralik (mutlak fiyat biriminde volatilite).",
    },
    "atr_pct": {
        "params": {"period": {"type": "int", "min": 2, "max": 100, "default": 14}},
        "outputs": ["value"],
        "needs": ["high", "low", "close"],
        "family": "volatility",
        "desc": "ATR'nin kapanisa orani (yuzde) - rejim/volatilite filtresi icin.",
    },
    "adx": {
        "params": {"period": {"type": "int", "min": 2, "max": 100, "default": 14}},
        "outputs": ["value", "plus_di", "minus_di"],
        "needs": ["high", "low", "close"],
        "family": "trend",
        "desc": "Trend gucu (0-100). 20 ustu trend var, 25+ guclu trend.",
    },
    "macd": {
        "params": {
            "fast": {"type": "int", "min": 2, "max": 100, "default": 12},
            "slow": {"type": "int", "min": 3, "max": 200, "default": 26},
            "signal": {"type": "int", "min": 2, "max": 100, "default": 9},
        },
        "outputs": ["value", "signal", "hist"],
        "needs": ["close"],
        "family": "momentum",
        "desc": "MACD cizgisi / sinyal cizgisi / histogram.",
    },
    "bbands": {
        "params": {
            "period": {"type": "int", "min": 5, "max": 200, "default": 20},
            "std": {"type": "float", "min": 0.5, "max": 5.0, "default": 2.0},
        },
        "outputs": ["upper", "middle", "lower", "width", "pctb"],
        "needs": ["close"],
        "family": "volatility",
        "desc": "Bollinger bantlari: ust/orta/alt, bant genisligi (%), %B konumu.",
    },
    "roc": {
        "params": {"period": {"type": "int", "min": 1, "max": 200, "default": 10}},
        "outputs": ["value"],
        "needs": ["close"],
        "family": "momentum",
        "desc": "Yuzde degisim orani (momentum).",
    },
    "stoch": {
        "params": {
            "period": {"type": "int", "min": 3, "max": 100, "default": 14},
            "smooth": {"type": "int", "min": 1, "max": 20, "default": 3},
        },
        "outputs": ["k", "d"],
        "needs": ["high", "low", "close"],
        "family": "momentum",
        "desc": "Stokastik osilator: %K ve %D (0-100).",
    },
    "vwap": {
        "params": {"period": {"type": "int", "min": 5, "max": 400, "default": 20}},
        "outputs": ["value"],
        "needs": ["high", "low", "close", "volume"],
        "family": "volume",
        "desc": "Hareketli hacim agirlikli ortalama fiyat.",
    },
    "vol_sma": {
        "params": {"period": {"type": "int", "min": 2, "max": 200, "default": 20}},
        "outputs": ["value"],
        "needs": ["volume"],
        "family": "volume",
        "desc": "Hacim hareketli ortalamasi.",
    },
    "rel_volume": {
        "params": {"period": {"type": "int", "min": 2, "max": 200, "default": 20}},
        "outputs": ["value"],
        "needs": ["volume"],
        "family": "volume",
        "desc": "Hacim / hacim ortalamasi (1.0 = ortalama).",
    },
    "donchian": {
        "params": {"period": {"type": "int", "min": 5, "max": 400, "default": 20}},
        "outputs": ["upper", "lower", "mid"],
        "needs": ["high", "low"],
        "family": "structure",
        "desc": "Donchian kanali (N barlik en yuksek/en dusuk).",
    },
}

RAW_OPERANDS: List[str] = ["open", "high", "low", "close", "volume"]

#: Kural ifadelerinde kullanilabilecek sabitler (kolaylik)
CONSTANT_OPERANDS = {"zero": 0.0, "one": 1.0}


def catalog_for_prompt() -> str:
    """LLM prompt'larina gomulecek kompakt katalog metni."""
    lines: List[str] = []
    for kind, spec in INDICATOR_KINDS.items():
        params = ", ".join(
            f"{name}:{meta['type']}={meta['default']}" for name, meta in spec["params"].items()
        )
        outputs = "/".join(spec["outputs"])
        lines.append(f"- {kind}({params}) -> outputs[{outputs}] :: {spec['desc']}")
    lines.append(f"- raw operands: {', '.join(RAW_OPERANDS)}")
    lines.append(f"- constants: {', '.join(CONSTANT_OPERANDS)}")
    return "\n".join(lines)

