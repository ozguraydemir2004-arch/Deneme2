"""Indikator ve veri madenciligi testleri.

En kritik test: Donchian kanalinin ileriye bakmadigini (look-ahead yok)
dogrulamak. Bu hata breakout stratejilerinin hic islem almamasina yol
aciyordu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agency.data.indicators import (
    adx,
    atr,
    bollinger,
    donchian,
    ema,
    macd,
    roc,
    rsi,
    sma,
    stochastic,
    volume_sma,
)
from agency.data.mining import (
    classify_regime,
    compute_features,
    feature_dict,
    mine_snapshots,
)


@pytest.fixture()
def series() -> pd.Series:
    rng = np.random.default_rng(7)
    values = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=400)))
    return pd.Series(values)


@pytest.fixture()
def ohlcv(series: pd.Series) -> pd.DataFrame:
    open_ = series.shift(1).fillna(series.iloc[0])
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(series), freq="1h", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, series) * 1.001,
            "low": np.minimum(open_, series) * 0.999,
            "close": series,
            "volume": np.linspace(10, 40, len(series)),
        }
    )


def test_sma_ema_lengths_and_warmup(series: pd.Series) -> None:
    fast = sma(series, 10)
    slow = ema(series, 10)
    assert len(fast) == len(series) == len(slow)
    assert fast.iloc[:9].isna().all()
    assert slow.iloc[:9].isna().all()
    assert fast.iloc[-1] == pytest.approx(series.iloc[-10:].mean())


def test_rsi_bounded(series: pd.Series) -> None:
    values = rsi(series, 14).dropna()
    assert not values.empty
    assert values.min() >= 0.0 and values.max() <= 100.0


def test_atr_positive(ohlcv: pd.DataFrame) -> None:
    values = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14).dropna()
    assert (values > 0).all()


def test_adx_outputs(ohlcv: pd.DataFrame) -> None:
    result = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
    assert set(result) == {"adx", "plus_di", "minus_di"}
    adx_values = result["adx"].dropna()
    assert not adx_values.empty
    assert (adx_values >= 0).all() and (adx_values <= 100).all()


def test_macd_hist_relation(series: pd.Series) -> None:
    values = macd(series, 12, 26, 9)
    last = values["macd"].iloc[-1] - values["signal"].iloc[-1]
    assert values["hist"].iloc[-1] == pytest.approx(last)


def test_bollinger_ordering(series: pd.Series) -> None:
    bands = bollinger(series, 20, 2.0)
    upper = bands["upper"].dropna()
    middle = bands["middle"].dropna()
    lower = bands["lower"].dropna()
    assert not upper.empty
    assert (upper >= middle).all()
    assert (middle >= lower).all()
    assert bands["pctb"].dropna().between(-0.5, 1.5).all()


def test_roc_and_stochastic(ohlcv: pd.DataFrame) -> None:
    assert roc(ohlcv["close"], 10).dropna().notna().all()
    stoch = stochastic(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14, 3)
    assert stoch["k"].dropna().between(0, 100).all()


def test_volume_sma(ohlcv: pd.DataFrame) -> None:
    values = volume_sma(ohlcv["volume"], 20)
    assert values.iloc[-1] == pytest.approx(ohlcv["volume"].iloc[-20:].mean())


def test_donchian_has_no_lookahead(ohlcv: pd.DataFrame) -> None:
    """Kanal, mevcut bari DISLAMALI (shift(1)) - look-ahead regresyon testi."""
    period = 20
    index = 50
    channel = donchian(ohlcv["high"], ohlcv["low"], period)
    expected_upper = ohlcv["high"].iloc[index - period : index].max()
    expected_lower = ohlcv["low"].iloc[index - period : index].min()
    assert channel["upper"].iloc[index] == pytest.approx(expected_upper)
    assert channel["lower"].iloc[index] == pytest.approx(expected_lower)

    # Mevcut barin high degerini yapay olarak sisir: kanal DEGISMEMELI
    mutated = ohlcv.copy()
    mutated.loc[index, "high"] = mutated["high"].iloc[index] * 10
    mutated.loc[index, "low"] = mutated["low"].iloc[index] / 10
    mutated_channel = donchian(mutated["high"], mutated["low"], period)
    assert mutated_channel["upper"].iloc[index] == pytest.approx(expected_upper)
    assert mutated_channel["lower"].iloc[index] == pytest.approx(expected_lower)


def test_compute_features_columns(ohlcv: pd.DataFrame) -> None:
    features = compute_features(ohlcv)
    for column in ("rsi", "adx", "atr_pct", "bb_width", "rel_volume", "trend_score"):
        assert column in features.columns
    last = feature_dict(features.iloc[-1])
    assert "rsi" in last and 0 <= last["rsi"] <= 100


def test_classify_regime_labels(ohlcv: pd.DataFrame) -> None:
    features = compute_features(ohlcv)
    regime, strength, volatility = classify_regime(feature_dict(features.iloc[-1]))
    assert regime in {
        "trend_up",
        "trend_down",
        "range",
        "volatile",
        "squeeze",
        "oversold_range",
        "overbought_range",
    }
    assert strength in {"strong", "moderate", "weak"}
    assert volatility in {"high", "normal", "low"}


def test_mine_snapshots_labels(ohlcv: pd.DataFrame) -> None:
    features = compute_features(ohlcv)
    snapshots = mine_snapshots(
        ohlcv,
        features,
        horizon=12,
        winner_threshold_pct=0.8,
        max_snapshots=40,
    )
    assert snapshots
    for snapshot in snapshots:
        assert snapshot["label"] in {"winner", "loser"}
        assert "features" in snapshot and "bar_index" in snapshot
    # Gap kurali: ayni hareketten ardisik snapshot alinmamali
    indexes = sorted(s["bar_index"] for s in snapshots)
    assert all(b - a >= 6 for a, b in zip(indexes, indexes[1:]))


def test_mine_snapshots_empty_when_threshold_impossible(ohlcv: pd.DataFrame) -> None:
    features = compute_features(ohlcv)
    snapshots = mine_snapshots(
        ohlcv, features, horizon=12, winner_threshold_pct=999.0, loser_threshold_pct=999.0
    )
    assert snapshots == []
