"""Piyasa verisi saglayicisi.

Iki kaynak desteklenir:

* ``ccxt``      : gercek borsa verisi (varsayilan Binance spot, BTC/USDT).
* ``synthetic`` : cevrimdisi/demo icin deterministik, rejim degistiren
                  sentetik OHLCV uretimi (ag yoksa otomatik devreye girer).

``data.provider: auto`` modunda once CCXT denenir; basarisiz olursa sentetik
veri kullanilir ve bu durum ``warnings`` alaninda raporlanir. Boylece sistem
internet olmadan da tam dongu calistirabilir (test/demo).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agency.config import Settings
from agency.exceptions import DataError
from agency.utils import iso_now, utc_now

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

TIMEFRAME_MINUTES: Dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200,
}


@dataclass
class MarketDataResult:
    """OHLCV verisi + kaynak bilgisi."""

    frame: pd.DataFrame
    source: str
    symbol: str
    timeframe: str
    fetched_at: str = field(default_factory=iso_now)
    cached: bool = False
    warnings: List[str] = field(default_factory=list)
    order_book: Dict[str, Any] = field(default_factory=dict)

    @property
    def bars(self) -> int:
        return int(len(self.frame))

    def validate(self) -> None:
        if self.frame is None or self.frame.empty:
            raise DataError("OHLCV verisi bos", details={"source": self.source})
        missing = [c for c in OHLCV_COLUMNS if c not in self.frame.columns]
        if missing:
            raise DataError(f"OHLCV kolonlari eksik: {missing}")
        if not self.frame["close"].gt(0).all():
            raise DataError("Gecersiz kapanis fiyatlari (<=0)", details={"source": self.source})


def timeframe_minutes(timeframe: str) -> int:
    if timeframe not in TIMEFRAME_MINUTES:
        raise DataError(f"Desteklenmeyen timeframe: {timeframe}")
    return TIMEFRAME_MINUTES[timeframe]


class MarketDataProvider:
    """CCXT ve sentetik veri kaynaklarini tek arayuzde birlestirir."""

    def __init__(self, settings: Settings):
        self.settings = settings
        cfg = settings.data
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._exchange = None
        self.last_source: str = "unknown"
        self.last_warnings: List[str] = []

    # ------------------------------------------------------------ public API
    def fetch_ohlcv(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
        *,
        use_cache: bool = True,
    ) -> MarketDataResult:
        symbol = symbol or self.settings.exchange.symbol
        timeframe = timeframe or self.settings.exchange.timeframe
        limit = int(limit or self.settings.exchange.limit)
        mode = self.settings.data.provider
        warnings: List[str] = []

        cache_path = self._cache_path(symbol, timeframe, limit)
        if use_cache and cache_path.is_file():
            age_minutes = (time.time() - cache_path.stat().st_mtime) / 60.0
            if age_minutes <= self.settings.data.cache_ttl_minutes:
                frame = pd.read_csv(cache_path)
                result = MarketDataResult(
                    frame=frame[OHLCV_COLUMNS],
                    source="cache",
                    symbol=symbol,
                    timeframe=timeframe,
                    cached=True,
                )
                self.last_source = result.source
                return result

        if mode in {"auto", "ccxt"}:
            try:
                frame, source = self._fetch_ccxt(symbol, timeframe, limit)
                result = MarketDataResult(frame=frame, source=source, symbol=symbol, timeframe=timeframe)
                result.validate()
                self._write_cache(cache_path, frame)
                self.last_source = source
                self.last_warnings = warnings
                return result
            except Exception as exc:  # ag/kurulum hatalari
                message = f"CCXT veri cekilemedi ({type(exc).__name__}: {exc})."
                if mode == "ccxt":
                    raise DataError(message, details={"symbol": symbol, "timeframe": timeframe}) from exc
                warnings.append(message + " Sentetik veriye gecildi.")

        frame = self._fetch_synthetic(symbol, timeframe, limit)
        result = MarketDataResult(
            frame=frame,
            source="synthetic",
            symbol=symbol,
            timeframe=timeframe,
            warnings=warnings,
        )
        result.validate()
        self._write_cache(cache_path, frame)
        self.last_source = "synthetic"
        self.last_warnings = warnings
        return result


    def fetch_order_book_summary(self, symbol: Optional[str] = None, depth: Optional[int] = None) -> Dict[str, Any]:
        """Emir defteri dengesizligi (bid/ask imbalance) ozeti."""
        symbol = symbol or self.settings.exchange.symbol
        depth = int(depth or self.settings.exchange.order_book_depth)
        if self.settings.data.provider == "synthetic":
            return {"source": "synthetic", "imbalance": 0.0, "spread_bps": 0.0, "depth": depth}
        try:
            exchange = self._get_exchange()
            book = exchange.fetch_order_book(symbol, limit=depth)
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            bid_vol = sum(float(amount) for _, amount in bids)
            ask_vol = sum(float(amount) for _, amount in asks)
            total = bid_vol + ask_vol
            best_bid = float(bids[0][0]) if bids else 0.0
            best_ask = float(asks[0][0]) if asks else 0.0
            mid = (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0
            spread_bps = ((best_ask - best_bid) / mid * 10000.0) if mid else 0.0
            return {
                "source": "ccxt",
                "imbalance": round((bid_vol - ask_vol) / total, 4) if total else 0.0,
                "spread_bps": round(spread_bps, 3),
                "depth": depth,
                "best_bid": best_bid,
                "best_ask": best_ask,
            }
        except Exception as exc:  # pragma: no cover - ag bagimli
            return {"source": "unavailable", "error": str(exc), "imbalance": 0.0, "spread_bps": 0.0}

    def close(self) -> None:
        if self._exchange is not None:
            try:
                self._exchange.close()
            except Exception:  # pragma: no cover - en iyi caba
                pass
            self._exchange = None

    # --------------------------------------------------------------- ccxt
    def _get_exchange(self):
        if self._exchange is not None:
            return self._exchange
        try:
            import ccxt  # yerel import: kutuphane yoksa sistem sentetik moda duser
        except ImportError as exc:  # pragma: no cover
            raise DataError("ccxt kurulu degil") from exc

        exchange_id = self.settings.exchange.id
        if not hasattr(ccxt, exchange_id):
            raise DataError(f"ccxt '{exchange_id}' borsasini desteklemiyor")
        exchange_class = getattr(ccxt, exchange_id)
        self._exchange = exchange_class(
            {
                "enableRateLimit": self.settings.exchange.enable_rate_limit,
                "timeout": self.settings.exchange.request_timeout_ms,
                "options": {"defaultType": "spot"},
            }
        )
        return self._exchange

    def _fetch_ccxt(self, symbol: str, timeframe: str, limit: int) -> tuple[pd.DataFrame, str]:
        exchange = self._get_exchange()
        attempts = max(1, int(self.settings.exchange.max_retries) + 1)
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                if not getattr(exchange, "markets", None):
                    exchange.load_markets()
                raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if not raw:
                    raise DataError("Borsa bos OHLCV dondurdu")
                frame = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
                frame = frame.dropna().sort_values("timestamp").reset_index(drop=True)
                return frame, f"ccxt:{self.settings.exchange.id}"
            except Exception as exc:  # pragma: no cover - ag bagimli
                last_error = exc
                time.sleep(min(2.0 * (attempt + 1), 5.0))
        raise DataError(f"CCXT fetch basarisiz: {last_error}") from last_error


    # ----------------------------------------------------------- synthetic
    def _fetch_synthetic(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Deterministik, rejim degistiren sentetik OHLCV.

        Gercek piyasa istatistiklerine yaklasmak icin:
            * rejim bloklari (trend / yatay / yuksek volatilite)
            * yillik volatilite ve drift parametreleri
            * hacim <-> |getiri| korelasyonu
        """
        cfg = self.settings.data
        bars = int(max(limit, cfg.synthetic_bars))
        rng = np.random.default_rng(int(cfg.synthetic_seed))
        minutes = timeframe_minutes(timeframe)
        bars_per_year = max(1.0, 365.0 * 24.0 * 60.0 / minutes)
        bar_vol = float(cfg.synthetic_annual_vol) / np.sqrt(bars_per_year)
        bar_drift = float(cfg.synthetic_drift) / bars_per_year

        vol_multiplier = np.ones(bars)
        drift_multiplier = np.ones(bars)
        index = 0
        while index < bars:
            block = int(rng.integers(80, 320))
            end = min(bars, index + block)
            regime = int(rng.integers(0, 3))
            if regime == 0:      # trend
                vol_multiplier[index:end] = rng.uniform(0.7, 1.1)
                drift_multiplier[index:end] = rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 1.6)
            elif regime == 1:    # yatay / sikisma
                vol_multiplier[index:end] = rng.uniform(0.35, 0.7)
                drift_multiplier[index:end] = rng.uniform(-0.2, 0.2)
            else:                # yuksek volatilite
                vol_multiplier[index:end] = rng.uniform(1.4, 2.6)
                drift_multiplier[index:end] = rng.uniform(-0.8, 0.8)
            index = end

        shocks = np.zeros(bars)
        shock_count = max(1, bars // 120)
        shock_idx = rng.choice(bars, size=shock_count, replace=False)
        shocks[shock_idx] = rng.normal(0.0, bar_vol * 3.0, size=shock_count)

        returns = (
            rng.normal(0.0, bar_vol, size=bars) * vol_multiplier
            + bar_drift * drift_multiplier
            + shocks
        )
        returns = np.clip(returns, -0.12, 0.12)
        close = float(cfg.synthetic_start_price) * np.exp(np.cumsum(returns))
        open_ = np.concatenate([[float(cfg.synthetic_start_price)], close[:-1]])

        wick = np.abs(rng.normal(0.0, bar_vol * 0.8, size=bars))
        high = np.maximum(open_, close) * (1.0 + wick)
        low = np.minimum(open_, close) * (1.0 - wick)
        volume = (
            900.0
            * (1.0 + 6.0 * np.abs(returns) / max(bar_vol, 1e-9))
            * rng.lognormal(0.0, 0.35, size=bars)
        )

        end_ts = pd.Timestamp(utc_now()).floor(f"{minutes}min")
        timestamps = pd.date_range(end=end_ts, periods=bars, freq=f"{minutes}min", tz="UTC")

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )[OHLCV_COLUMNS]

    # --------------------------------------------------------------- cache
    def _cache_path(self, symbol: str, timeframe: str, limit: int) -> Path:
        slug = symbol.replace("/", "").replace(":", "").lower()
        mode = self.settings.data.provider
        return self.cache_dir / f"{slug}_{timeframe}_{limit}_{mode}.csv"

    @staticmethod
    def _write_cache(path: Path, frame: pd.DataFrame) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            frame.to_csv(tmp, index=False)
            tmp.replace(path)
        except OSError:  # pragma: no cover - disk sorunu
            pass


