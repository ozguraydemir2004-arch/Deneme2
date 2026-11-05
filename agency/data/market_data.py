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

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agency.config import Settings
from agency.exceptions import DataError
from agency.utils import iso_now, read_json, utc_now, write_json

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

TIMEFRAME_MINUTES: Dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200,
}


COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "BNB": "binancecoin",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "LTC": "litecoin",
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


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
        self._exchanges: Dict[str, Any] = {}
        self.active_exchange_id: Optional[str] = None
        self.active_symbol: Optional[str] = None
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
        self.last_warnings = []

        cache_path = self._cache_path(symbol, timeframe, limit)
        if use_cache and cache_path.is_file():
            age_minutes = (time.time() - cache_path.stat().st_mtime) / 60.0
            if age_minutes <= self.settings.data.cache_ttl_minutes:
                frame = pd.read_csv(cache_path)
                meta = read_json(self._cache_meta_path(cache_path), default={}) or {}
                origin = str(meta.get("source") or "cache")
                result = MarketDataResult(
                    frame=frame[OHLCV_COLUMNS],
                    # Gercek kaynak korunur ("cache" yalnizca bir tasima bicimi)
                    source=origin,
                    symbol=symbol,
                    timeframe=timeframe,
                    cached=True,
                    warnings=[f"Veri onbellekten okundu ({age_minutes:.1f} dk)"],
                )
                self.last_source = result.source
                return result

        if mode in {"auto", "ccxt"}:
            try:
                frame, source = self._fetch_ccxt(symbol, timeframe, limit)
                warnings = list(self.last_warnings)
                # CCXT borsalari genelde 720 bar ile sinirlidir; daha uzun gecmis
                # varsa REST kaynagi ile genislet (ayni kaynaktan tek parca veri).
                if (
                    mode == "auto"
                    and self.settings.data.rest_fallback_enabled
                    and len(frame) < int(limit * 0.9)
                ):
                    try:
                        extended = self._fetch_coingecko(symbol, timeframe, limit)
                        if len(extended) > len(frame):
                            warnings.append(
                                f"CCXT {len(frame)} bar dondurdu; "
                                f"CoinGecko {len(extended)} bar ile gecmis genisletildi."
                            )
                            frame, source = extended, "coingecko"
                    except Exception as exc:
                        warnings.append(f"Gecmis genisletilemedi ({type(exc).__name__}): {exc}")
                result = MarketDataResult(
                    frame=frame,
                    source=source,
                    symbol=symbol,
                    timeframe=timeframe,
                    warnings=warnings,
                )
                result.validate()
                self._write_cache(cache_path, frame, source)
                self.last_source = source
                return result
            except Exception as exc:  # ag/kurulum hatalari
                message = f"CCXT veri cekilemedi ({type(exc).__name__}: {exc})."
                if mode == "ccxt":
                    raise DataError(message, details={"symbol": symbol, "timeframe": timeframe}) from exc
                warnings.append(message)

        if mode == "auto" and self.settings.data.rest_fallback_enabled:
            try:
                frame = self._fetch_coingecko(symbol, timeframe, limit)
                warnings.append("CCXT kullanilamadi; CoinGecko REST verisi kullanildi.")
                result = MarketDataResult(
                    frame=frame,
                    source="coingecko",
                    symbol=symbol,
                    timeframe=timeframe,
                    warnings=warnings,
                )
                result.validate()
                self._write_cache(cache_path, frame, "coingecko")
                self.last_source = "coingecko"
                self.last_warnings = warnings
                return result
            except Exception as exc:
                warnings.append(
                    f"CoinGecko verisi alinamadi ({type(exc).__name__}: {exc}); sentetik veriye gecildi."
                )

        frame = self._fetch_synthetic(symbol, timeframe, limit)
        result = MarketDataResult(
            frame=frame,
            source="synthetic",
            symbol=symbol,
            timeframe=timeframe,
            warnings=warnings,
        )
        result.validate()
        self._write_cache(cache_path, frame, "synthetic")
        self.last_source = "synthetic"
        self.last_warnings = warnings
        return result


    def fetch_order_book_summary(self, symbol: Optional[str] = None, depth: Optional[int] = None) -> Dict[str, Any]:
        """Emir defteri dengesizligi (bid/ask imbalance) ozeti."""
        symbol = symbol or self.active_symbol or self.settings.exchange.symbol
        depth = int(depth or self.settings.exchange.order_book_depth)
        if self.settings.data.provider == "synthetic":
            return {"source": "synthetic", "imbalance": 0.0, "spread_bps": 0.0, "depth": depth}
        try:
            exchange = self._get_exchange(self.active_exchange_id or self.settings.exchange.id)
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
        for exchange in list(self._exchanges.values()):
            try:
                exchange.close()
            except Exception:  # pragma: no cover - en iyi caba
                pass
        self._exchanges.clear()
        self._exchange = None

    # --------------------------------------------------------------- ccxt
    def _get_exchange(self, exchange_id: Optional[str] = None):
        exchange_id = exchange_id or self.settings.exchange.id
        if exchange_id in self._exchanges:
            self._exchange = self._exchanges[exchange_id]
            return self._exchange
        try:
            import ccxt  # yerel import: kutuphane yoksa sistem sentetik moda duser
        except ImportError as exc:  # pragma: no cover
            raise DataError("ccxt kurulu degil") from exc

        if not hasattr(ccxt, exchange_id):
            raise DataError(f"ccxt '{exchange_id}' borsasini desteklemiyor")
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(
            {
                "enableRateLimit": self.settings.exchange.enable_rate_limit,
                "timeout": self.settings.exchange.request_timeout_ms,
                "options": {"defaultType": "spot"},
            }
        )
        self._exchanges[exchange_id] = exchange
        self._exchange = exchange
        return exchange

    def _candidate_exchanges(self) -> List[str]:
        primary = self.settings.exchange.id
        chain = [primary]
        for fallback in self.settings.data.fallback_exchanges:
            if fallback and fallback not in chain:
                chain.append(fallback)
        return chain

    @staticmethod
    def _symbol_candidates(symbol: str) -> List[str]:
        candidates = [symbol]
        if "/" in symbol:
            base, quote = symbol.split("/", 1)
            if quote.upper() == "USDT":
                candidates.append(f"{base}/USD")
        return candidates

    def _fetch_paged(self, exchange: Any, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
        """Gecmise dogru sayfalayarak ``limit`` kadar mum toplar."""
        timeframe_ms = int(exchange.parse_timeframe(timeframe)) * 1000
        since = exchange.milliseconds() - int((limit + 10) * timeframe_ms)
        rows: List[List[float]] = []
        guard = 0
        while len(rows) < limit and guard < 10:
            guard += 1
            batch_size = int(min(720, max(1, limit - len(rows))))
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch_size)
            except Exception:
                if rows:
                    break
                # Bazi borsalar 'since' parametresini reddeder: son N mumu al
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=batch_size)
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 2:
                break
            since = int(batch[-1][0]) + timeframe_ms
        deduped: Dict[int, List[float]] = {}
        for row in rows:
            deduped[int(row[0])] = list(row)
        return [deduped[key] for key in sorted(deduped)][-limit:]

    def _fetch_ccxt(self, symbol: str, timeframe: str, limit: int) -> tuple[pd.DataFrame, str]:
        errors: List[str] = []
        for exchange_id in self._candidate_exchanges():
            try:
                exchange = self._get_exchange(exchange_id)
                if not getattr(exchange, "markets", None):
                    exchange.load_markets()
                resolved_symbol = None
                for candidate in self._symbol_candidates(symbol):
                    if candidate in exchange.markets:
                        resolved_symbol = candidate
                        break
                if resolved_symbol is None:
                    raise DataError(f"{symbol} sembolu {exchange_id} uzerinde yok")

                raw = self._fetch_paged(exchange, resolved_symbol, timeframe, limit)
                if not raw:
                    raise DataError("Borsa bos OHLCV dondurdu")
                frame = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
                frame = frame.dropna().sort_values("timestamp").reset_index(drop=True)
                if len(frame) < 150:
                    raise DataError(f"yetersiz veri ({len(frame)} bar)")

                self.active_exchange_id = exchange_id
                self.active_symbol = resolved_symbol
                if exchange_id != self.settings.exchange.id:
                    self.last_warnings.append(
                        f"{self.settings.exchange.id} kullanilamadi; {exchange_id} borsasindan "
                        f"veri alindi."
                    )
                if resolved_symbol != symbol:
                    self.last_warnings.append(f"{symbol} yerine {resolved_symbol} kullanildi.")
                return frame, f"ccxt:{exchange_id}"
            except Exception as exc:  # pragma: no cover - ag bagimli
                errors.append(f"{exchange_id}: {type(exc).__name__}: {exc}")
                continue
        raise DataError("CCXT fetch basarisiz: " + " | ".join(errors[:3]))


    # ------------------------------------------------------------ coingecko
    def _fetch_coingecko(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Anahtarsiz REST kaynagi (CCXT tamamen engelliyse).

        CoinGecko market_chart yalnizca fiyat/hacim noktalari verir; OHLCV
        mumlari bu noktalardan **muhafazakar** sekilde turetilir:
        ``open = onceki kapanis``, ``high/low = max/min(open, close)``.
        Yani fitil (wick) verisi yoktur; stop/take-profit tetiklenmesi gercek
        piyasadan daha az iyimser olur.
        """
        base = symbol.split("/")[0].upper()
        coin_id = COINGECKO_IDS.get(base)
        if not coin_id:
            raise DataError(f"CoinGecko bu sembolu desteklemiyor: {symbol}")

        minutes = timeframe_minutes(timeframe)
        days = 90 if minutes <= 60 else 365
        interval = "hourly" if minutes <= 60 else "daily"
        url = (
            f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
            f"?vs_currency=usd&days={days}&interval={interval}"
        )
        request = urllib.request.Request(
            url, headers={"User-Agent": "agency-system/1.0", "Accept": "application/json"}
        )
        timeout = max(5.0, float(self.settings.exchange.request_timeout_ms) / 1000.0)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise DataError(f"CoinGecko HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DataError(f"CoinGecko baglanti hatasi: {exc}") from exc

        prices = payload.get("prices") or []
        volumes = payload.get("total_volumes") or []
        if len(prices) < 100:
            raise DataError(f"CoinGecko yetersiz veri ({len(prices)} nokta)")

        price_df = pd.DataFrame(prices, columns=["timestamp", "close"])
        volume_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        frame = price_df.merge(volume_df, on="timestamp", how="left")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = (
            frame.drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        frame["open"] = frame["close"].shift(1).fillna(frame["close"])
        frame["high"] = frame[["open", "close"]].max(axis=1)
        frame["low"] = frame[["open", "close"]].min(axis=1)
        frame["volume"] = frame["volume"].fillna(float(frame["volume"].median() or 0.0))
        frame = frame[OHLCV_COLUMNS]

        if minutes > 60:
            frame = (
                frame.set_index("timestamp")
                .resample(f"{minutes}min")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
                .reset_index()
            )
        return frame.tail(limit).reset_index(drop=True)

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
    def _cache_meta_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".meta.json")

    @classmethod
    def _write_cache(cls, path: Path, frame: pd.DataFrame, source: str = "unknown") -> None:
        """CSV + kaynak meta dosyasi yazar (onbellekten okurken kaynak korunur)."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            frame.to_csv(tmp, index=False)
            tmp.replace(path)
            write_json(
                cls._cache_meta_path(path),
                {"source": source, "bars": int(len(frame)), "written_at": iso_now()},
            )
        except OSError:  # pragma: no cover - disk sorunu
            pass


