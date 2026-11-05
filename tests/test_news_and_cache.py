"""Haber tarayici ve piyasa verisi (cache) testleri."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agency.agents.news import NewsScanner
from agency.data.market_data import MarketDataProvider


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Bitcoin ETF approval triggers record inflow", "bullish"),
        ("Exchange hack causes massive liquidation and selloff", "bearish"),
        ("Bitcoin trades sideways ahead of FOMC", "neutral"),
        ("yükseliş rekor onay", "bullish"),
        ("düşüş çöküş yasak", "bearish"),
    ],
)
def test_news_sentiment(text: str, expected: str) -> None:
    assert NewsScanner._sentiment(text.lower()) == expected


def test_news_digest_format() -> None:
    items = [
        {"title": "A  very   long   title", "sentiment": "bullish"},
        {"title": "Short news", "sentiment": "bearish"},
    ]
    digest = NewsScanner.digest(items)
    assert digest[0].startswith("[bullish]")
    assert "  " not in digest[0]  # bosluklar normalize edilir


def test_news_disabled_returns_empty() -> None:
    scanner = NewsScanner(endpoint="http://127.0.0.1:9/none", enabled=False)
    assert scanner.fetch() == []
    assert scanner.last_error is None


def test_news_network_failure_is_contained() -> None:
    scanner = NewsScanner(endpoint="http://127.0.0.1:9/none", timeout=1, enabled=True)
    assert scanner.fetch() == []
    assert scanner.last_error is not None


def test_cache_preserves_source(settings, tmp_path: Path) -> None:
    settings.data.cache_dir = tmp_path / "cache"
    provider = MarketDataProvider(settings)
    try:
        fresh = provider.fetch_ohlcv(use_cache=False)
        assert fresh.source == "synthetic"
        assert fresh.cached is False

        cached = provider.fetch_ohlcv(use_cache=True)
        assert cached.cached is True
        # Onbellek bir tasima bicimidir; kaynak korunmali
        assert cached.source == "synthetic"

        meta_files = list((tmp_path / "cache").glob("*.meta.json"))
        assert meta_files
        payload = json.loads(meta_files[0].read_text(encoding="utf-8"))
        assert payload["source"] == "synthetic"
        assert payload["bars"] == len(fresh.frame)
    finally:
        provider.close()


def test_cache_disabled_reads_fresh(settings, tmp_path: Path) -> None:
    settings.data.cache_dir = tmp_path / "cache"
    provider = MarketDataProvider(settings)
    try:
        first = provider.fetch_ohlcv(use_cache=False)
        second = provider.fetch_ohlcv(use_cache=False)
        assert first.cached is False and second.cached is False
    finally:
        provider.close()
