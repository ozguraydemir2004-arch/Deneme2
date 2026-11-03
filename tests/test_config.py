"""Konfigurasyon katmani testleri."""

from __future__ import annotations

from pathlib import Path

import pytest

from agency.config import ConfigError, Settings, _env_overrides, deep_merge


def test_defaults_load(settings: Settings) -> None:
    assert settings.exchange.symbol == "BTC/USDT"
    assert settings.data.provider == "synthetic"
    assert settings.llm.provider == "offline"
    assert settings.runtime.backend == "native"
    assert abs(sum(settings.versioning.score_weights.values()) - 1.0) < 1e-9


def test_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = {
        "AGENCY_SYMBOL": "ETH/USDT",
        "AGENCY_TIMEFRAME": "4h",
        "AGENCY_MIN_SCORE": "61.5",
        "AGENCY_DB_PATH": str(tmp_path / "x.db"),
        "AGENCY_NEWS_ENABLED": "false",
    }
    overrides = _env_overrides(env)
    assert overrides["exchange"]["symbol"] == "ETH/USDT"
    assert overrides["targets"]["min_score"] == 61.5
    assert overrides["research"]["news_enabled"] is False

    settings = Settings.load(env=env)
    assert settings.exchange.timeframe == "4h"
    assert str(settings.memory.database).endswith("x.db")


def test_invalid_timeframe_rejected() -> None:
    with pytest.raises(ConfigError):
        Settings.load(overrides={"exchange": {"timeframe": "7m"}}, env={})


def test_invalid_score_weights_rejected() -> None:
    with pytest.raises(ConfigError):
        Settings.load(
            overrides={"versioning": {"score_weights": {"net_profit": 0.5}}}, env={}
        )


def test_invalid_provider_rejected() -> None:
    with pytest.raises(ConfigError):
        Settings.load(overrides={"llm": {"provider": "magic"}}, env={})


def test_deep_merge_is_non_destructive() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 9}, "e": 5}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 9, "c": 2}, "d": 3, "e": 5}
    assert base["a"]["b"] == 1


def test_relative_paths_are_resolved(settings: Settings) -> None:
    assert settings.memory.database.is_absolute()
    assert settings.versioning.library_dir.is_absolute()
    payload = settings.as_dict()
    assert isinstance(payload["memory"]["database"], str)


def test_ensure_directories(settings: Settings) -> None:
    for path in (
        settings.runtime.reports_dir,
        settings.logging.dir,
        settings.versioning.library_dir,
        settings.versioning.archive_dir,
    ):
        assert path.is_dir()
