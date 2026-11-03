"""Pytest ortak fixture'lari.

Tum testler gecici dizinlerde calisir: gercek ``runtime/`` ve
``agency/strategies/library`` dosyalarina dokunulmaz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agency.config import Settings  # noqa: E402
from agency.data.market_data import MarketDataProvider  # noqa: E402
from agency.memory.database import Database  # noqa: E402
from agency.memory.repositories import MemoryStore  # noqa: E402
from agency.memory.vector_store import FeatureVectorizer  # noqa: E402
from agency.schemas import StrategySpec  # noqa: E402
from agency.strategies.registry import StrategyRegistry  # noqa: E402


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    overrides = {
        "exchange": {"symbol": "BTC/USDT", "timeframe": "1h", "limit": 900},
        "data": {
            "provider": "synthetic",
            "cache_dir": str(tmp_path / "cache"),
            "synthetic_bars": 900,
            "synthetic_seed": 424242,
            "rest_fallback_enabled": False,
        },
        "backtest": {"min_trades": 3, "run_timeout_seconds": 120},
        "targets": {
            "candidates_per_cycle": 1,
            "max_rework_attempts": 1,
            "cycle_sleep_seconds": 0.0,
            "max_cycles_per_run": 1,
        },
        "memory": {"database": str(tmp_path / "agency.db"), "vector_dim": 64},
        "logging": {"dir": str(tmp_path / "logs"), "jsonl": True},
        "llm": {"provider": "offline"},
        "versioning": {
            "library_dir": str(tmp_path / "library"),
            "archive_dir": str(tmp_path / "archive"),
        },
        "runtime": {
            "state_file": str(tmp_path / "state.json"),
            "reports_dir": str(tmp_path / "reports"),
            "backend": "native",
        },
    }
    loaded = Settings.load(overrides=overrides, env={})
    loaded.ensure_directories()
    return loaded


@pytest.fixture()
def market_data(settings: Settings):
    provider = MarketDataProvider(settings)
    result = provider.fetch_ohlcv(use_cache=False)
    yield result
    provider.close()


@pytest.fixture()
def memory(settings: Settings) -> MemoryStore:
    database = Database(settings.memory.database)
    store = MemoryStore(
        database, vectorizer=FeatureVectorizer(settings.memory.vector_dim)
    )
    yield store
    store.close()


@pytest.fixture()
def registry(settings: Settings) -> StrategyRegistry:
    return StrategyRegistry(settings)


@pytest.fixture()
def agent_context(settings: Settings, memory: MemoryStore, registry: StrategyRegistry):
    from agency.agents.base import AgentContext
    from agency.agents.planner import StrategyPlanner
    from agency.data.market_data import MarketDataProvider
    from agency.llm import build_llm
    from agency.logging_utils import build_logger

    logger = build_logger(settings.logging.dir, console=False, run_id="test")
    context = AgentContext(
        settings=settings,
        memory=memory,
        logger=logger,
        llm=build_llm(settings),
        registry=registry,
        data_provider=MarketDataProvider(settings),
        planner=StrategyPlanner(settings),
        cycle=1,
        run_id="test",
    )
    yield context
    context.data_provider.close()


@pytest.fixture()
def config_file(settings: Settings, tmp_path: Path) -> Path:
    import yaml

    path = tmp_path / "settings.yaml"
    path.write_text(
        yaml.safe_dump(settings.as_dict(relative=False), sort_keys=False), encoding="utf-8"
    )
    return path



@pytest.fixture()
def sample_spec() -> StrategySpec:
    return StrategySpec.from_dict(
        {
            "name": "ema_trend_test",
            "family": "trend_following",
            "thesis": "EMA kesisimi + ADX filtresi",
            "timeframe": "1h",
            "indicators": [
                {"key": "ema_fast", "kind": "ema", "params": {"period": 12}},
                {"key": "ema_slow", "kind": "ema", "params": {"period": 48}},
                {"key": "adx_14", "kind": "adx", "output": "value", "params": {"period": 14}},
                {"key": "rsi", "kind": "rsi", "params": {"period": 14}},
            ],
            "entry_long": [
                {"left": "ema_fast", "op": "cross_above", "right": "ema_slow"},
                {"left": "rsi", "op": "<", "right": 70},
            ],
            "exit_long": [{"left": "ema_fast", "op": "cross_below", "right": "ema_slow"}],
            "filters": [{"left": "adx_14", "op": ">", "right": 15}],
            "stop_loss_atr": 2.0,
            "take_profit_atr": 4.0,
            "risk_per_trade_pct": 1.5,
        }
    )
