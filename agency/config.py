"""Konfigurasyon yukleyici.

Oncelik sirasi (dusukten yuksege):
    1. ``config/settings.yaml`` varsayilanlari
    2. ``.env`` / ortam degiskenleri (``AGENCY_*``)
    3. Programatik override'lar (CLI argumanlari)

Tasarim notu: pydantic zorunlu degildir; dataclass + tip donusumu ile ayni
dogrulama saglanir, boylece cekirdek calisma zamani minimum bagimlilikla
ayakta kalir.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class ConfigError(ValueError):
    """Gecersiz veya eksik konfigurasyon."""


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------
def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "evet"}


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(float(value))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Ic ice sozlukleri bozmadan birlestirir."""
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_dotenv(path: Path) -> None:
    """Bagimliliksiz .env okuyucu (mevcut ortam degiskenlerini ezmez)."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


# ---------------------------------------------------------------------------
# Bolumler
# ---------------------------------------------------------------------------
@dataclass
class ExchangeSettings:
    id: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 1500
    order_book_depth: int = 50
    request_timeout_ms: int = 25000
    enable_rate_limit: bool = True
    max_retries: int = 2


@dataclass
class DataSettings:
    provider: str = "auto"
    cache_dir: Path = Path("runtime/cache")
    cache_ttl_minutes: int = 15
    synthetic_bars: int = 2880
    synthetic_seed: int = 1337
    synthetic_start_price: float = 42000.0
    synthetic_annual_vol: float = 0.55
    synthetic_drift: float = 0.18


@dataclass
class BacktestSettings:
    initial_capital: float = 10000.0
    fee_bps: float = 7.5
    slippage_bps: float = 2.0
    allow_short: bool = False
    min_trades: int = 8
    risk_per_trade_pct: float = 1.5
    run_timeout_seconds: int = 120
    max_memory_mb: int = 1536
    train_split: float = 0.7


@dataclass
class TargetSettings:
    min_net_profit_pct: float = 3.0
    min_win_rate: float = 45.0
    max_drawdown_pct: float = 25.0
    min_sharpe: float = 0.30
    min_profit_factor: float = 1.15
    min_score: float = 55.0
    max_rework_attempts: int = 3
    max_cycles_per_run: int = 0
    cycle_sleep_seconds: float = 60.0
    candidates_per_cycle: int = 3


@dataclass
class MemorySettings:
    database: Path = Path("runtime/agency.db")
    vector_dim: int = 256
    top_k_snapshots: int = 6
    top_k_strategies: int = 5
    winner_forward_return_pct: float = 1.5
    winner_horizon_bars: int = 12
    max_snapshots_per_cycle: int = 400


@dataclass
class ResearchSettings:
    news_enabled: bool = True
    news_endpoint: str = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    news_timeout_seconds: int = 10
    max_news_items: int = 12
    playbook: Path = Path("agency/knowledge/strategy_playbook.json")


@dataclass
class LoggingSettings:
    dir: Path = Path("runtime/logs")
    level: str = "INFO"
    jsonl: bool = True
    log_llm_calls: bool = True


@dataclass
class LLMSettings:
    provider: str = "auto"
    model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-sonnet-latest"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.35
    max_tokens: int = 2200
    timeout_seconds: int = 90
    max_retries: int = 2


@dataclass
class VersioningSettings:
    library_dir: Path = Path("agency/strategies/library")
    archive_dir: Path = Path("runtime/archive")
    score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "net_profit": 0.35,
            "win_rate": 0.20,
            "drawdown": 0.20,
            "sharpe": 0.15,
            "profit_factor": 0.10,
        }
    )
    elite_score: float = 70.0


@dataclass
class RuntimeSettings:
    state_file: Path = Path("runtime/state.json")
    reports_dir: Path = Path("runtime/reports")
    quote_currency: str = "USDT"


# ---------------------------------------------------------------------------
# Kok konfigurasyon
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    exchange: ExchangeSettings = field(default_factory=ExchangeSettings)
    data: DataSettings = field(default_factory=DataSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    targets: TargetSettings = field(default_factory=TargetSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    research: ResearchSettings = field(default_factory=ResearchSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    versioning: VersioningSettings = field(default_factory=VersioningSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    project_root: Path = PROJECT_ROOT
    source_path: Optional[Path] = None

    # -- yukleme ------------------------------------------------------------
    @classmethod
    def load(
        cls,
        config_path: Optional[Path | str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "Settings":
        env_map = dict(os.environ if env is None else env)
        if env is None:
            _load_dotenv(PROJECT_ROOT / ".env")

        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        raw: Dict[str, Any] = {}
        if path.is_file():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, Mapping):
                raise ConfigError(f"{path} bir sozluk (mapping) icermeli.")
            raw = dict(loaded)
        elif config_path is not None:
            raise ConfigError(f"Konfigurasyon dosyasi bulunamadi: {path}")

        raw = deep_merge(raw, _env_overrides(env_map))
        if overrides:
            raw = deep_merge(raw, overrides)

        settings = cls._from_mapping(raw, project_root=PROJECT_ROOT)
        settings.source_path = path if path.is_file() else None
        settings.validate()
        return settings

    @classmethod
    def _from_mapping(cls, raw: Mapping[str, Any], project_root: Path) -> "Settings":
        def section(name: str) -> Dict[str, Any]:
            value = raw.get(name) or {}
            if not isinstance(value, Mapping):
                raise ConfigError(f"'{name}' bolumu sozluk olmali.")
            return dict(value)

        defaults = cls()

        def build(dc_type, name: str, path_fields: Iterable[str] = ()):
            data = section(name)
            base = asdict(getattr(defaults, name))
            merged = deep_merge(base, data)
            for f in path_fields:
                merged[f] = _resolve_path(str(merged[f]), project_root)
            for key in list(merged):
                if key not in base:
                    merged.pop(key)
            return dc_type(**merged)

        settings = cls(
            exchange=build(ExchangeSettings, "exchange"),
            data=build(DataSettings, "data", ("cache_dir",)),
            backtest=build(BacktestSettings, "backtest"),
            targets=build(TargetSettings, "targets"),
            memory=build(MemorySettings, "memory", ("database",)),
            research=build(ResearchSettings, "research", ("playbook",)),
            logging=build(LoggingSettings, "logging", ("dir",)),
            llm=build(LLMSettings, "llm"),
            versioning=build(VersioningSettings, "versioning", ("library_dir", "archive_dir")),
            runtime=build(RuntimeSettings, "runtime", ("state_file", "reports_dir")),
        )
        settings.project_root = project_root
        return settings

    # -- dogrulama ----------------------------------------------------------
    def validate(self) -> None:
        if self.exchange.timeframe not in _VALID_TIMEFRAMES:
            raise ConfigError(
                f"Gecersiz timeframe '{self.exchange.timeframe}'. "
                f"Desteklenen: {', '.join(sorted(_VALID_TIMEFRAMES))}"
            )
        if self.data.provider not in {"auto", "ccxt", "synthetic"}:
            raise ConfigError("data.provider auto|ccxt|synthetic olmali.")
        if self.llm.provider not in {"auto", "openai", "anthropic", "offline"}:
            raise ConfigError("llm.provider auto|openai|anthropic|offline olmali.")
        if not 0.0 < self.backtest.train_split < 1.0:
            raise ConfigError("backtest.train_split 0 ile 1 arasinda olmali.")
        if self.backtest.initial_capital <= 0:
            raise ConfigError("backtest.initial_capital pozitif olmali.")
        if self.targets.candidates_per_cycle < 1:
            raise ConfigError("targets.candidates_per_cycle en az 1 olmali.")
        if self.targets.max_rework_attempts < 0:
            raise ConfigError("targets.max_rework_attempts negatif olamaz.")
        if self.memory.vector_dim < 8:
            raise ConfigError("memory.vector_dim en az 8 olmali.")
        weights = self.versioning.score_weights
        if not weights or abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ConfigError("versioning.score_weights toplami 1.0 olmali.")
        if not self.versioning.library_dir.is_dir():
            self.versioning.library_dir.mkdir(parents=True, exist_ok=True)

    # -- yardimcilar --------------------------------------------------------
    def ensure_directories(self) -> None:
        """Calisma zamani dizinlerini olusturur."""
        for directory in (
            self.runtime.reports_dir,
            self.logging.dir,
            self.versioning.library_dir,
            self.versioning.archive_dir,
            self.data.cache_dir,
            self.memory.database.parent,
            self.runtime.state_file.parent,
        ):
            Path(directory).mkdir(parents=True, exist_ok=True)

    @property
    def symbol_slug(self) -> str:
        return self.exchange.symbol.replace("/", "").replace(":", "").lower()

    def as_dict(self, relative: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("project_root", None)
        data.pop("source_path", None)
        if relative:
            for section in data.values():
                if not isinstance(section, dict):
                    continue
                for key, value in list(section.items()):
                    if isinstance(value, Path):
                        try:
                            section[key] = str(value.relative_to(self.project_root))
                        except ValueError:
                            section[key] = str(value)
        return data


_VALID_TIMEFRAMES = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}


def _env_overrides(env: Mapping[str, str]) -> Dict[str, Any]:
    """``AGENCY_<SECTION>_<FIELD>`` kalibindaki degiskenleri sozluge cevirir."""
    mapping: Dict[str, tuple[str, str, Any]] = {
        "AGENCY_EXCHANGE_ID": ("exchange", "id", str),
        "AGENCY_SYMBOL": ("exchange", "symbol", str),
        "AGENCY_TIMEFRAME": ("exchange", "timeframe", str),
        "AGENCY_EXCHANGE_LIMIT": ("exchange", "limit", _as_int),
        "AGENCY_DATA_PROVIDER": ("data", "provider", str),
        "AGENCY_SYNTHETIC_BARS": ("data", "synthetic_bars", _as_int),
        "AGENCY_SYNTHETIC_SEED": ("data", "synthetic_seed", _as_int),
        "AGENCY_DB_PATH": ("memory", "database", str),
        "AGENCY_LOG_DIR": ("logging", "dir", str),
        "AGENCY_LOG_LEVEL": ("logging", "level", str),
        "AGENCY_MIN_NET_PROFIT_PCT": ("targets", "min_net_profit_pct", _as_float),
        "AGENCY_MIN_WIN_RATE": ("targets", "min_win_rate", _as_float),
        "AGENCY_MAX_DRAWDOWN_PCT": ("targets", "max_drawdown_pct", _as_float),
        "AGENCY_MIN_SCORE": ("targets", "min_score", _as_float),
        "AGENCY_MAX_CYCLES": ("targets", "max_cycles_per_run", _as_int),
        "AGENCY_CYCLE_SLEEP_SECONDS": ("targets", "cycle_sleep_seconds", _as_float),
        "AGENCY_MAX_REWORK_ATTEMPTS": ("targets", "max_rework_attempts", _as_int),
        "AGENCY_CANDIDATES_PER_CYCLE": ("targets", "candidates_per_cycle", _as_int),
        "AGENCY_INITIAL_CAPITAL": ("backtest", "initial_capital", _as_float),
        "AGENCY_FEE_BPS": ("backtest", "fee_bps", _as_float),
        "AGENCY_RISK_PER_TRADE_PCT": ("backtest", "risk_per_trade_pct", _as_float),
        "AGENCY_LLM_PROVIDER": ("llm", "provider", str),
        "AGENCY_LLM_MODEL": ("llm", "model", str),
        "AGENCY_ANTHROPIC_MODEL": ("llm", "anthropic_model", str),
        "OPENAI_BASE_URL": ("llm", "base_url", str),
        "AGENCY_LLM_TEMPERATURE": ("llm", "temperature", _as_float),
        "AGENCY_LLM_MAX_TOKENS": ("llm", "max_tokens", _as_int),
        "AGENCY_VECTOR_DIM": ("memory", "vector_dim", _as_int),
        "AGENCY_ELITE_SCORE": ("versioning", "elite_score", _as_float),
        "AGENCY_NEWS_ENABLED": ("research", "news_enabled", _as_bool),
        "AGENCY_LOG_LLM_CALLS": ("logging", "log_llm_calls", _as_bool),
    }
    result: Dict[str, Any] = {}
    for env_key, (section, key, caster) in mapping.items():
        if env_key not in env:
            continue
        raw_value = env[env_key]
        if raw_value is None or str(raw_value).strip() == "":
            continue
        try:
            converted = caster(raw_value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - koruma
            raise ConfigError(f"{env_key} degeri gecersiz: {raw_value!r}") from exc
        result.setdefault(section, {})[key] = converted
    return result


def load_settings(
    config_path: Optional[Path | str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Settings:
    """Kolaylik fonksiyonu: konfigurasyonu yukler ve dizinleri hazirlar."""
    settings = Settings.load(config_path=config_path, overrides=overrides)
    settings.ensure_directories()
    return settings



