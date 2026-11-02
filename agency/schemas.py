"""Ajanlar arasi veri sozlesmeleri (dataclass + JSON).

Bu semalar tum ajanlarin ortak dilidir:

    ResearchReport  -> arastirma ajani cikisi (piyasa durumu + aday spec'ler)
    StrategySpec    -> stratejinin kod-oncesi bildirimsel tanimi
    BacktestReport  -> simulasyon sonucu (metrikler + hata matrisi)
    ReviewDecision  -> critic karari (skor, verdikt, direktifler)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agency.data.indicators import CONSTANT_OPERANDS, INDICATOR_KINDS, RAW_OPERANDS
from agency.exceptions import StrategyValidationError
from agency.utils import round_or_none, sha256_text, slugify

VALID_OPERATORS = {">", ">=", "<", "<=", "cross_above", "cross_below"}
VALID_FAMILIES = (
    "trend_following",
    "mean_reversion",
    "breakout",
    "momentum",
    "volatility_compression",
    "hybrid",
)
VALID_ORIGINS = ("research", "coder", "memory", "bootstrap", "manual")
VALID_VERDICTS = ("promote", "rework", "archive")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class Rule:
    """Tek bir mantiksal kosul: ``left <op> right``."""

    left: str
    op: str
    right: Any

    def validate(self, allowed_operands: Sequence[str]) -> List[str]:
        errors: List[str] = []
        if not isinstance(self.left, str) or not self.left:
            errors.append("rule.left bos olamaz")
        elif self.left not in allowed_operands:
            errors.append(f"bilinmeyen operand: {self.left}")
        if self.op not in VALID_OPERATORS:
            errors.append(f"gecersiz operator: {self.op!r} (izinli: {sorted(VALID_OPERATORS)})")
        right = self.right
        if isinstance(right, (int, float)) and not isinstance(right, bool):
            pass
        elif isinstance(right, str) and right in allowed_operands:
            pass
        elif isinstance(right, str):
            try:
                float(right)
            except ValueError:
                errors.append(f"bilinmeyen sag operand: {right}")
        else:
            errors.append(f"gecersiz sag operand tipi: {type(right).__name__}")
        return errors

    def describe(self) -> str:
        return f"{self.left} {self.op} {self.right}"

    def to_dict(self) -> Dict[str, Any]:
        return {"left": self.left, "op": self.op, "right": self.right}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Rule":
        if not isinstance(data, Mapping):
            raise StrategyValidationError(f"Kural sozluk olmali: {data!r}")
        op = str(data.get("op", "")).strip().lower()
        aliases = {
            "crosses_above": "cross_above",
            "cross_over": "cross_above",
            "crossover": "cross_above",
            "crosses_below": "cross_below",
            "cross_under": "cross_below",
            "crossunder": "cross_below",
        }
        op = aliases.get(op, op)
        return cls(
            left=str(data.get("left", "")).strip(),
            op=op,
            right=data.get("right"),
        )


@dataclass
class IndicatorSpec:
    """Katalogdan secilmis tek bir indikator kurulumu."""

    key: str
    kind: str
    output: str = "value"
    params: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> List[str]:
        errors: List[str] = []
        key = str(self.key or "").strip()
        if not key:
            errors.append("indicator.key bos olamaz")
        elif not key.replace("_", "").isalnum() or key[0].isdigit():
            errors.append(f"indicator.key gecersiz (snake_case olmali): {key!r}")
        if self.kind not in INDICATOR_KINDS:
            errors.append(f"bilinmeyen indikator turu: {self.kind!r}")
            return errors
        meta = INDICATOR_KINDS[self.kind]
        if self.output not in meta["outputs"]:
            errors.append(
                f"{self.kind} icin gecersiz output {self.output!r}; izinli: {meta['outputs']}"
            )
        for name, pmeta in meta["params"].items():
            if name not in self.params:
                errors.append(f"{self.kind}: eksik parametre {name!r}")
                continue
            value = self.params[name]
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"{self.kind}.{name} sayisal olmali: {value!r}")
                continue
            if not (pmeta["min"] <= number <= pmeta["max"]):
                errors.append(
                    f"{self.kind}.{name}={number} aralik disi "
                    f"({pmeta['min']}..{pmeta['max']})"
                )
        extra = set(self.params) - set(meta["params"])
        if extra:
            errors.append(f"{self.kind} icin bilinmeyen parametreler: {sorted(extra)}")
        return errors

    def resolved_params(self) -> Dict[str, float]:
        meta = INDICATOR_KINDS.get(self.kind, {"params": {}})
        resolved: Dict[str, float] = {}
        for name, pmeta in meta["params"].items():
            raw = self.params.get(name, pmeta["default"])
            resolved[name] = float(raw) if pmeta["type"] == "float" else int(round(float(raw)))
        return resolved

    def column_name(self) -> str:
        return self.key

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "output": self.output,
            "params": self.resolved_params(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IndicatorSpec":
        if not isinstance(data, Mapping):
            raise StrategyValidationError(f"Indikator tanimi sozluk olmali: {data!r}")
        kind = str(data.get("kind", "")).strip().lower()
        params = data.get("params") or {}
        if not isinstance(params, Mapping):
            raise StrategyValidationError("indicator.params sozluk olmali")
        key = str(data.get("key") or kind).strip().lower()
        outputs = INDICATOR_KINDS.get(kind, {}).get("outputs", ["value"])
        output = str(data.get("output") or outputs[0]).strip().lower()
        return cls(key=key, kind=kind, output=output, params=dict(params))


@dataclass
class StrategySpec:
    """Stratejinin bildirimsel tanimi (kodun tek kaynagi)."""

    name: str
    family: str = "hybrid"
    thesis: str = ""
    timeframe: str = "1h"
    indicators: List[IndicatorSpec] = field(default_factory=list)
    entry_long: List[Rule] = field(default_factory=list)
    exit_long: List[Rule] = field(default_factory=list)
    filters: List[Rule] = field(default_factory=list)
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 3.0
    trailing_stop_atr: float = 0.0
    risk_per_trade_pct: float = 1.5
    cooldown_bars: int = 0
    max_holding_bars: int = 0
    params: Dict[str, float] = field(default_factory=dict)
    origin: str = "research"
    parent_version: Optional[str] = None
    memory_refs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # -- dogrulama ----------------------------------------------------------
    def allowed_operands(self) -> List[str]:
        names = [ind.key for ind in self.indicators]
        return list(dict.fromkeys(names + RAW_OPERANDS + list(CONSTANT_OPERANDS)))

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not str(self.name or "").strip():
            errors.append("spec.name bos olamaz")
        if self.family not in VALID_FAMILIES:
            errors.append(f"gecersiz family: {self.family!r}; izinli: {list(VALID_FAMILIES)}")
        if self.origin not in VALID_ORIGINS:
            errors.append(f"gecersiz origin: {self.origin!r}")
        if not self.indicators:
            errors.append("en az bir indikator gerekli")
        if not self.entry_long:
            errors.append("en az bir giris kurali (entry_long) gerekli")
        if len(self.indicators) > 8:
            errors.append("en fazla 8 indikator kullanilabilir")

        seen_keys = set()
        for ind in self.indicators:
            errors.extend(ind.validate())
            if ind.key in seen_keys:
                errors.append(f"yinelenen indikator anahtari: {ind.key}")
            seen_keys.add(ind.key)

        allowed = self.allowed_operands()
        for group_name, group in (
            ("entry_long", self.entry_long),
            ("exit_long", self.exit_long),
            ("filters", self.filters),
        ):
            if len(group) > 8:
                errors.append(f"{group_name} en fazla 8 kural icerebilir")
            for rule in group:
                for err in rule.validate(allowed):
                    errors.append(f"{group_name}: {err}")

        ranges = {
            "stop_loss_atr": (0.2, 10.0),
            "take_profit_atr": (0.0, 30.0),
            "trailing_stop_atr": (0.0, 10.0),
            "risk_per_trade_pct": (0.1, 10.0),
            "cooldown_bars": (0, 500),
            "max_holding_bars": (0, 5000),
        }
        for attr, (low, high) in ranges.items():
            value = getattr(self, attr)
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"{attr} sayisal olmali: {value!r}")
                continue
            if not (low <= number <= high):
                errors.append(f"{attr}={number} aralik disi ({low}..{high})")
        return errors

    def ensure_valid(self) -> "StrategySpec":
        errors = self.validate()
        if errors:
            raise StrategyValidationError(
                "Strateji spec'i gecersiz: " + "; ".join(errors[:10]),
                details={"errors": errors, "spec": self.to_dict()},
            )
        return self

    # -- kimlik / ozet ------------------------------------------------------
    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return sha256_text(payload)[:16]

    def canonical(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "indicators": [ind.to_dict() for ind in self.indicators],
            "entry_long": [r.to_dict() for r in self.entry_long],
            "exit_long": [r.to_dict() for r in self.exit_long],
            "filters": [r.to_dict() for r in self.filters],
            "stop_loss_atr": round(float(self.stop_loss_atr), 4),
            "take_profit_atr": round(float(self.take_profit_atr), 4),
            "trailing_stop_atr": round(float(self.trailing_stop_atr), 4),
            "risk_per_trade_pct": round(float(self.risk_per_trade_pct), 4),
            "cooldown_bars": int(self.cooldown_bars),
            "max_holding_bars": int(self.max_holding_bars),
        }

    def describe(self) -> str:
        indicators = ", ".join(f"{i.kind}({i.resolved_params()})" for i in self.indicators)
        entries = " AND ".join(r.describe() for r in self.entry_long) or "-"
        exits = " AND ".join(r.describe() for r in self.exit_long) or "- (sadece SL/TP)"
        filters = " AND ".join(r.describe() for r in self.filters) or "-"
        return (
            f"{self.name} [{self.family}] ind={indicators} | gir={entries} | "
            f"cik={exits} | filtre={filters} | SL={self.stop_loss_atr}ATR "
            f"TP={self.take_profit_atr}ATR risk={self.risk_per_trade_pct}%"
        )

    def with_mutation(self, **changes: Any) -> "StrategySpec":
        payload = self.to_dict()
        payload.update(changes)
        return StrategySpec.from_dict(payload)


    # -- serilestirme -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "thesis": self.thesis,
            "timeframe": self.timeframe,
            "indicators": [ind.to_dict() for ind in self.indicators],
            "entry_long": [r.to_dict() for r in self.entry_long],
            "exit_long": [r.to_dict() for r in self.exit_long],
            "filters": [r.to_dict() for r in self.filters],
            "stop_loss_atr": float(self.stop_loss_atr),
            "take_profit_atr": float(self.take_profit_atr),
            "trailing_stop_atr": float(self.trailing_stop_atr),
            "risk_per_trade_pct": float(self.risk_per_trade_pct),
            "cooldown_bars": int(self.cooldown_bars),
            "max_holding_bars": int(self.max_holding_bars),
            "params": _jsonable(self.params),
            "origin": self.origin,
            "parent_version": self.parent_version,
            "memory_refs": list(self.memory_refs),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StrategySpec":
        if not isinstance(data, Mapping):
            raise StrategyValidationError(f"Spec sozluk olmali: {type(data).__name__}")
        indicators = [
            ind if isinstance(ind, IndicatorSpec) else IndicatorSpec.from_dict(ind)
            for ind in (data.get("indicators") or [])
        ]
        entry = [
            r if isinstance(r, Rule) else Rule.from_dict(r) for r in (data.get("entry_long") or [])
        ]
        exit_ = [
            r if isinstance(r, Rule) else Rule.from_dict(r) for r in (data.get("exit_long") or [])
        ]
        filters = [
            r if isinstance(r, Rule) else Rule.from_dict(r) for r in (data.get("filters") or [])
        ]
        name = str(data.get("name") or f"strategy_{slugify(str(data.get('family', 'hybrid')))}")
        return cls(
            name=name,
            family=str(data.get("family") or "hybrid"),
            thesis=str(data.get("thesis") or ""),
            timeframe=str(data.get("timeframe") or "1h"),
            indicators=indicators,
            entry_long=entry,
            exit_long=exit_,
            filters=filters,
            stop_loss_atr=float(data.get("stop_loss_atr", 2.0)),
            take_profit_atr=float(data.get("take_profit_atr", 3.0)),
            trailing_stop_atr=float(data.get("trailing_stop_atr", 0.0)),
            risk_per_trade_pct=float(data.get("risk_per_trade_pct", 1.5)),
            cooldown_bars=int(data.get("cooldown_bars", 0)),
            max_holding_bars=int(data.get("max_holding_bars", 0)),
            params=dict(data.get("params") or {}),
            origin=str(data.get("origin") or "research"),
            parent_version=data.get("parent_version"),
            memory_refs=[str(x) for x in (data.get("memory_refs") or [])],
            notes=[str(x) for x in (data.get("notes") or [])],
        )


@dataclass
class MarketContext:
    """Arastirma ajaninin cikardigi anlik piyasa rejimi ozeti."""

    symbol: str
    timeframe: str
    data_source: str
    bars: int
    last_price: float
    price_change_pct: float = 0.0
    atr_pct: float = 0.0
    rsi: float = 50.0
    adx: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    volume_ratio: float = 1.0
    regime: str = "neutral"
    trend_strength: str = "weak"
    volatility_regime: str = "normal"
    start: Optional[str] = None
    end: Optional[str] = None
    fetched_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "data_source": self.data_source,
            "bars": int(self.bars),
            "last_price": round_or_none(self.last_price, 4),
            "price_change_pct": round_or_none(self.price_change_pct, 4),
            "atr_pct": round_or_none(self.atr_pct, 4),
            "rsi": round_or_none(self.rsi, 2),
            "adx": round_or_none(self.adx, 2),
            "ema_fast": round_or_none(self.ema_fast, 4),
            "ema_slow": round_or_none(self.ema_slow, 4),
            "volume_ratio": round_or_none(self.volume_ratio, 3),
            "regime": self.regime,
            "trend_strength": self.trend_strength,
            "volatility_regime": self.volatility_regime,
            "start": self.start,
            "end": self.end,
            "fetched_at": self.fetched_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketContext":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})  # type: ignore[arg-type]

    @property
    def regime_summary(self) -> str:
        return (
            f"{self.symbol} {self.timeframe} rejim={self.regime}/{self.trend_strength} "
            f"vol={self.volatility_regime} RSI={self.rsi:.1f} ADX={self.adx:.1f} "
            f"ATR%={self.atr_pct:.2f} hacim={self.volume_ratio:.2f}x"
        )


@dataclass
class ResearchReport:
    """Research Agent'in dongu basina urettigi tam rapor."""

    cycle: int
    market: MarketContext
    candidates: List[StrategySpec] = field(default_factory=list)
    news_digest: List[Dict[str, Any]] = field(default_factory=list)
    snapshot_count: int = 0
    winner_count: int = 0
    top_matches: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    playbook_refs: List[str] = field(default_factory=list)
    generated_at: Optional[str] = None
    data_source: str = "unknown"
    bars_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": int(self.cycle),
            "market": self.market.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "news_digest": _jsonable(self.news_digest),
            "snapshot_count": int(self.snapshot_count),
            "winner_count": int(self.winner_count),
            "top_matches": _jsonable(self.top_matches),
            "insights": list(self.insights),
            "playbook_refs": list(self.playbook_refs),
            "generated_at": self.generated_at,
            "data_source": self.data_source,
            "bars_used": int(self.bars_used),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchReport":
        return cls(
            cycle=int(data.get("cycle", 0)),
            market=MarketContext.from_dict(data.get("market") or {}),
            candidates=[StrategySpec.from_dict(c) for c in (data.get("candidates") or [])],
            news_digest=list(data.get("news_digest") or []),
            snapshot_count=int(data.get("snapshot_count", 0)),
            winner_count=int(data.get("winner_count", 0)),
            top_matches=list(data.get("top_matches") or []),
            insights=[str(x) for x in (data.get("insights") or [])],
            playbook_refs=[str(x) for x in (data.get("playbook_refs") or [])],
            generated_at=data.get("generated_at"),
            data_source=str(data.get("data_source") or "unknown"),
            bars_used=int(data.get("bars_used", 0)),
        )


@dataclass
class ErrorEntry:
    """Hata matrisinin tek bir satiri."""

    phase: str
    type: str
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    severity: str = "error"
    hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "type": self.type,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "severity": self.severity,
            "hint": self.hint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ErrorEntry":
        return cls(
            phase=str(data.get("phase", "unknown")),
            type=str(data.get("type", "Error")),
            message=str(data.get("message", "")),
            file=data.get("file"),
            line=int(data["line"]) if data.get("line") is not None else None,
            severity=str(data.get("severity", "error")),
            hint=data.get("hint"),
        )


@dataclass
class ErrorMatrix:
    """Hata matrisi: tum fazlardan toplanan yapilandirilmis hatalar."""

    entries: List[ErrorEntry] = field(default_factory=list)
    raw_stderr: str = ""

    @property
    def has_errors(self) -> bool:
        return any(e.severity in {"error", "critical"} for e in self.entries)

    @property
    def summary(self) -> str:
        if not self.entries:
            return "hata yok"
        parts = [f"{e.phase}:{e.type}: {e.message[:160]}" for e in self.entries[:5]]
        return " | ".join(parts)

    def add(
        self,
        phase: str,
        type_: str,
        message: str,
        *,
        file: Optional[str] = None,
        line: Optional[int] = None,
        severity: str = "error",
        hint: Optional[str] = None,
    ) -> ErrorEntry:
        entry = ErrorEntry(
            phase=phase,
            type=type_,
            message=str(message),
            file=file,
            line=line,
            severity=severity,
            hint=hint,
        )
        self.entries.append(entry)
        return entry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "raw_stderr": self.raw_stderr[-4000:],
            "has_errors": self.has_errors,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ErrorMatrix":
        return cls(
            entries=[ErrorEntry.from_dict(e) for e in (data.get("entries") or [])],
            raw_stderr=str(data.get("raw_stderr") or ""),
        )


@dataclass
class BacktestMetrics:
    """Simulasyon performans metrikleri."""

    net_profit_pct: float = 0.0
    net_profit_quote: float = 0.0
    final_equity: float = 0.0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    profit_factor: float = 0.0
    trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    expectancy_pct: float = 0.0
    total_fees_quote: float = 0.0
    exposure_pct: float = 0.0
    avg_holding_bars: float = 0.0
    buy_hold_pct: float = 0.0
    bars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "net_profit_pct": round(self.net_profit_pct, 4),
            "net_profit_quote": round(self.net_profit_quote, 4),
            "final_equity": round(self.final_equity, 4),
            "win_rate": round(self.win_rate, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe": round(self.sharpe, 4),
            "sortino": round(self.sortino, 4),
            "profit_factor": round(self.profit_factor, 4),
            "trades": int(self.trades),
            "winning_trades": int(self.winning_trades),
            "losing_trades": int(self.losing_trades),
            "avg_trade_pct": round(self.avg_trade_pct, 4),
            "avg_win_pct": round(self.avg_win_pct, 4),
            "avg_loss_pct": round(self.avg_loss_pct, 4),
            "expectancy_pct": round(self.expectancy_pct, 4),
            "total_fees_quote": round(self.total_fees_quote, 4),
            "exposure_pct": round(self.exposure_pct, 2),
            "avg_holding_bars": round(self.avg_holding_bars, 2),
            "buy_hold_pct": round(self.buy_hold_pct, 4),
            "bars": int(self.bars),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BacktestMetrics":
        if not data:
            return cls()
        values: Dict[str, Any] = {}
        for key, field_def in cls.__dataclass_fields__.items():
            raw = data.get(key, field_def.default)
            if key in {"trades", "winning_trades", "losing_trades", "bars"}:
                values[key] = int(raw or 0)
            else:
                values[key] = float(raw or 0.0)
        return cls(**values)

    def passes(self, gates: Mapping[str, float]) -> bool:
        checks = {
            "min_net_profit_pct": self.net_profit_pct,
            "min_win_rate": self.win_rate,
            "min_profit_factor": self.profit_factor,
            "min_sharpe": self.sharpe,
        }
        for gate, value in checks.items():
            if gate in gates and value < float(gates[gate]):
                return False
        if "max_drawdown_pct" in gates and self.max_drawdown_pct > float(gates["max_drawdown_pct"]):
            return False
        return True

    def short(self) -> str:
        return (
            f"net%={self.net_profit_pct:.2f} win%={self.win_rate:.1f} "
            f"dd%={self.max_drawdown_pct:.2f} sharpe={self.sharpe:.2f} "
            f"pf={self.profit_factor:.2f} trades={self.trades}"
        )


@dataclass
class BacktestReport:
    """Backtest Agent'in urettigi tam simulasyon raporu."""

    strategy_version: str
    name: str
    status: str = "ok"  # ok | error | failed
    split: str = "full"
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    oos_metrics: Optional[BacktestMetrics] = None
    errors: ErrorMatrix = field(default_factory=ErrorMatrix)
    runtime_seconds: float = 0.0
    equity_curve: List[List[Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    data_source: str = "unknown"
    bars: int = 0
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    created_at: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and not self.errors.has_errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_version": self.strategy_version,
            "name": self.name,
            "status": self.status,
            "split": self.split,
            "metrics": self.metrics.to_dict(),
            "oos_metrics": self.oos_metrics.to_dict() if self.oos_metrics else None,
            "errors": self.errors.to_dict(),
            "runtime_seconds": round(self.runtime_seconds, 3),
            "equity_curve": _jsonable(self.equity_curve),
            "trades": _jsonable(self.trades),
            "data_source": self.data_source,
            "bars": int(self.bars),
            "period_start": self.period_start,
            "period_end": self.period_end,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BacktestReport":
        oos = data.get("oos_metrics")
        return cls(
            strategy_version=str(data.get("strategy_version", "")),
            name=str(data.get("name", "")),
            status=str(data.get("status", "ok")),
            split=str(data.get("split", "full")),
            metrics=BacktestMetrics.from_dict(data.get("metrics") or {}),
            oos_metrics=BacktestMetrics.from_dict(oos) if oos else None,
            errors=ErrorMatrix.from_dict(data.get("errors") or {}),
            runtime_seconds=float(data.get("runtime_seconds") or 0.0),
            equity_curve=list(data.get("equity_curve") or []),
            trades=list(data.get("trades") or []),
            data_source=str(data.get("data_source") or "unknown"),
            bars=int(data.get("bars") or 0),
            period_start=data.get("period_start"),
            period_end=data.get("period_end"),
            created_at=data.get("created_at"),
        )


@dataclass
class ReviewDecision:
    """Critic & Optimizer ajaninin karari."""

    strategy_version: str
    verdict: str = "archive"
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    metric_gates: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    directives: List[str] = field(default_factory=list)
    oos_confirmed: bool = False
    elite: bool = False
    created_at: Optional[str] = None

    @property
    def promoted(self) -> bool:
        return self.verdict == "promote"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_version": self.strategy_version,
            "verdict": self.verdict,
            "score": round(self.score, 2),
            "score_breakdown": {k: round(float(v), 2) for k, v in self.score_breakdown.items()},
            "metric_gates": dict(self.metric_gates),
            "reasons": list(self.reasons),
            "directives": list(self.directives),
            "oos_confirmed": bool(self.oos_confirmed),
            "elite": bool(self.elite),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewDecision":
        return cls(
            strategy_version=str(data.get("strategy_version", "")),
            verdict=str(data.get("verdict", "archive")),
            score=float(data.get("score") or 0.0),
            score_breakdown=dict(data.get("score_breakdown") or {}),
            metric_gates=dict(data.get("metric_gates") or {}),
            reasons=[str(x) for x in (data.get("reasons") or [])],
            directives=[str(x) for x in (data.get("directives") or [])],
            oos_confirmed=bool(data.get("oos_confirmed")),
            elite=bool(data.get("elite")),
            created_at=data.get("created_at"),
        )


@dataclass
class StrategyBuild:
    """Coder Agent'in urettigi, diske yazilmis strateji surumu."""

    version: str
    name: str
    slug: str
    module_path: str
    code_hash: str
    spec: StrategySpec
    cycle: int
    parent_version: Optional[str] = None
    attempt: int = 1
    created_at: Optional[str] = None
    repair_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "slug": self.slug,
            "module_path": self.module_path,
            "code_hash": self.code_hash,
            "spec": self.spec.to_dict(),
            "cycle": int(self.cycle),
            "parent_version": self.parent_version,
            "attempt": int(self.attempt),
            "created_at": self.created_at,
            "repair_notes": list(self.repair_notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StrategyBuild":
        return cls(
            version=str(data.get("version", "")),
            name=str(data.get("name", "")),
            slug=str(data.get("slug", "")),
            module_path=str(data.get("module_path", "")),
            code_hash=str(data.get("code_hash", "")),
            spec=StrategySpec.from_dict(data.get("spec") or {}),
            cycle=int(data.get("cycle", 0)),
            parent_version=data.get("parent_version"),
            attempt=int(data.get("attempt", 1)),
            created_at=data.get("created_at"),
            repair_notes=[str(x) for x in (data.get("repair_notes") or [])],
        )


@dataclass
class CycleResult:
    """Bir tam dongunun ozeti (state dosyasina ve rapora yazilir)."""

    cycle: int
    status: str = "running"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: float = 0.0
    data_source: str = "unknown"
    market: Optional[Dict[str, Any]] = None
    research_summary: str = ""
    candidate_count: int = 0
    versions: List[str] = field(default_factory=list)
    best_version: Optional[str] = None
    best_score: float = 0.0
    promoted: List[str] = field(default_factory=list)
    archived: List[str] = field(default_factory=list)
    error: Optional[str] = None
    lessons: List[str] = field(default_factory=list)
    next_directives: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": int(self.cycle),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "data_source": self.data_source,
            "market": _jsonable(self.market),
            "research_summary": self.research_summary,
            "candidate_count": int(self.candidate_count),
            "versions": list(self.versions),
            "best_version": self.best_version,
            "best_score": round(self.best_score, 2),
            "promoted": list(self.promoted),
            "archived": list(self.archived),
            "error": self.error,
            "lessons": list(self.lessons),
            "next_directives": list(self.next_directives),
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CycleResult":
        known = cls.__dataclass_fields__.keys()
        payload = {k: v for k, v in data.items() if k in known}
        for key in ("versions", "promoted", "archived", "lessons", "next_directives"):
            payload.setdefault(key, [])
        payload.setdefault("cycle", 0)
        return cls(**payload)  # type: ignore[arg-type]


def dumps(payload: Any) -> str:
    """Dataclass/sozluk karisik yapilari JSON'a cevirir."""
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()  # type: ignore[union-attr]
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def loads(payload: str) -> Dict[str, Any]:
    return json.loads(payload)







