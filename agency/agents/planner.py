"""Deterministik strateji planlayicisi.

Iki gorevi var:

1. **Uretim**: piyasa rejimi + basarili snapshot istatistikleri + dersler +
   direktiflerden, katalogla uyumlu strateji spec'leri uretir. Ayni girdi +
   ayni dongu numarasi her zaman ayni spec'i verir (tekrarlanabilirlik).
2. **Onarim**: backtest sonucu esikleri tutmadiginda metrikleri yorumlayip
   parametre/kural duzeyinde somut degisiklik onerir (LLM'siz "optimizasyon").

Bu modul LLM'e bagimli degildir; LLM varsa urettigi spec'ler buradaki
dogrulayici ile ayni kapidan gecer.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from agency.config import Settings
from agency.schemas import CONSTANT_OPERANDS, StrategySpec
from agency.utils import clamp, slugify

FAMILIES = (
    "trend_following",
    "mean_reversion",
    "breakout",
    "momentum",
    "volatility_compression",
    "hybrid",
)

#: Rejim -> oncelikli aile sirasi
REGIME_PREFERENCE: Dict[str, Tuple[str, ...]] = {
    "trend_up": ("trend_following", "breakout", "momentum", "hybrid"),
    "trend_down": ("mean_reversion", "trend_following", "hybrid", "momentum"),
    "range": ("mean_reversion", "hybrid", "volatility_compression", "breakout"),
    "volatile": ("momentum", "breakout", "hybrid", "trend_following"),
    "squeeze": ("volatility_compression", "breakout", "mean_reversion", "hybrid"),
    "oversold_range": ("mean_reversion", "hybrid", "momentum"),
    "overbought_range": ("mean_reversion", "hybrid", "trend_following"),
    "neutral": ("trend_following", "mean_reversion", "breakout", "hybrid"),
}


@dataclass
class PlanContext:
    """Planlayiciya verilen dongu baglami."""

    cycle: int
    timeframe: str = "1h"
    regime: str = "neutral"
    atr_pct: float = 1.2
    rsi: float = 50.0
    adx: float = 20.0
    winner_features: List[Mapping[str, float]] = None  # type: ignore[assignment]
    lessons: List[str] = None  # type: ignore[assignment]
    directives: List[str] = None  # type: ignore[assignment]
    elite_specs: List[Mapping[str, Any]] = None  # type: ignore[assignment]
    avoid_fingerprints: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.winner_features = list(self.winner_features or [])
        self.lessons = list(self.lessons or [])
        self.directives = list(self.directives or [])
        self.elite_specs = list(self.elite_specs or [])
        self.avoid_fingerprints = list(self.avoid_fingerprints or [])


def _median(values: Iterable[float], default: float) -> float:
    data = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not data:
        return default
    middle = len(data) // 2
    if len(data) % 2:
        return data[middle]
    return (data[middle - 1] + data[middle]) / 2.0


def _ind(key: str, kind: str, output: str = "value", **params: Any) -> Dict[str, Any]:
    return {"key": key, "kind": kind, "output": output, "params": params}


def _rule(left: str, op: str, right: Any) -> Dict[str, Any]:
    return {"left": left, "op": op, "right": right}


# PLANNER_CLASS_PLACEHOLDER


def _template_trend_following(
    rng: random.Random, ctx: PlanContext, bias: Mapping[str, float]
) -> Dict[str, Any]:
    fast = rng.choice([9, 12, 15, 20, 21, 25])
    slow = rng.choice([45, 50, 55, 60, 80])
    if slow <= fast * 1.6:
        slow = int(fast * 2.5)
    adx_threshold = round(clamp(bias["adx_min"] + rng.uniform(-2, 4), 12, 32), 1)
    rsi_ceiling = round(bias["rsi_ceiling"], 1)
    stop = round(clamp(2.0 * bias["stop_scale"] * rng.uniform(0.85, 1.15), 1.0, 4.5), 2)
    tp = round(stop * rng.uniform(1.6, 2.6), 2)
    return {
        "name": f"ema_trend_{fast}_{slow}",
        "family": "trend_following",
        "thesis": (
            f"EMA({fast}/{slow}) kesisimi trend baslangicini yakalar; ADX>{adx_threshold} "
            "filtresi yatay piyasada kayip ureten sinyalleri eler."
        ),
        "indicators": [
            _ind("ema_fast", "ema", period=fast),
            _ind("ema_slow", "ema", period=slow),
            _ind("rsi", "rsi", period=14),
            _ind("adx_14", "adx", period=14),
        ],
        "entry_long": [
            _rule("ema_fast", "cross_above", "ema_slow"),
            _rule("rsi", "<", rsi_ceiling),
        ],
        "exit_long": [_rule("ema_fast", "cross_below", "ema_slow")],
        "filters": [_rule("adx_14", ">", adx_threshold), _rule("close", ">", "ema_slow")],
        "stop_loss_atr": stop,
        "take_profit_atr": tp,
        "trailing_stop_atr": rng.choice([0.0, round(stop * 1.4, 2)]),
        "risk_per_trade_pct": round(rng.uniform(0.75, 1.75), 2),
        "cooldown_bars": rng.choice([0, 2, 4]),
        "max_holding_bars": rng.choice([0, 0, 96]),
    }


def _template_mean_reversion(
    rng: random.Random, ctx: PlanContext, bias: Mapping[str, float]
) -> Dict[str, Any]:
    bb_period = rng.choice([20, 24, 30])
    bb_std = rng.choice([2.0, 2.2, 2.5])
    pctb_entry = round(rng.uniform(0.05, 0.30), 2)
    rsi_floor = round(clamp(bias["rsi_floor"] + rng.uniform(-2, 8), 26, 48), 1)
    adx_ceiling = round(rng.uniform(24, 34), 1)
    stop = round(clamp(1.6 * bias["stop_scale"], 0.8, 3.0), 2)
    tp = round(stop * rng.uniform(1.0, 1.8), 2)
    filters = [_rule("adx_14", "<", adx_ceiling)]
    if rng.random() < 0.5:
        filters.append(_rule("rel_volume", ">", round(rng.uniform(0.7, 1.1), 2)))
    return {
        "name": f"bb_reversion_{bb_period}_{int(bb_std * 10)}",
        "family": "mean_reversion",
        "thesis": (
            f"Fiyat Bollinger alt bandina yaklasirken (pctb<{pctb_entry}) ve RSI<{rsi_floor} iken "
            f"ortalamaya donus beklenir; ADX<{adx_ceiling} filtresi guclu trendlerde islem almaz."
        ),
        "indicators": [
            _ind("bb_pctb", "bbands", output="pctb", period=bb_period, std=bb_std),
            _ind("rsi", "rsi", period=14),
            _ind("adx_14", "adx", period=14),
            _ind("rel_volume", "rel_volume", period=20),
        ],
        "entry_long": [_rule("bb_pctb", "<", pctb_entry), _rule("rsi", "<", rsi_floor)],
        "exit_long": [
            _rule("bb_pctb", ">", round(rng.uniform(0.45, 0.65), 2)),
            _rule("rsi", ">", round(rng.uniform(55, 66), 1)),
        ],
        "filters": filters,
        "stop_loss_atr": stop,
        "take_profit_atr": tp,
        "trailing_stop_atr": 0.0,
        "risk_per_trade_pct": round(rng.uniform(0.75, 1.5), 2),
        "cooldown_bars": rng.choice([2, 4, 6]),
        "max_holding_bars": rng.choice([24, 36, 48]),
    }


def _template_breakout(
    rng: random.Random, ctx: PlanContext, bias: Mapping[str, float]
) -> Dict[str, Any]:
    channel = rng.choice([20, 24, 30, 40])
    volume_min = round(clamp(bias["volume_min"] + rng.uniform(-0.1, 0.3), 0.9, 2.0), 2)
    adx_threshold = round(clamp(bias["adx_min"], 14, 30), 1)
    stop = round(clamp(1.9 * bias["stop_scale"], 1.0, 4.0), 2)
    tp = round(stop * rng.uniform(2.0, 3.2), 2)
    return {
        "name": f"donchian_breakout_{channel}",
        "family": "breakout",
        "thesis": (
            f"{channel} barlik kanal kirilimi + hacim onayi (>{volume_min}x) ile momentum "
            "patlamalari yakalanir; trailing stop trend boyunca tasir."
        ),
        "indicators": [
            _ind("dc_upper", "donchian", output="upper", period=channel),
            _ind("dc_lower", "donchian", output="lower", period=channel),
            _ind("rel_volume", "rel_volume", period=20),
            _ind("adx_14", "adx", period=14),
        ],
        "entry_long": [_rule("close", "cross_above", "dc_upper")],
        "exit_long": [_rule("close", "cross_below", "dc_lower")],
        "filters": [_rule("rel_volume", ">", volume_min), _rule("adx_14", ">", adx_threshold)],
        "stop_loss_atr": stop,
        "take_profit_atr": tp,
        "trailing_stop_atr": round(clamp(stop * 1.6, 1.0, 6.0), 2),
        "risk_per_trade_pct": round(rng.uniform(0.75, 1.75), 2),
        "cooldown_bars": rng.choice([0, 2, 4]),
        "max_holding_bars": 0,
    }


def _template_momentum(
    rng: random.Random, ctx: PlanContext, bias: Mapping[str, float]
) -> Dict[str, Any]:
    fast = rng.choice([8, 10, 12, 14])
    slow = rng.choice([20, 24, 26, 30])
    signal = rng.choice([7, 9, 12])
    trend_period = rng.choice([50, 80, 100])
    roc_threshold = round(rng.uniform(0.3, 1.5), 2)
    stop = round(clamp(2.1 * bias["stop_scale"], 1.0, 4.5), 2)
    tp = round(stop * rng.uniform(1.8, 3.0), 2)
    return {
        "name": f"macd_momentum_{fast}_{slow}",
        "family": "momentum",
        "thesis": (
            f"MACD({fast},{slow},{signal}) histograminin sifiri yukari kesmesi + ROC>{roc_threshold}% "
            f"momentum onayi; EMA{trend_period} trend filtresi ters islemleri engeller."
        ),
        "indicators": [
            _ind("macd_hist", "macd", output="hist", fast=fast, slow=slow, signal=signal),
            _ind("roc_10", "roc", period=10),
            _ind("ema_trend", "ema", period=trend_period),
            _ind("rel_volume", "rel_volume", period=20),
        ],
        "entry_long": [_rule("macd_hist", "cross_above", 0), _rule("roc_10", ">", roc_threshold)],
        "exit_long": [_rule("macd_hist", "cross_below", 0)],
        "filters": [_rule("close", ">", "ema_trend"), _rule("rel_volume", ">", round(bias["volume_min"], 2))],
        "stop_loss_atr": stop,
        "take_profit_atr": tp,
        "trailing_stop_atr": 0.0,
        "risk_per_trade_pct": round(rng.uniform(0.75, 1.75), 2),
        "cooldown_bars": rng.choice([0, 2, 4]),
        "max_holding_bars": rng.choice([0, 72]),
    }


def _template_volatility_compression(
    rng: random.Random, ctx: PlanContext, bias: Mapping[str, float]
) -> Dict[str, Any]:
    bb_period = rng.choice([20, 24, 30])
    channel = rng.choice([20, 30, 40])
    width_max = round(rng.uniform(2.5, 5.0), 2)
    stop = round(clamp(1.7 * bias["stop_scale"], 0.9, 3.5), 2)
    tp = round(stop * rng.uniform(2.0, 3.5), 2)
    return {
        "name": f"squeeze_breakout_{channel}",
        "family": "volatility_compression",
        "thesis": (
            f"Bollinger bant genisligi %{width_max} altina sikistiginda enerji birikir; "
            f"{channel} barlik kanal kirilimi patlamayi yakalar."
        ),
        "indicators": [
            _ind("bb_width", "bbands", output="width", period=bb_period, std=2.0),
            _ind("dc_upper", "donchian", output="upper", period=channel),
            _ind("dc_mid", "donchian", output="mid", period=channel),
            _ind("atr_pct_14", "atr_pct", period=14),
        ],
        "entry_long": [_rule("close", "cross_above", "dc_upper")],
        "exit_long": [_rule("close", "cross_below", "dc_mid")],
        "filters": [
            _rule("bb_width_prev", "<", width_max),
            _rule("atr_pct_14", ">", 0.10),
        ],
        "stop_loss_atr": stop,
        "take_profit_atr": tp,
        "trailing_stop_atr": round(clamp(stop * 1.6, 1.0, 6.0), 2),
        "risk_per_trade_pct": round(rng.uniform(0.6, 1.25), 2),
        "cooldown_bars": rng.choice([2, 4]),
        "max_holding_bars": rng.choice([0, 96]),
    }


def _template_hybrid(
    rng: random.Random, ctx: PlanContext, bias: Mapping[str, float]
) -> Dict[str, Any]:
    fast = rng.choice([20, 25, 30])
    slow = rng.choice([50, 60, 80])
    rsi_pullback = round(clamp(bias["rsi_floor"] + rng.uniform(6, 14), 35, 55), 1)
    stop = round(clamp(2.0 * bias["stop_scale"], 1.0, 4.0), 2)
    tp = round(stop * rng.uniform(1.7, 2.8), 2)
    return {
        "name": f"trend_pullback_{fast}_{slow}",
        "family": "hybrid",
        "thesis": (
            f"EMA({fast}) > EMA({slow}) trendi icinde RSI {rsi_pullback} uzerine donerken alim; "
            "trend + momentum + hacim uclusu birlikte aranir."
        ),
        "indicators": [
            _ind("ema_fast", "ema", period=fast),
            _ind("ema_slow", "ema", period=slow),
            _ind("rsi", "rsi", period=14),
            _ind("rel_volume", "rel_volume", period=20),
        ],
        "entry_long": [
            _rule("rsi", "cross_above", rsi_pullback),
            _rule("ema_fast", ">", "ema_slow"),
        ],
        "exit_long": [_rule("rsi", ">", 70), _rule("close", "cross_below", "ema_fast")],
        "filters": [_rule("rel_volume", ">", 0.9)],
        "stop_loss_atr": stop,
        "take_profit_atr": tp,
        "trailing_stop_atr": 0.0,
        "risk_per_trade_pct": round(rng.uniform(0.75, 1.5), 2),
        "cooldown_bars": rng.choice([2, 4, 6]),
        "max_holding_bars": rng.choice([48, 72, 0]),
    }


TEMPLATES = {
    "trend_following": _template_trend_following,
    "mean_reversion": _template_mean_reversion,
    "breakout": _template_breakout,
    "momentum": _template_momentum,
    "volatility_compression": _template_volatility_compression,
    "hybrid": _template_hybrid,
}




class StrategyPlanner:
    """Rejim farkindalikli, deterministik spec uretici ve onarici."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.targets = settings.targets
        self.rng_seed_base = 7919

    # ------------------------------------------------------------- helpers
    def _rng(self, ctx: PlanContext, family: str, index: int) -> random.Random:
        seed = self.rng_seed_base + ctx.cycle * 101 + FAMILIES.index(family) * 17 + index * 7
        return random.Random(seed)

    def _bias(self, ctx: PlanContext) -> Dict[str, float]:
        """Basarili snapshot'lardan esik onerileri cikarir."""
        adx_values = [f.get("adx") for f in ctx.winner_features if f.get("adx") is not None]
        rsi_values = [f.get("rsi") for f in ctx.winner_features if f.get("rsi") is not None]
        atr_values = [f.get("atr_pct") for f in ctx.winner_features if f.get("atr_pct") is not None]
        vol_values = [
            f.get("rel_volume") for f in ctx.winner_features if f.get("rel_volume") is not None
        ]
        median_atr = _median(atr_values, ctx.atr_pct)
        stop_scale = clamp(median_atr / ctx.atr_pct, 0.8, 2.0) if ctx.atr_pct > 0 else 1.0
        return {
            "adx_min": clamp(_median(adx_values, 20.0) * 0.9, 12.0, 32.0),
            "rsi_ceiling": clamp(_median(rsi_values, 60.0) + 8.0, 55.0, 80.0),
            "rsi_floor": clamp(_median(rsi_values, 40.0) - 8.0, 20.0, 45.0),
            "stop_scale": stop_scale,
            "volume_min": clamp(_median(vol_values, 1.1), 0.8, 2.0),
        }

    def _finalize(
        self, payload: Dict[str, Any], ctx: PlanContext, origin: str = "research"
    ) -> StrategySpec:
        payload = dict(payload)
        payload.setdefault("timeframe", ctx.timeframe)
        payload.setdefault("origin", origin)
        payload.setdefault("notes", [])
        payload["notes"] = list(payload["notes"]) + [
            f"rejim={ctx.regime}",
            f"winner_snapshot={len(ctx.winner_features)}",
        ]
        spec = StrategySpec.from_dict(payload)
        return self.sanitize(spec, ctx)

    # ---------------------------------------------------------------- plan
    def plan(self, ctx: PlanContext, count: Optional[int] = None) -> List[StrategySpec]:
        """Baglamdan en az 1, en fazla ``count`` gecerli spec uretir."""
        count = int(count or self.settings.targets.candidates_per_cycle)
        preference = REGIME_PREFERENCE.get(ctx.regime, FAMILIES)
        bias = self._bias(ctx)
        specs: List[StrategySpec] = []
        seen: set[str] = set(ctx.avoid_fingerprints)

        elite = self.elite_variant(ctx)
        if elite is not None:
            specs.append(elite)
            seen.add(elite.fingerprint())

        for index, family in enumerate(preference):
            if len(specs) >= count:
                break
            rng = self._rng(ctx, family, index)
            payload = TEMPLATES[family](rng, ctx, bias)
            spec = self._finalize(payload, ctx)
            spec = self.apply_directives(spec, ctx.directives)
            fingerprint = spec.fingerprint()
            if fingerprint in seen or spec.validate():
                continue
            seen.add(fingerprint)
            specs.append(spec)

        # Rejim tercihi yetmediyse kalan ailelerle tamamla
        for index, family in enumerate(FAMILIES):
            if len(specs) >= count:
                break
            rng = self._rng(ctx, family, index + 10)
            spec = self._finalize(TEMPLATES[family](rng, ctx, bias), ctx)
            fingerprint = spec.fingerprint()
            if fingerprint in seen or spec.validate():
                continue
            seen.add(fingerprint)
            specs.append(spec)
        return specs[:count]

    def elite_variant(self, ctx: PlanContext) -> Optional[StrategySpec]:
        """Kalici hafizadaki en iyi stratejiden kucuk bir varyasyon uretir."""
        if not ctx.elite_specs:
            return None
        base_payload = dict(ctx.elite_specs[0])
        try:
            base = StrategySpec.from_dict(base_payload)
        except Exception:
            return None
        rng = self._rng(ctx, base.family if base.family in FAMILIES else "hybrid", 99)
        payload = base.to_dict()
        payload["name"] = f"{slugify(base.name)}_mut{ctx.cycle}"
        payload["origin"] = "memory"
        payload["parent_version"] = base_payload.get("_version")
        payload["stop_loss_atr"] = round(
            clamp(base.stop_loss_atr * rng.uniform(0.9, 1.15), 0.6, 6.0), 2
        )
        payload["take_profit_atr"] = round(
            clamp(base.take_profit_atr * rng.uniform(0.9, 1.2), 0.0, 12.0), 2
        )
        payload["risk_per_trade_pct"] = round(
            clamp(base.risk_per_trade_pct * rng.uniform(0.85, 1.1), 0.25, 3.0), 2
        )
        # Giriş kuralı esiklerinde kucuk oynama (kesif icin)
        entry = []
        for rule in payload.get("entry_long", []):
            rule = dict(rule)
            if isinstance(rule.get("right"), float) and rule.get("op") in {">", ">="}:
                rule["right"] = round(rule["right"] * rng.uniform(1.0, 1.12), 3)
            elif isinstance(rule.get("right"), float) and rule.get("op") in {"<", "<="}:
                rule["right"] = round(rule["right"] * rng.uniform(0.88, 1.0), 3)
            entry.append(rule)
        payload["entry_long"] = entry or payload.get("entry_long")
        payload.setdefault("notes", []).append(f"elite_varyasyon cycle={ctx.cycle}")
        try:
            spec = StrategySpec.from_dict(payload)
        except Exception:
            return None
        if spec.validate():
            return None
        return spec

    # -------------------------------------------------------- directives
    def apply_directives(self, spec: StrategySpec, directives: Sequence[str]) -> StrategySpec:
        """Metin direktiflerini somut parametre degisikliklerine cevirir."""
        if not directives:
            return spec
        payload = spec.to_dict()
        notes = list(payload.get("notes") or [])
        numeric_fields = (
            "stop_loss_atr",
            "take_profit_atr",
            "trailing_stop_atr",
            "risk_per_trade_pct",
            "cooldown_bars",
            "max_holding_bars",
        )
        for raw in directives:
            text = str(raw).lower()
            for field in numeric_fields:
                match = re.search(rf"{field}\s*([0-9.]+)\s*(?:->|=>|>|to)\s*([0-9.]+)", text)
                if match:
                    old, new = float(match.group(1)), float(match.group(2))
                    if abs(new - old) > 1e-9:
                        payload[field] = new
                        notes.append(f"{field} {old}->{new} (direktif)")

            indicator_match = re.search(
                r"(adx|rsi|atr|volume|bb|macd|donchian)[a-z_0-9]*\s*([0-9.]+)\s*(?:->|=>|>)\s*([0-9.]+)",
                text,
            )
            if indicator_match:
                token, _, new_value = indicator_match.groups()
                changed = False
                for group in ("entry_long", "exit_long", "filters"):
                    for rule in payload.get(group) or []:
                        if token in str(rule.get("left", "")) and isinstance(
                            rule.get("right"), (int, float)
                        ):
                            rule["right"] = float(new_value)
                            changed = True
                if changed:
                    notes.append(f"{token} esigi -> {new_value} (direktif)")

            if payload["stop_loss_atr"] > 0 and any(
                word in text for word in ("daralt", "sikilastir", "tighten", "kucult")
            ):
                payload["stop_loss_atr"] = round(payload["stop_loss_atr"] * 0.8, 2)
                notes.append("stop mesafesi %20 daraltildi (direktif)")
            if any(word in text for word in ("filtre ekle", "trend filtresi", "add filter")):
                keys = {ind["key"] for ind in payload.get("indicators", [])}
                if "ema_slow" in keys:
                    filters = payload.get("filters") or []
                    exists = any(
                        r.get("left") == "close" and r.get("right") == "ema_slow" for r in filters
                    )
                    if not exists:
                        filters.append({"left": "close", "op": ">", "right": "ema_slow"})
                        payload["filters"] = filters
                        notes.append("close > ema_slow filtresi eklendi (direktif)")
            if any(word in text for word in ("az islem", "islem sayisini azalt", "overtrade", "chop")):
                payload["cooldown_bars"] = int(payload.get("cooldown_bars", 0)) + 3
                notes.append("cooldown +3 (direktif)")

        payload["notes"] = notes[-12:]
        self._clamp_payload(payload)
        return StrategySpec.from_dict(payload)


    # ------------------------------------------------------------- repair
    def repair(
        self,
        spec: StrategySpec,
        *,
        metrics: Mapping[str, Any],
        gates: Mapping[str, bool],
        directives: Sequence[str] = (),
        errors: Sequence[str] = (),
        attempt: int = 1,
        ctx: Optional[PlanContext] = None,
    ) -> StrategySpec:
        """Basarisiz stratejiyi metrik gerekceli olarak onarir."""
        payload = spec.to_dict()
        notes = list(payload.get("notes") or [])
        payload["name"] = f"{slugify(spec.name)}_r{attempt}"
        payload["origin"] = "coder"
        trades = int(metrics.get("trades", 0) or 0)
        bars = int(metrics.get("bars", 0) or 0)
        profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
        stop_before = float(payload["stop_loss_atr"])
        risk_before = float(payload["risk_per_trade_pct"])
        tp_before = float(payload["take_profit_atr"])

        if gates.get("drawdown") is False:
            payload["stop_loss_atr"] = round(stop_before * 0.8, 2)
            payload["risk_per_trade_pct"] = round(risk_before * 0.6, 2)
            if float(payload.get("trailing_stop_atr", 0.0)) <= 0:
                payload["trailing_stop_atr"] = round(
                    clamp(float(payload["stop_loss_atr"]) * 1.5, 0.5, 6.0), 2
                )
            notes.append(
                f"drawdown asimi: stop {stop_before}->{payload['stop_loss_atr']}, "
                f"risk {risk_before}->{payload['risk_per_trade_pct']}"
            )

        if gates.get("win_rate") is False:
            for rule in payload.get("entry_long", []):
                if "adx" in str(rule.get("left", "")) and isinstance(rule.get("right"), (int, float)):
                    rule["right"] = float(rule["right"]) + 2.0
                    notes.append(f"ADX giris esigi +2 -> {rule['right']}")
                    break
            for rule in payload.get("filters", []):
                if "adx" in str(rule.get("left", "")) and isinstance(rule.get("right"), (int, float)):
                    rule["right"] = float(rule["right"]) + 2.0
                    notes.append(f"ADX filtresi +2 -> {rule['right']}")
                    break
            if not payload.get("exit_long"):
                payload["exit_long"] = [{"left": "rsi", "op": ">", "right": 68}]
                notes.append("exit kurali eklendi: rsi > 68")

        if gates.get("trades") is False:
            changed = False
            if trades == 0:
                # Hic islem yok: kosullar birbiriyle celisiyor -> yapisal onarim
                if payload.get("filters"):
                    notes.append(
                        f"0 islem: {len(payload['filters'])} filtre kaldirildi (celisen kosullar)"
                    )
                    payload["filters"] = []
                for rule in payload.get("entry_long", []):
                    if rule.get("op") == "cross_above":
                        rule["op"] = ">"
                    elif rule.get("op") == "cross_below":
                        rule["op"] = "<"
                payload["cooldown_bars"] = 0
                notes.append("0 islem: girisler karsilastirmaya cevrildi, cooldown sifirlandi")
                changed = True
            for rule in payload.get("entry_long", []):
                if rule.get("op") in {"cross_above", "cross_below"}:
                    rule["op"] = ">" if rule["op"] == "cross_above" else "<"
                    changed = True
                    notes.append("giris kesisimi karsilastirmaya cevrildi (daha fazla islem)")
                    break
            for rule in payload.get("filters", []):
                if isinstance(rule.get("right"), (int, float)) and "close" not in str(rule.get("left", "")):
                    value = float(rule["right"])
                    rule["right"] = round(
                        value * 0.8 if rule.get("op") in {">", ">="} else value * 1.2, 2
                    )
                    changed = True
                    notes.append(f"filtre gevsetildi: {rule['left']} {rule['op']} {rule['right']}")
                    break
            payload["cooldown_bars"] = max(0, int(payload.get("cooldown_bars", 0)) - 1)

            filters = payload.get("filters") or []
            if len(filters) > 1:
                dropped = filters.pop()
                payload["filters"] = filters
                notes.append(
                    f"filtre kaldirildi: {dropped.get('left')} (islem sikligi icin)"
                )

            longest_key, longest_value = None, 0
            for indicator in payload.get("indicators", []):
                period = (indicator.get("params") or {}).get("period")
                if isinstance(period, (int, float)) and int(period) > longest_value:
                    longest_key, longest_value = indicator.get("key"), int(period)
            if longest_key and longest_value > 20:
                new_period = max(5, int(longest_value * 0.7))
                for indicator in payload["indicators"]:
                    if indicator.get("key") == longest_key:
                        indicator.setdefault("params", {})["period"] = new_period
                notes.append(
                    f"{longest_key} periyodu {longest_value}->{new_period} (islem sikligi icin)"
                )
            if not changed:
                notes.append("islem sayisi dusuk: yapisal gevsetme uygulandi")

        if gates.get("net_profit") is False:
            if profit_factor < 1.0:
                for rule in payload.get("filters", []):
                    if "adx" in str(rule.get("left", "")) and isinstance(rule.get("right"), (int, float)):
                        rule["right"] = float(rule["right"]) + 1.0
                        notes.append(f"kayip egilimi: ADX filtresi +1 -> {rule['right']}")
                        break
                payload["take_profit_atr"] = round(max(tp_before * 0.85, 0.6), 2)
                notes.append(f"take_profit {tp_before}->{payload['take_profit_atr']} (hizli kar al)")
            else:
                payload["take_profit_atr"] = round(min(tp_before * 1.2, 12.0), 2)
                notes.append(f"take_profit {tp_before}->{payload['take_profit_atr']} (kazanani tasi)")

        if gates.get("sharpe") is False:
            payload["cooldown_bars"] = int(payload.get("cooldown_bars", 0)) + 2
            payload["risk_per_trade_pct"] = round(float(payload["risk_per_trade_pct"]) * 0.85, 2)
            notes.append("sharpe dusuk: cooldown +2, risk %15 azaltildi")

        if bars and trades > max(10, bars // 6):
            payload["cooldown_bars"] = int(payload.get("cooldown_bars", 0)) + 4
            notes.append("asiri islem: cooldown +4")

        if errors:
            notes.append(f"hata matrisi: {str(errors[0])[:120]}")

        payload["notes"] = notes[-14:]
        self._clamp_payload(payload)
        repaired = StrategySpec.from_dict(payload)
        repaired = self.apply_directives(repaired, directives)
        repaired = self.sanitize(repaired, ctx)

        if repaired.validate():
            fallback_ctx = ctx or PlanContext(cycle=attempt + 1)
            family = spec.family if spec.family in TEMPLATES else "hybrid"
            rng = self._rng(fallback_ctx, family, attempt + 50)
            repaired = self._finalize(
                TEMPLATES[family](rng, fallback_ctx, self._bias(fallback_ctx)), fallback_ctx
            )
            repaired.notes.append("onarim gecersiz kaldi; sablondan yeniden uretildi")
        return repaired


    # ----------------------------------------------------------- sanitize
    @staticmethod
    def _clamp_payload(payload: Dict[str, Any]) -> None:
        payload["stop_loss_atr"] = round(clamp(float(payload.get("stop_loss_atr", 2.0)), 0.5, 6.0), 2)
        payload["take_profit_atr"] = round(
            clamp(float(payload.get("take_profit_atr", 3.0)), 0.0, 12.0), 2
        )
        payload["trailing_stop_atr"] = round(
            clamp(float(payload.get("trailing_stop_atr", 0.0)), 0.0, 6.0), 2
        )
        payload["risk_per_trade_pct"] = round(
            clamp(float(payload.get("risk_per_trade_pct", 1.5)), 0.25, 3.0), 2
        )
        payload["cooldown_bars"] = int(clamp(float(payload.get("cooldown_bars", 0)), 0, 50))
        payload["max_holding_bars"] = int(clamp(float(payload.get("max_holding_bars", 0)), 0, 1000))

    def sanitize(self, spec: StrategySpec, ctx: Optional[PlanContext] = None) -> StrategySpec:
        """Gecersiz indikator/kurallari ayiklar; en az bir giris kurali garanti eder."""
        payload = spec.to_dict()
        valid_indicators = [
            indicator.to_dict()
            for indicator in spec.indicators
            if not indicator.validate()
        ]
        payload["indicators"] = valid_indicators
        keys = {ind["key"] for ind in valid_indicators}
        base_operands = keys | {"open", "high", "low", "close", "volume"} | set(CONSTANT_OPERANDS)
        allowed = base_operands | {f"{name}_prev" for name in base_operands}

        def keep(rule: Dict[str, Any]) -> bool:
            left = str(rule.get("left", ""))
            right = rule.get("right")
            if left not in allowed:
                return False
            if isinstance(right, str) and right not in allowed:
                try:
                    float(right)
                except ValueError:
                    return False
            return True

        for group in ("entry_long", "exit_long", "filters"):
            payload[group] = [r for r in (payload.get(group) or []) if keep(r)]

        if not valid_indicators:
            fallback_ctx = ctx or PlanContext(cycle=1)
            rng = self._rng(fallback_ctx, "trend_following", 3)
            base = TEMPLATES["trend_following"](rng, fallback_ctx, self._bias(fallback_ctx))
            return StrategySpec.from_dict(base)
        if not payload["entry_long"]:
            ema_keys = [k for k in keys if "ema" in k]
            if len(ema_keys) >= 2:
                payload["entry_long"] = [{"left": ema_keys[0], "op": ">", "right": ema_keys[1]}]
            else:
                payload["entry_long"] = [{"left": "close", "op": ">", "right": sorted(keys)[0]}]
        self._clamp_payload(payload)
        return StrategySpec.from_dict(payload)

    # ------------------------------------------------------------ LLM kopru
    def from_llm_ideas(
        self, payload: Optional[Mapping[str, Any]], ctx: PlanContext
    ) -> List[StrategySpec]:
        """LLM cevabindaki ``strategy_ideas`` listesini dogrulanmis spec'e cevirir."""
        if not payload:
            return []
        ideas = payload.get("strategy_ideas") or payload.get("strategies") or []
        if isinstance(ideas, Mapping):
            ideas = [ideas]
        specs: List[StrategySpec] = []
        for item in ideas:
            try:
                spec = StrategySpec.from_dict(item)
            except Exception:
                continue
            spec = self.sanitize(spec, ctx)
            if spec.validate():
                continue
            specs.append(spec)
        return specs

    def repair_with_llm_payload(
        self, payload: Optional[Mapping[str, Any]], ctx: PlanContext
    ) -> Optional[StrategySpec]:
        """LLM'in urettigi tek spec'i (cikis veya onarim) dogrular."""
        if not payload:
            return None
        candidate = payload.get("strategy") or payload.get("spec") or payload
        if not isinstance(candidate, Mapping):
            return None
        try:
            spec = StrategySpec.from_dict(candidate)
        except Exception:
            return None
        spec = self.sanitize(spec, ctx)
        if spec.validate():
            return None
        return spec

    # --------------------------------------------------------- exploration
    def perturb(self, spec: StrategySpec, attempt: int = 1) -> StrategySpec:
        """Ayni kalan onarimlarda kesif icin garantili degisiklik uygular.

        Onarim dongusu sinirlara dayanip ayni spec'i uretirse ogrenme durur.
        Bu fonksiyon her cagrida fingerprint'i degistiren bir "mutasyon"
        uygular: stop/TP orani, giris esigi, trailing veya cooldown.
        """
        payload = spec.to_dict()
        payload["name"] = f"{slugify(spec.name)}_p{attempt}"
        payload["origin"] = "coder"
        rng = random.Random(self.rng_seed_base + attempt * 31 + len(spec.name) * 7)
        knob = attempt % 4
        notes = list(payload.get("notes") or [])

        if knob == 0:
            before = float(payload["stop_loss_atr"])
            payload["stop_loss_atr"] = round(clamp(before * rng.choice([0.85, 1.15]), 0.5, 6.0), 2)
            notes.append(f"kesif: stop {before}->{payload['stop_loss_atr']}")
        elif knob == 1:
            before = float(payload["take_profit_atr"])
            payload["take_profit_atr"] = round(
                clamp(before * rng.choice([0.8, 1.25]), 0.4, 12.0), 2
            )
            notes.append(f"kesif: TP {before}->{payload['take_profit_atr']}")
        elif knob == 2:
            adjusted = False
            for group in ("entry_long", "filters"):
                for rule in payload.get(group) or []:
                    if isinstance(rule.get("right"), (int, float)) and rule.get("left") not in {
                        "close",
                        "open",
                        "high",
                        "low",
                        "volume",
                    }:
                        before = float(rule["right"])
                        factor = rng.choice([0.9, 1.1])
                        rule["right"] = round(before * factor, 3)
                        notes.append(f"kesif: {rule['left']} esigi {before}->{rule['right']}")
                        adjusted = True
                        break
                if adjusted:
                    break
            if not adjusted:
                payload["risk_per_trade_pct"] = round(
                    clamp(float(payload["risk_per_trade_pct"]) * 0.8, 0.25, 3.0), 2
                )
                notes.append("kesif: risk %20 azaltildi")
        else:
            before = float(payload.get("trailing_stop_atr", 0.0))
            payload["trailing_stop_atr"] = (
                0.0 if before > 0 else round(clamp(float(payload["stop_loss_atr"]) * 1.6, 0.5, 6.0), 2)
            )
            payload["cooldown_bars"] = int(payload.get("cooldown_bars", 0)) + 2
            notes.append(f"kesif: trailing {before}->{payload['trailing_stop_atr']}, cooldown +2")

        payload["notes"] = notes[-14:]
        self._clamp_payload(payload)
        # Davranissal degisiklik garantisi: stop/TP her zaman hafifce oynatilir.
        # (Sadece esik degistirmek, esik hic baglayici degilse ayni metrikleri
        #  uretebiliyordu -> ogrenme duruyordu.)
        scale = 1.08 if attempt % 2 else 0.93
        payload["stop_loss_atr"] = round(
            clamp(float(payload["stop_loss_atr"]) * scale, 0.5, 6.0), 2
        )
        payload["take_profit_atr"] = round(
            clamp(float(payload["take_profit_atr"]) * (2.0 - scale), 0.4, 12.0), 2
        )
        notes.append(f"kesif: stop x{scale}, TP x{round(2.0 - scale, 2)}")
        payload["notes"] = notes[-14:]
        self._clamp_payload(payload)
        return StrategySpec.from_dict(payload)





