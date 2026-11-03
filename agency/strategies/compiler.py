"""Strateji derleyicisi: bildirimsel :class:`StrategySpec` -> Python kodu.

Neden derleyici?
    LLM'lerin serbest kod uretimi; sozdizimi hatalari, guvenlik riskleri ve
    tekrarlanabilirlik sorunlari yaratir. Bunun yerine LLM **bildirimsel bir
    spec** uretir; bu modul o spec'i katalogdaki indikatorlerle sinirli,
    deterministik ve her zaman calisan Python koduna cevirir.

Uretilen kod:
    * ``BaseStrategy`` alt sinifi (OOP)
    * ``prepare`` icinde vektorize indikator hesabi
    * ``generate_signal`` icinde kural tabanli karar
    * ``risk_defaults`` icinde SL/TP/boyut ayarlari
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Set

from agency.schemas import Rule, StrategySpec
from agency.utils import slugify

#: Indikator turu -> (kullanilan fonksiyonlar, tek satirlik kod sablonu)
#: Sablonda ``{key}``, ``{p}`` (sozluk) ve ``{out}`` yer tutuculari kullanilir.
INDICATOR_TEMPLATES: Dict[str, str] = {
    "ema": 'out["{key}"] = ema(out["close"], {p}["{key}_period"])',
    "sma": 'out["{key}"] = sma(out["close"], {p}["{key}_period"])',
    "rsi": 'out["{key}"] = rsi(out["close"], {p}["{key}_period"])',
    "atr": 'out["{key}"] = atr(out["high"], out["low"], out["close"], {p}["{key}_period"])',
    "atr_pct": 'out["{key}"] = atr(out["high"], out["low"], out["close"], {p}["{key}_period"]) / out["close"] * 100.0',
    "adx": 'out["{key}"] = adx(out["high"], out["low"], out["close"], {p}["{key}_period"])["{output}"]',
    "macd": (
        'out["{key}"] = macd(out["close"], {p}["{key}_fast"], {p}["{key}_slow"], '
        '{p}["{key}_signal"])["{output}"]'
    ),
    "bbands": 'out["{key}"] = bollinger(out["close"], {p}["{key}_period"], {p}["{key}_std"])["{output}"]',
    "roc": 'out["{key}"] = roc(out["close"], {p}["{key}_period"])',
    "stoch": (
        'out["{key}"] = stochastic(out["high"], out["low"], out["close"], '
        '{p}["{key}_period"], {p}["{key}_smooth"])["{output}"]'
    ),
    "vwap": 'out["{key}"] = rolling_vwap(out["high"], out["low"], out["close"], out["volume"], {p}["{key}_period"])',
    "vol_sma": 'out["{key}"] = volume_sma(out["volume"], {p}["{key}_period"])',
    "rel_volume": 'out["{key}"] = out["volume"] / volume_sma(out["volume"], {p}["{key}_period"])',
    "donchian": 'out["{key}"] = donchian(out["high"], out["low"], {p}["{key}_period"])["{output}"]',
}

#: Kullanilan fonksiyonlari import satirina cevirmek icin
FUNCTION_IMPORTS: Dict[str, str] = {
    "ema": "ema",
    "sma": "sma",
    "rsi": "rsi",
    "atr": "atr",
    "adx": "adx",
    "macd": "macd",
    "bollinger": "bollinger",
    "roc": "roc",
    "stochastic": "stochastic",
    "rolling_vwap": "rolling_vwap",
    "volume_sma": "volume_sma",
    "donchian": "donchian",
}

FUNCTION_FOR_KIND: Dict[str, str] = {
    "ema": "ema",
    "sma": "sma",
    "rsi": "rsi",
    "atr": "atr",
    "atr_pct": "atr",
    "adx": "adx",
    "macd": "macd",
    "bbands": "bollinger",
    "roc": "roc",
    "stoch": "stochastic",
    "vwap": "rolling_vwap",
    "vol_sma": "volume_sma",
    "rel_volume": "volume_sma",
    "donchian": "donchian",
}

RAW_COLUMNS = {"open", "high", "low", "close", "volume"}
CONSTANTS = {"zero": 0.0, "one": 1.0}


class CompileError(RuntimeError):
    """Spec kod uretimine uygun degil."""


def _clean(text: Any, limit: int = 300) -> str:
    """Docstring/yorum icine gomulecek metni zararsizlastirir."""
    value = str(text or "").replace('"""', "'''").replace("\r", " ").replace("\n", " ")
    return value[:limit].strip()


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _operand(name: str, row_var: str) -> str:
    """Kural operandini Python ifadesine cevirir.

    ``"<kolon>_prev"`` bicimi onceki barin degerini kullanir (orn. sikisma
    filtresinin breakout barindan once kontrol edilmesi).
    """
    if isinstance(name, (int, float)) and not isinstance(name, bool):
        return _format_number(float(name))
    text = str(name)
    if text.endswith("_prev") and len(text) > 5:
        base = text[: -len("_prev")]
        target_var = "prev" if row_var == "row" else row_var
        if base in RAW_COLUMNS:
            return f'{target_var}["{base}"]'
        return f'{target_var}["{base}"]'
    if text in RAW_COLUMNS:
        return f'{row_var}["{text}"]'
    if text in CONSTANTS:
        return _format_number(CONSTANTS[text])
    try:
        return _format_number(float(text))
    except ValueError:
        pass
    return f'{row_var}["{text}"]'


def rule_expression(rule: Rule, row_var: str = "row", prev_var: str = "prev") -> str:
    """Tek kurali guvenli Python ifadesine cevirir (cross_* dahil)."""
    left = _operand(rule.left, row_var)
    right = _operand(rule.right, row_var)
    if rule.op == "cross_above":
        prev_left = _operand(rule.left, prev_var)
        prev_right = _operand(rule.right, prev_var)
        return f"({left} > {right} and {prev_left} <= {prev_right})"
    if rule.op == "cross_below":
        prev_left = _operand(rule.left, prev_var)
        prev_right = _operand(rule.right, prev_var)
        return f"({left} < {right} and {prev_left} >= {prev_right})"
    return f"({left} {rule.op} {right})"


def rule_block(rules: Sequence[Rule], row_var: str = "row", prev_var: str = "prev") -> str:
    """Kurallari ``and`` ile birlestirir; bos liste -> True."""
    if not rules:
        return "True"
    parts = [rule_expression(rule, row_var, prev_var) for rule in rules]
    return " and ".join(parts)


def _param_entries(spec: StrategySpec) -> Dict[str, Any]:
    params: Dict[str, Any] = {"atr_period": 14}
    has_atr_column = any(ind.key == "atr" or ind.kind == "atr" for ind in spec.indicators)
    for ind in spec.indicators:
        resolved = ind.resolved_params()
        for name, value in resolved.items():
            key = f"{ind.key}_{name}"
            if key == "atr_period" and has_atr_column and ind.kind == "atr":
                # Kullanicinin ATR indikatoru baz ATR kolonunu besler
                params["atr_period"] = value
                params.pop(key, None)
                continue
            params[key] = value
    return params


#: Coklu cikti ureten indikatorlerde "value" takma adinin gercek anahtari
OUTPUT_ALIASES: Dict[str, Dict[str, str]] = {
    "adx": {"value": "adx"},
    "macd": {"value": "macd"},
}


def _resolved_output(kind: str, output: str) -> str:
    return OUTPUT_ALIASES.get(kind, {}).get(output, output)


def _indicator_lines(spec: StrategySpec) -> tuple[List[str], Set[str]]:
    """Indikator hesaplama satirlarini ve gereken importlari uretir."""
    lines: List[str] = []
    imports: Set[str] = {"atr"}
    keys = {ind.key for ind in spec.indicators}

    for ind in spec.indicators:
        template = INDICATOR_TEMPLATES.get(ind.kind)
        if template is None:
            raise CompileError(f"Indikator turu icin sablon yok: {ind.kind}")
        lines.append(
            template.format(
                key=ind.key,
                p="self.params",
                output=_resolved_output(ind.kind, ind.output),
                out="out",
            )
        )
        function = FUNCTION_FOR_KIND.get(ind.kind)
        if function:
            imports.add(function)

    if "atr" not in keys:
        lines.insert(
            0,
            'out["atr"] = atr(out["high"], out["low"], out["close"], self.params["atr_period"])',
        )
    return lines, imports


def compile_strategy(
    spec: StrategySpec,
    *,
    version: str = "v001",
    generated_by: str = "coder_agent",
    notes: Sequence[str] = (),
) -> str:
    """Spec'i calistirilabilir Python modul kaynagina cevirir."""
    spec.ensure_valid()
    params = _param_entries(spec)
    indicator_lines, imports = _indicator_lines(spec)
    indented_indicators = "\n".join("        " + line for line in indicator_lines)

    entry_condition = rule_block(spec.entry_long)
    if spec.filters:
        entry_condition = f"({rule_block(spec.filters)}) and ({entry_condition})"
    exit_condition = rule_block(spec.exit_long)

    risk_defaults = {
        "stop_loss_atr": float(spec.stop_loss_atr),
        "take_profit_atr": float(spec.take_profit_atr),
        "trailing_stop_atr": float(spec.trailing_stop_atr),
        "risk_per_trade_pct": float(spec.risk_per_trade_pct),
        "cooldown_bars": int(spec.cooldown_bars),
        "max_holding_bars": int(spec.max_holding_bars),
    }

    param_lines = "\n".join(f'            "{k}": {_format_number(float(v))},' for k, v in params.items())
    risk_lines = "\n".join(
        f'            "{k}": {_format_number(float(v))},' for k, v in risk_defaults.items()
    )
    risk_lines += '\n            "allow_short": False,'
    note_text = "\n".join(f"#   - {_clean(note)}" for note in dict.fromkeys(notes) if note)
    thesis = _clean(spec.thesis or "")
    summary = _clean(spec.describe())

    header = f'''"""Otomatik uretilmis strateji - ELLE DUZENLEMEYIN.

Uretici : {generated_by}
Surum   : {version}
Aile    : {spec.family}
Tez     : {thesis or "-"}
Ozet    : {summary}
{note_text}
Kod, StrategySpec'ten deterministik olarak derlenmistir. Degisiklik icin
spec'i guncelleyip yeniden derleyin (agency.strategies.compiler).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pandas as pd

from agency.data.indicators import {", ".join(sorted(FUNCTION_IMPORTS[f] for f in imports if f in FUNCTION_IMPORTS))}
from agency.strategies.base import BaseStrategy, Position, safe_float

SPEC: Dict[str, Any] = {spec.to_dict()!r}


class Strategy(BaseStrategy):
    """{spec.name} - {spec.family} stratejisi."""

    name = {spec.name!r}
    family = {spec.family!r}
    version = {version!r}
    timeframe = {spec.timeframe!r}
    thesis = {thesis!r}

    risk_defaults = {{
{risk_lines}
    }}

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {{
{param_lines}
        }}

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Indikatorleri vektorize olarak ekler."""
        out = frame.copy()
        for column in ("open", "high", "low", "close", "volume"):
            out[column] = out[column].astype(float)
{indented_indicators}
        return out

    def generate_signal(
        self,
        row: Mapping[str, Any],
        prev: Mapping[str, Any],
        position: Optional[Position],
    ) -> Dict[str, Any]:
        """Kural tabanli giris/cikis karari."""
        atr_value = safe_float(row.get("atr"), 0.0)

        if position is None:
            if {entry_condition}:
                return {{
                    "action": "buy",
                    "reason": "giris kosullari saglandi (atr=%.5f)" % atr_value,
                }}
            return {{"action": "hold", "reason": "giris kosulu saglanmadi"}}

        if {exit_condition}:
            return {{"action": "sell", "reason": "cikis kosulu saglandi"}}

        return {{"action": "hold", "reason": "pozisyon korunuyor"}}
'''
    return header


def build_filename(version: str, name: str) -> str:
    """``v001_trend_ema_rsi.py`` biciminde dosya adi uretir."""
    return f"{version}_{slugify(name)}.py"


def spec_from_payload(payload: Any) -> StrategySpec:
    """LLM/harici kaynaktan gelen spec payload'ini normalize eder."""
    if isinstance(payload, StrategySpec):
        return payload
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise CompileError(f"Spec sozluk olmali: {type(payload).__name__}")
    for key in ("strategy", "spec", "strategy_spec"):
        if isinstance(payload.get(key), dict):
            payload = payload[key]
            break
    return StrategySpec.from_dict(payload)

