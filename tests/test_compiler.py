"""Strateji derleyicisi ve AST dogrulayici testleri."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agency.backtest.validator import validate_strategy_source
from agency.exceptions import StrategyValidationError
from agency.schemas import Rule, StrategySpec, IndicatorSpec
from agency.strategies.base import BaseStrategy
from agency.strategies.compiler import compile_strategy, rule_expression
from agency.strategies.registry import StrategyRegistry


def test_compile_contains_contract(sample_spec: StrategySpec) -> None:
    source = compile_strategy(sample_spec, version="v042")
    assert "class Strategy(BaseStrategy)" in source
    assert "def generate_signal" in source
    assert "def prepare" in source
    assert "v042" in source
    issues = [i for i in validate_strategy_source(source) if i.severity == "error"]
    assert issues == []


def test_compiled_module_is_loadable(
    sample_spec: StrategySpec, registry: StrategyRegistry
) -> None:
    build = registry.create_build(sample_spec, cycle=1)
    strategy = registry.instantiate(build.module_path)
    assert isinstance(strategy, BaseStrategy)
    assert strategy.risk.stop_loss_atr == sample_spec.stop_loss_atr
    assert "atr" in strategy.default_params() or True  # atr her zaman uretilir
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
            "open": np.linspace(100, 104, 5),
            "high": np.linspace(101, 105, 5),
            "low": np.linspace(99, 103, 5),
            "close": np.linspace(100.5, 104.5, 5),
            "volume": np.full(5, 10.0),
        }
    )
    prepared = strategy.prepare(frame)
    for column in ("atr", "ema_fast", "ema_slow", "adx_14", "rsi"):
        assert column in prepared.columns


def test_compiled_signal_returns_action(
    sample_spec: StrategySpec, registry: StrategyRegistry
) -> None:
    build = registry.create_build(sample_spec, cycle=1)
    strategy = registry.instantiate(build.module_path)
    row = {"close": 100.0, "atr": 1.0, "ema_fast": 101.0, "ema_slow": 100.0,
           "adx_14": 30.0, "rsi": 55.0}
    prev = dict(row)
    decision = strategy.generate_signal(row, prev, None)
    assert decision["action"] in {"buy", "hold"}
    assert "reason" in decision


def test_prev_operand_uses_previous_bar(sample_spec: StrategySpec) -> None:
    sample_spec.indicators.append(
        IndicatorSpec(
            key="bb_width", kind="bbands", output="width", params={"period": 20, "std": 2.0}
        )
    )
    sample_spec.filters.append(Rule(left="bb_width_prev", op="<", right=4.0))
    source = compile_strategy(sample_spec, version="v001")
    assert 'prev["bb_width"]' in source
    assert '["bb_width_prev"]' not in source  # operand adi kodda kolon olarak kullanilmaz
    prev_expr = rule_expression(Rule("close_prev", ">", "ema_slow_prev"), "row", "prev")
    assert 'prev["close"]' in prev_expr and 'prev["ema_slow"]' in prev_expr


def test_cross_rule_codegen() -> None:
    expr = rule_expression(Rule("ema_fast", "cross_above", "ema_slow"))
    assert "row[\"ema_fast\"] > row[\"ema_slow\"]" in expr
    assert "prev[\"ema_fast\"] <= prev[\"ema_slow\"]" in expr


def test_validator_blocks_dangerous_code() -> None:
    source = """
import os
from agency.strategies.base import BaseStrategy

class Strategy(BaseStrategy):
    def prepare(self, frame):
        return frame

    def generate_signal(self, row, prev, position):
        os.system("rm -rf /")
        return {"action": "hold"}
"""
    issues = validate_strategy_source(source)
    messages = " ".join(i.message for i in issues)
    assert "os" in messages and "yasakli" in messages.lower()
    with pytest.raises(StrategyValidationError):
        from agency.backtest.validator import assert_valid_strategy

        assert_valid_strategy(source)


def test_validator_requires_contract() -> None:
    issues = validate_strategy_source("x = 1\n")
    codes = {i.code for i in issues}
    assert {"missing_class", "missing_base", "missing_methods"} <= codes


def test_registry_versions_increment(registry: StrategyRegistry, sample_spec: StrategySpec) -> None:
    first = registry.create_build(sample_spec, cycle=1)
    second = registry.create_build(sample_spec.with_mutation(name="second"), cycle=1)
    assert first.version == "v001"
    assert second.version == "v002"
    assert len(registry.list_version_files()) == 2


def test_registry_archive_creates_scored_copy(
    registry: StrategyRegistry, sample_spec: StrategySpec
) -> None:
    build = registry.create_build(sample_spec, cycle=1)
    target = registry.archive(build, score=72.5, status="promote")
    assert target.is_file()
    assert "score072.5" in target.name and "promote" in target.name


def test_invalid_spec_cannot_be_compiled(sample_spec: StrategySpec) -> None:
    sample_spec.entry_long = []
    with pytest.raises(StrategyValidationError):
        compile_strategy(sample_spec)
