"""LLM katmani testleri: JSON ayristirma ve istemci secimi (ag cagrisi yok)."""

from __future__ import annotations

import pytest

from agency.llm import (
    AnthropicClient,
    OfflineHeuristicClient,
    OpenAICompatibleClient,
    build_llm,
    extract_json,
)


def test_extract_plain_json() -> None:
    assert extract_json('{"a": 1, "b": [1,2]}') == {"a": 1, "b": [1, 2]}


def test_extract_fenced_json() -> None:
    text = 'Aciklama:\n```json\n{"verdict": "promote", "score": 88}\n```\nTesekkurler'
    assert extract_json(text) == {"verdict": "promote", "score": 88}


def test_extract_json_inside_prose() -> None:
    text = 'Sonuc: {"strategy": {"name": "x"}} ve devam'
    assert extract_json(text) == {"strategy": {"name": "x"}}


def test_extract_json_invalid() -> None:
    assert extract_json("") is None
    assert extract_json("json yok") is None
    assert extract_json("[1, 2, 3]") is None


def test_extract_json_nested_balanced() -> None:
    text = 'bas {"a": {"b": {"c": 3}}, "d": "}"} son'
    assert extract_json(text) == {"a": {"b": {"c": 3}}, "d": "}"}


def test_build_llm_offline_without_keys(monkeypatch: pytest.MonkeyPatch, settings) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings.llm.provider = "auto"
    client = build_llm(settings)
    assert isinstance(client, OfflineHeuristicClient)
    assert client.available is False
    assert client.json("sys", "prompt") is None
    assert client.text("sys", "prompt") == ""
    assert "offline" in client.describe()


def test_build_llm_openai_with_key(monkeypatch: pytest.MonkeyPatch, settings) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings.llm.provider = "auto"
    client = build_llm(settings)
    assert isinstance(client, OpenAICompatibleClient)
    assert client.available is True
    assert client.describe().startswith("openai:")


def test_build_llm_anthropic_with_key(monkeypatch: pytest.MonkeyPatch, settings) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    settings.llm.provider = "anthropic"
    client = build_llm(settings)
    assert isinstance(client, AnthropicClient)
    assert client.describe().startswith("anthropic:")
