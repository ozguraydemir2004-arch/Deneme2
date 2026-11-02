"""Ajan altyapisi: ortak baglam (context) ve taban sinif."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agency.agents.planner import StrategyPlanner
from agency.config import Settings
from agency.data.market_data import MarketDataProvider
from agency.llm import LLMClient
from agency.logging_utils import AgencyLogger
from agency.memory.repositories import MemoryStore
from agency.strategies.registry import StrategyRegistry


@dataclass
class AgentContext:
    """Tum ajanlarin paylastigi calisma zamani baglami."""

    settings: Settings
    memory: MemoryStore
    logger: AgencyLogger
    llm: LLMClient
    registry: StrategyRegistry
    data_provider: MarketDataProvider
    planner: StrategyPlanner
    cycle: int = 1
    run_id: str = "run"
    state: Dict[str, Any] = field(default_factory=dict)

    def child_logger(self, agent: str, cycle: Optional[int] = None) -> AgencyLogger:
        return self.logger.child(agent=agent, cycle=cycle if cycle is not None else self.cycle)


class BaseAgent(ABC):
    """Ajanlarin ortak davranisi: loglama ve LLM cagrisi sarmalayici."""

    name: str = "agent"
    role: str = ""

    def __init__(self, context: AgentContext):
        self.ctx = context
        self.settings: Settings = context.settings
        self.memory: MemoryStore = context.memory
        self.llm: LLMClient = context.llm
        self.registry: StrategyRegistry = context.registry
        self.planner: StrategyPlanner = context.planner
        self.log: AgencyLogger = context.child_logger(self.name)

    # ------------------------------------------------------------------ log
    def info(self, event: str, message: str = "", **data: Any) -> None:
        self.log.info(event, message, data=data or None)

    def warn(self, event: str, message: str = "", **data: Any) -> None:
        self.log.warning(event, message, data=data or None)

    def error(self, event: str, message: str = "", **data: Any) -> None:
        self.log.error(event, message, data=data or None)

    # ------------------------------------------------------------------ llm
    def ask_json(
        self,
        system: str,
        prompt: str,
        *,
        label: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """LLM'den JSON ister; basarisizsa ``None`` doner (dongu durmaz)."""
        if not self.llm.available:
            self.info(
                "llm_skipped",
                f"LLM devre disi ({self.llm.describe()}); deterministik yol kullanilacak",
                label=label,
            )
            return None
        self.info("llm_request", f"LLM cagrisi: {label}", prompt_chars=len(prompt))
        payload = self.llm.json(
            system, prompt, temperature=temperature, max_tokens=max_tokens
        )
        if payload is None:
            self.warn("llm_empty", f"LLM gecerli JSON dondurmedi: {label}")
            return None
        self.info(
            "llm_response",
            f"LLM cevabi alindi: {label}",
            keys=sorted(payload.keys())[:10],
        )
        return payload

    def ask_text(self, system: str, prompt: str, *, label: str) -> str:
        if not self.llm.available:
            return ""
        text = self.llm.text(system, prompt)
        if text:
            self.info("llm_response", f"LLM metin cevabi: {label}", chars=len(text))
        return text
