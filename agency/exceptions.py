"""Ajans sistemine ozgu istisnalar.

Her istisna, Critic/Optimizer ajaninin "Hata Matrisi"ne donusturebilecegi
yapilandirilmis alanlar tasir.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class AgencyError(RuntimeError):
    """Tum ajans hatalarinin tabani."""

    phase: str = "unknown"

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details: Dict[str, Any] = details or {}

    def to_error_matrix(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "type": type(self).__name__,
            "message": self.message,
            "details": self.details,
        }


class DataError(AgencyError):
    """Piyasa verisi cekilemedi / bozuk."""

    phase = "data"


class StrategyValidationError(AgencyError):
    """Uretilen strateji kodu guvenlik/dogrulama kontrollerinden gecmedi."""

    phase = "validation"


class BacktestError(AgencyError):
    """Simulasyon calistirilamadi veya sonuc uretmedi."""

    phase = "backtest"


class LLMError(AgencyError):
    """LLM saglayicisi hata dondurdu."""

    phase = "llm"


class MemoryError_(AgencyError):  # noqa: N801 - Python'un MemoryError'i ile karismasin
    """Hafiza katmani hatasi."""

    phase = "memory"


class CycleAborted(AgencyError):
    """Dongu kontrollu sekilde durduruldu (SIGINT / max runtime)."""

    phase = "orchestrator"
