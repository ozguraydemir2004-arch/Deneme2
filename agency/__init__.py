"""Kripto Al-Sat Botu Gelistirme Ajans Sistemi.

Bes katmanli, kendi kendini optimize eden coklu ajan mimarisi:

    research  -> veri madenciligi + piyasa/strateji arastirmasi
    coder     -> OOP strateji kodu uretimi (versiyonlanmis)
    backtest  -> CCXT verisi uzerinde sandbox simulasyon + metrikler
    critic    -> skorlama, ders cikarma, kalici hafizaya terfi
    orchestrator -> 24/7 otonom dongu

Public API:
    ``from agency import AgencyOrchestrator, load_settings``
"""

from __future__ import annotations

__version__ = "1.0.0"

from agency.config import Settings, load_settings  # noqa: E402

__all__ = ["Settings", "load_settings", "__version__"]
