"""Kripto haber tarayicisi (Research Agent'in web bacagi).

Ag erisimi yoksa veya saglayici hata verirse bos liste doner; sistem haber
olmadan da calismaya devam eder. Sentiment, anahtar kelime tabanli basit bir
skorla hesaplanir (LLM gerekmez).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

BULLISH_WORDS = {
    "surge", "rally", "bullish", "record", "approval", "inflow", "adoption",
    "upgrade", "breakout", "gains", "soar", "etf", "institutional", "halving",
    "yükseliş", "rekor", "onay", "giriş",
}
BEARISH_WORDS = {
    "crash", "plunge", "bearish", "hack", "exploit", "outflow", "ban", "lawsuit",
    "liquidation", "selloff", "drop", "fear", "regulation", "fraud",
    "düşüş", "çöküş", "yasak", "dava",
}
KEYWORDS = (
    "bitcoin", "btc", "etf", "halving", "sec", "fed", "inflation", "rate",
    "liquidity", "whale", "miner", "regulation", "stablecoin", "altcoin",
)


class NewsScanner:
    """Hafif, bagimliliksiz haber tarayicisi."""

    def __init__(self, endpoint: str, timeout: int = 10, max_items: int = 12, enabled: bool = True):
        self.endpoint = endpoint
        self.timeout = int(timeout)
        self.max_items = int(max_items)
        self.enabled = bool(enabled)
        self.last_error: Optional[str] = None

    def fetch(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        request = urllib.request.Request(
            self.endpoint,
            headers={"User-Agent": "agency-research-agent/1.0", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="ignore"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []

        raw_items = payload.get("Data") if isinstance(payload, dict) else payload
        if not isinstance(raw_items, list):
            self.last_error = "beklenmeyen haber formati"
            return []

        items: List[Dict[str, Any]] = []
        for entry in raw_items[: self.max_items]:
            if not isinstance(entry, dict):
                continue
            title = str(entry.get("title") or "").strip()
            if not title:
                continue
            body = f"{title} {entry.get('body', '')}".lower()
            items.append(
                {
                    "title": title[:240],
                    "source": str(entry.get("source_info", {}).get("name") or entry.get("source") or ""),
                    "url": str(entry.get("url") or ""),
                    "published_at": str(entry.get("published_on") or ""),
                    "sentiment": self._sentiment(body),
                    "keywords": [k for k in KEYWORDS if k in body][:6],
                }
            )
        self.last_error = None
        return items

    @staticmethod
    def _sentiment(text: str) -> str:
        bull = sum(1 for w in BULLISH_WORDS if w in text)
        bear = sum(1 for w in BEARISH_WORDS if w in text)
        if bull > bear:
            return "bullish"
        if bear > bull:
            return "bearish"
        return "neutral"

    @staticmethod
    def digest(items: List[Dict[str, Any]], limit: int = 8) -> List[str]:
        lines = []
        for item in items[:limit]:
            title = re.sub(r"\s+", " ", item.get("title", ""))[:150]
            lines.append(f"[{item.get('sentiment', 'neutral')}] {title}")
        return lines
