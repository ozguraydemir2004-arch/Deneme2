"""LLM katmani: OpenAI-uyumlu / Anthropic istemcileri + cevrimdisi mod.

Tasarim ilkeleri:
    * **Sifir zorunlu bagimlilik**: HTTP cagrilari ``urllib`` ile yapilir.
    * **Zarif dusus**: API anahtari yoksa veya cagri basarisiz olursa sistem
      deterministik planlayiciya duser; dongu asla durmaz.
    * **JSON sozlesmesi**: Ajanlar LLM'den yalnizca JSON ister ve cevabi
      :func:`extract_json` ile saglamlastirir.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from agency.config import Settings
from agency.exceptions import LLMError
from agency.utils import truncate

MessageLogger = Callable[[str, Dict[str, Any]], None]


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Metinden ilk gecerli JSON nesnesini cikarir.

    LLM'ler JSON'u ```json ... ``` blogu icinde veya aciklama metniyle
    dondurebilir; bu fonksiyon ikisini de tolere eder.
    """
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Dengeli parantez taramasi (string ve kacis karakterlerine saygi duyar)
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(cleaned):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                chunk = cleaned[start : index + 1]
                try:
                    parsed = json.loads(chunk)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    # Bu blok gecerli JSON degil: taramaya devam et
                    start = -1
    return None


class LLMClient(ABC):
    """Tum LLM istemcilerinin arayuzu."""

    provider: str = "offline"
    model: str = "none"
    available: bool = False

    def __init__(self, settings: Settings, log: Optional[MessageLogger] = None):
        self.settings = settings
        self._log = log

    # -- alt siniflar doldurur ---------------------------------------------
    @abstractmethod
    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = True,
    ) -> LLMResult:
        raise NotImplementedError

    # -- ortak yardimcilar --------------------------------------------------
    def json(
        self,
        system: str,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        result = self.complete(system, prompt, temperature=temperature, max_tokens=max_tokens)
        if not result.ok:
            return None
        return extract_json(result.text)

    def text(self, system: str, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        if not self.available:
            return ""
        result = self.complete(system, prompt, max_tokens=max_tokens, json_mode=False)
        return result.text if result.ok else ""

    # -- izleme -------------------------------------------------------------
    def _notify(self, event: str, payload: Dict[str, Any]) -> None:
        if self._log is not None:
            try:
                self._log(event, payload)
            except Exception:  # pragma: no cover - loglama asla akisi bozmaz
                pass

    def describe(self) -> str:
        return f"{self.provider}:{self.model}" if self.available else "offline(heuristik)"


class _HTTPClient(LLMClient):
    """Ortak HTTP yardimcilari (urllib tabanli, bagimliliksiz)."""

    def _post(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        for key, value in headers.items():
            request.add_header(key, value)

        attempts = max(1, int(self.settings.llm.max_retries) + 1)
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(
                    request, timeout=float(self.settings.llm.timeout_seconds)
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8")[:500]
                except Exception:  # pragma: no cover
                    pass
                last_error = LLMError(f"HTTP {exc.code}: {detail or exc.reason}")
                if exc.code in {400, 401, 403, 404}:
                    break  # tekrar denemek anlamsiz
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = LLMError(f"{type(exc).__name__}: {exc}")
            time.sleep(min(1.5 * (attempt + 1), 4.0))
        raise last_error or LLMError("Bilinmeyen LLM hatasi")


class OpenAICompatibleClient(_HTTPClient):
    """OpenAI Chat Completions uyumlu API (OpenAI, proxy, yerel vLLM...)."""

    provider = "openai"

    def __init__(self, settings: Settings, api_key: str, log: Optional[MessageLogger] = None):
        super().__init__(settings, log)
        self.api_key = api_key
        self.model = settings.llm.model
        self.base_url = settings.llm.base_url.rstrip("/")
        self.available = True

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = True,
    ) -> LLMResult:
        started = time.perf_counter()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(
                self.settings.llm.temperature if temperature is None else temperature
            ),
            "max_tokens": int(self.settings.llm.max_tokens if max_tokens is None else max_tokens),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            data = self._post(f"{self.base_url}/chat/completions", payload, headers)
            choices = data.get("choices") or []
            text = ""
            if choices:
                text = (choices[0].get("message") or {}).get("content") or ""
            result = LLMResult(
                text=text,
                provider=self.provider,
                model=self.model,
                latency_seconds=round(time.perf_counter() - started, 3),
            )
        except Exception as exc:
            result = LLMResult(
                text="",
                provider=self.provider,
                model=self.model,
                latency_seconds=round(time.perf_counter() - started, 3),
                error=str(exc),
            )
        self._notify(
            "llm_call",
            {
                "provider": self.provider,
                "model": self.model,
                "prompt_chars": len(prompt),
                "ok": result.ok,
                "latency": result.latency_seconds,
                "error": result.error,
                "response_head": truncate(result.text, 300),
            },
        )
        return result


class AnthropicClient(_HTTPClient):
    """Anthropic Messages API istemcisi."""

    provider = "anthropic"

    def __init__(self, settings: Settings, api_key: str, log: Optional[MessageLogger] = None):
        super().__init__(settings, log)
        self.api_key = api_key
        self.model = settings.llm.anthropic_model
        self.available = True

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = True,
    ) -> LLMResult:
        started = time.perf_counter()
        instruction = prompt
        if json_mode:
            instruction += "\n\nYalnizca gecerli bir JSON nesnesi dondur; aciklama ekleme."
        payload = {
            "model": self.model,
            "max_tokens": int(self.settings.llm.max_tokens if max_tokens is None else max_tokens),
            "temperature": float(
                self.settings.llm.temperature if temperature is None else temperature
            ),
            "system": system,
            "messages": [{"role": "user", "content": instruction}],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        try:
            data = self._post("https://api.anthropic.com/v1/messages", payload, headers)
            blocks = data.get("content") or []
            text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
            result = LLMResult(
                text=text,
                provider=self.provider,
                model=self.model,
                latency_seconds=round(time.perf_counter() - started, 3),
            )
        except Exception as exc:
            result = LLMResult(
                text="",
                provider=self.provider,
                model=self.model,
                latency_seconds=round(time.perf_counter() - started, 3),
                error=str(exc),
            )
        self._notify(
            "llm_call",
            {
                "provider": self.provider,
                "model": self.model,
                "prompt_chars": len(prompt),
                "ok": result.ok,
                "latency": result.latency_seconds,
                "error": result.error,
                "response_head": truncate(result.text, 300),
            },
        )
        return result


class OfflineHeuristicClient(LLMClient):
    """Anahtar yokken kullanilan deterministik istemci.

    ``available = False`` oldugu icin ajanlar LLM yolunu atlar ve kendi
    deterministik planlayicilarini kullanir. Bu sinif yalnizca arayuzu
    korumak ve loglarda net bir mesaj vermek icin vardir.
    """

    provider = "offline"
    model = "deterministic-planner"
    available = False

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = True,
    ) -> LLMResult:
        return LLMResult(
            text="",
            provider=self.provider,
            model=self.model,
            error="LLM anahtari yok; deterministik planlayici kullaniliyor",
        )


def build_llm(settings: Settings, log: Optional[MessageLogger] = None) -> LLMClient:
    """Konfigurasyona gore uygun istemciyi secer (asla hata firlatmaz)."""
    import os

    provider = (settings.llm.provider or "auto").lower()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    if provider == "openai" and openai_key:
        return OpenAICompatibleClient(settings, openai_key, log)
    if provider == "anthropic" and anthropic_key:
        return AnthropicClient(settings, anthropic_key, log)
    if provider == "auto":
        if openai_key:
            return OpenAICompatibleClient(settings, openai_key, log)
        if anthropic_key:
            return AnthropicClient(settings, anthropic_key, log)
    return OfflineHeuristicClient(settings, log)


