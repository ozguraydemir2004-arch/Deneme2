"""Yapilandirilmis loglama.

Iki kanal uretir:
    * ``runtime/logs/agency-YYYY-MM-DD.jsonl``  -> makine okunur olay kaydi
    * ``runtime/logs/agency-YYYY-MM-DD.log``     -> insan okunur ozet

Her kayit su alanlari tasir: ``ts``, ``level``, ``cycle``, ``agent``,
``event``, ``message`` ve ``data``.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from agency.utils import ensure_dir, iso_now, truncate, utc_now

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class AgencyLogger:
    """Dongu ve ajan bazli, thread-safe, cift kanalli logger."""

    def __init__(
        self,
        log_dir: Path | str,
        *,
        level: str = "INFO",
        jsonl: bool = True,
        console: bool = True,
        cycle: Optional[int] = None,
        agent: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.log_dir = ensure_dir(Path(log_dir))
        self.level = level.upper()
        self.jsonl_enabled = jsonl
        self.cycle = cycle
        self.agent = agent
        self.run_id = run_id
        self._lock = threading.Lock()
        self._events: list[Dict[str, Any]] = []

        stem = f"agency-{utc_now().strftime('%Y-%m-%d')}"
        self.jsonl_path = self.log_dir / f"{stem}.jsonl"
        self.text_path = self.log_dir / f"{stem}.log"

        self._console = console
        self._logger = logging.getLogger(f"agency.{run_id or 'default'}")
        self._logger.setLevel(_LEVELS.get(self.level, logging.INFO))
        self._logger.propagate = False
        if console and not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    # -- kanal yonetimi -----------------------------------------------------
    def child(self, *, cycle: Optional[int] = None, agent: Optional[str] = None) -> "AgencyLogger":
        return AgencyLogger(
            self.log_dir,
            level=self.level,
            jsonl=self.jsonl_enabled,
            console=self._console,
            cycle=cycle if cycle is not None else self.cycle,
            agent=agent if agent is not None else self.agent,
            run_id=self.run_id,
        )

    # -- yazma --------------------------------------------------------------
    def event(
        self,
        event: str,
        message: str = "",
        *,
        level: str = "INFO",
        data: Optional[Dict[str, Any]] = None,
        cycle: Optional[int] = None,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = {
            "ts": iso_now(),
            "level": level.upper(),
            "run_id": self.run_id,
            "cycle": cycle if cycle is not None else self.cycle,
            "agent": agent if agent is not None else self.agent,
            "event": event,
            "message": message,
            "data": data or {},
        }
        line = self._format_text(record)
        with self._lock:
            if self.jsonl_enabled:
                with self.jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            with self.text_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._events.append(record)
            if len(self._events) > 5000:
                del self._events[:2500]
        numeric = _LEVELS.get(record["level"], logging.INFO)
        if self._logger.isEnabledFor(numeric):
            self._logger.log(numeric, line)
        return record

    def debug(self, event: str, message: str = "", **kwargs: Any) -> Dict[str, Any]:
        return self.event(event, message, level="DEBUG", **kwargs)

    def info(self, event: str, message: str = "", **kwargs: Any) -> Dict[str, Any]:
        return self.event(event, message, level="INFO", **kwargs)

    def warning(self, event: str, message: str = "", **kwargs: Any) -> Dict[str, Any]:
        return self.event(event, message, level="WARNING", **kwargs)

    def error(self, event: str, message: str = "", **kwargs: Any) -> Dict[str, Any]:
        return self.event(event, message, level="ERROR", **kwargs)

    def exception(self, event: str, message: str = "", **kwargs: Any) -> Dict[str, Any]:
        import traceback

        data = dict(kwargs.pop("data", {}) or {})
        data.setdefault("traceback", traceback.format_exc(limit=8))
        return self.event(event, message, level="ERROR", data=data, **kwargs)

    # -- okuma --------------------------------------------------------------
    def recent_events(self, limit: int = 50) -> list[Dict[str, Any]]:
        with self._lock:
            return list(self._events[-limit:])

    @staticmethod
    def read_jsonl(path: Path, limit: int = 200) -> list[Dict[str, Any]]:
        path = Path(path)
        if not path.is_file():
            return []
        rows: list[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows[-limit:]

    # -- bicimlendirme ------------------------------------------------------
    @staticmethod
    def _format_text(record: Dict[str, Any]) -> str:
        scope_parts = []
        if record.get("cycle") is not None:
            scope_parts.append(f"c{record['cycle']:03d}")
        if record.get("agent"):
            scope_parts.append(str(record["agent"]))
        scope = f"[{'/'.join(scope_parts)}] " if scope_parts else ""
        message = truncate(record.get("message", ""), 400)
        data = record.get("data") or {}
        extra = ""
        if data:
            try:
                extra = " | " + truncate(json.dumps(data, ensure_ascii=False, default=str), 600)
            except (TypeError, ValueError):  # pragma: no cover
                extra = ""
        return (
            f"{record['ts']} {record['level']:<7} {scope}"
            f"{record['event']:<28} {message}{extra}"
        )


def build_logger(
    log_dir: Path | str,
    *,
    level: str = "INFO",
    jsonl: bool = True,
    console: bool = True,
    run_id: Optional[str] = None,
) -> AgencyLogger:
    return AgencyLogger(log_dir, level=level, jsonl=jsonl, console=console, run_id=run_id)

