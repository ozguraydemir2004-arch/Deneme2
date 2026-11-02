"""Kucuk, bagimliliksiz yardimci fonksiyonlar."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat(timespec="seconds")


def timestamp_slug() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str, max_length: int = 48) -> str:
    """Dosya adi guvenli, ASCII slug uretir (Turkce karakterleri cevirir)."""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return (text or "unnamed")[:max_length]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def write_text_atomic(path: Path, content: str) -> Path:
    """Yazma islemini atomik yapar; yarim dosya kalmasini engeller."""
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, payload: Any, indent: int = 2) -> Path:
    return write_text_atomic(
        path,
        json.dumps(payload, indent=indent, ensure_ascii=False, default=str) + "\n",
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def round_or_none(value: Any, digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / abs(old) * 100.0


def truncate(text: Any, limit: int = 500) -> str:
    value = str(text).replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


@contextmanager
def timer() -> Iterator[dict]:
    """``with timer() as t:`` -> t["elapsed"] saniye cinsinden dolar."""
    box: dict = {"start": time.perf_counter(), "elapsed": 0.0}
    try:
        yield box
    finally:
        box["elapsed"] = round(time.perf_counter() - box["start"], 3)


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"
