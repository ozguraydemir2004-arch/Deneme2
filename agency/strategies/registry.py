"""Strateji kayit defteri: uretim, versiyonlama, yukleme, arsivleme.

Her strateji ``agency/strategies/library/vNNN_slug.py`` olarak yazilir.
Surum numarasi monotonic artar; hicbir surum silinmez (denetlenebilirlik).
Basarisina gore puanlanan kopyalar ``runtime/archive`` altina alinir.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Type

from agency.backtest.validator import assert_valid_strategy
from agency.config import Settings
from agency.exceptions import StrategyValidationError
from agency.schemas import StrategyBuild, StrategySpec
from agency.strategies.base import BaseStrategy
from agency.strategies.compiler import build_filename, compile_strategy
from agency.utils import iso_now, sha256_text, slugify, write_text_atomic

VERSION_PATTERN = re.compile(r"^v(\d{3,})_.+\.py$")


@dataclass
class LoadedStrategy:
    """Yuklenmis strateji modulu."""

    path: Path
    module: ModuleType
    strategy_class: Type[BaseStrategy]


class StrategyRegistry:
    """Dosya sistemi + AST dogrulama uzerinden strateji yasam dongusu."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.library_dir = Path(settings.versioning.library_dir)
        self.archive_dir = Path(settings.versioning.archive_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ versions
    def list_version_files(self) -> List[Path]:
        return sorted(p for p in self.library_dir.glob("v*.py") if VERSION_PATTERN.match(p.name))

    def next_version(self) -> str:
        highest = 0
        for path in self.list_version_files():
            match = VERSION_PATTERN.match(path.name)
            if match:
                highest = max(highest, int(match.group(1)))
        # Dosya adlari esas alinir; boylece DB silinse bile cakisma olmaz
        return f"v{highest + 1:03d}"

    def _ensure_unique_path(self, version: str, name: str) -> Path:
        base = build_filename(version, name)
        path = self.library_dir / base
        if path.exists():
            stem = Path(base).stem
            for index in range(1, 1000):
                candidate = self.library_dir / f"{stem}_{index}.py"
                if not candidate.exists():
                    return candidate
        return path


    # -------------------------------------------------------------- create
    def create_build(
        self,
        spec: StrategySpec,
        *,
        cycle: int,
        parent_version: Optional[str] = None,
        attempt: int = 1,
        notes: Sequence[str] = (),
        generated_by: str = "coder_agent",
        version: Optional[str] = None,
    ) -> StrategyBuild:
        """Spec'i derler, dogrular, diske yazar ve build nesnesi dondurur."""
        spec.ensure_valid()
        version = version or self.next_version()
        source = compile_strategy(
            spec, version=version, generated_by=generated_by, notes=notes
        )
        assert_valid_strategy(source, filename=f"{version}_{slugify(spec.name)}.py")

        path = self._ensure_unique_path(version, spec.name)
        write_text_atomic(path, source)
        # Sozdizimi/sozlesme hatasi varsa burada patlar (savunma amacli 2. kontrol)
        self.load_class(path)

        return StrategyBuild(
            version=version,
            name=spec.name,
            slug=slugify(spec.name),
            module_path=str(path),
            code_hash=sha256_text(source),
            spec=spec,
            cycle=int(cycle),
            parent_version=parent_version,
            attempt=int(attempt),
            created_at=iso_now(),
            repair_notes=list(notes),
        )

    # ---------------------------------------------------------------- load
    def load_module(self, path: Path | str) -> ModuleType:
        path = Path(path)
        if not path.is_file():
            raise StrategyValidationError(f"Strateji dosyasi bulunamadi: {path}")
        module_name = f"agency_strategy_{slugify(path.stem)}_{sha256_text(str(path))[:8]}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:  # pragma: no cover - savunma
            raise StrategyValidationError(f"Modul yuklenemedi: {path}")
        module = importlib.util.module_from_spec(spec)
        root = str(self.settings.project_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        spec.loader.exec_module(module)
        return module

    def load_class(self, path: Path | str) -> Type[BaseStrategy]:
        module = self.load_module(path)
        strategy_class = getattr(module, "Strategy", None)
        if strategy_class is None:
            raise StrategyValidationError(f"'Strategy' sinifi yok: {path}")
        if not isinstance(strategy_class, type) or not issubclass(strategy_class, BaseStrategy):
            raise StrategyValidationError(f"'Strategy' BaseStrategy'den turemiyor: {path}")
        return strategy_class

    def instantiate(self, path: Path | str, params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
        return self.load_class(path)(params)

    # -------------------------------------------------------------- archive
    def archive(self, build: StrategyBuild, *, score: float, status: str) -> Path:
        """Puanli arsiv kopyasi olusturur (``runtime/archive``)."""
        source_path = Path(build.module_path)
        if not source_path.is_file():
            raise StrategyValidationError(f"Arsivlenecek dosya yok: {source_path}")
        target = (
            self.archive_dir
            / f"{build.version}_{slugify(build.name)}_score{score:05.1f}_{status}.py"
        )
        write_text_atomic(target, source_path.read_text(encoding="utf-8"))
        return target

    # ---------------------------------------------------------------- info
    @staticmethod
    def relative(path: Path | str) -> str:
        path = Path(path)
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(path)

    def describe_library(self, limit: int = 10) -> List[Dict[str, Any]]:
        files = self.list_version_files()[-limit:]
        return [{"file": path.name, "bytes": path.stat().st_size} for path in files]

