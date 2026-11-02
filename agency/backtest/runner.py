"""Sandbox backtest calistirici (ayri surec).

Kullanim:
    python -m agency.backtest.runner --job job.json --out result.json

Neden ayri surec?
    * LLM/derleyici kaynakli hatalar ana donguyu cokertmez.
    * Zaman ve bellek limitleri (``RLIMIT_AS``) uygulanabilir.
    * ``stderr`` yakalanarak yapilandirilmis "Hata Matrisi" uretilir.

Job semasi::

    {
      "strategy_path": "agency/strategies/library/v001_x.py",
      "data_path": "runtime/tmp/train.csv",
      "split": "train",
      "engine": {"initial_capital": 10000, "fee_bps": 7.5, ...},
      "bars_per_year": 8760
    }
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

import pandas as pd

# Sandbox icinde tekrar eden BLAS thread'lerini sinirla (bellek dostu)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _load_module(path: Path) -> ModuleType:
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    module_name = f"sandbox_strategy_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Modul yuklenemedi: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _error_payload(phase: str, exc: BaseException) -> dict:
    tb = traceback.extract_tb(exc.__traceback__)
    frame = tb[-1] if tb else None
    return {
        "status": "error",
        "phase": phase,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "file": frame.filename if frame else None,
            "line": frame.lineno if frame else None,
            "function": frame.name if frame else None,
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:],
        },
    }


def run_job(job: dict) -> dict:
    started = time.perf_counter()
    strategy_path = Path(job["strategy_path"])
    data_path = Path(job["data_path"])

    if not strategy_path.is_file():
        return {
            "status": "error",
            "phase": "import",
            "error": {"type": "FileNotFoundError", "message": f"Strateji yok: {strategy_path}",
                      "traceback": "", "file": None, "line": None},
        }
    if not data_path.is_file():
        return {
            "status": "error",
            "phase": "data",
            "error": {"type": "FileNotFoundError", "message": f"Veri yok: {data_path}",
                      "traceback": "", "file": None, "line": None},
        }

    try:
        frame = pd.read_csv(data_path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    except Exception as exc:
        return _error_payload("data", exc)

    try:
        module = _load_module(strategy_path)
        strategy_class = getattr(module, "Strategy", None)
        if strategy_class is None:
            raise AttributeError("Modulde 'Strategy' sinifi yok")
        from agency.strategies.base import BaseStrategy

        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError("'Strategy' BaseStrategy'den turemiyor")
        strategy = strategy_class()
    except Exception as exc:
        return _error_payload("import", exc)

    try:
        from agency.backtest.engine import BacktestEngine, EngineConfig

        engine_cfg = EngineConfig(**job.get("engine", {}))
        result = BacktestEngine(engine_cfg).run(strategy, frame)
    except Exception as exc:
        return _error_payload("run", exc)

    return {
        "status": "ok",
        "phase": "done",
        "split": job.get("split", "full"),
        "metrics": result.metrics.to_dict(),
        "trades": result.trade_dicts(limit=200),
        "equity_curve": result.equity_sample(max_points=200),
        "bars": result.bars,
        "exposure_bars": result.exposure_bars,
        "buy_hold_pct": round(result.buy_hold_pct, 4),
        "total_fees": round(result.total_fees, 4),
        "period_start": result.period_start,
        "period_end": result.period_end,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sandbox backtest calistirici")
    parser.add_argument("--job", required=True, help="Job JSON dosyasi")
    parser.add_argument("--out", required=True, help="Sonuc JSON dosyasi")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    except Exception as exc:
        payload = _error_payload("job", exc)
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"status": "error", "type": type(exc).__name__, "message": str(exc)}))
        return 2

    payload = run_job(job)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    if payload.get("status") != "ok":
        error = payload.get("error", {})
        print(f"{error.get('type')}: {error.get('message')}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", "metrics": payload["metrics"]}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
