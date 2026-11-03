"""GERIYE DONUK TEST VE HATA YAKALAMA AJANI (Backtest & Debug Agent).

Akis:
    1. Strateji kaynagini AST ile dogrular (sozdizimi + guvenlik).
    2. Veriyi train/test olarak boler.
    3. Her kosuyu **ayri bir surecte** (sandbox) calistirir: zaman + bellek
       limiti, yakalanan stderr.
    4. Metrikleri (net kar, win rate, max drawdown, sharpe, profit factor) ve
       hata matrisini (dosya/satir/tip/mesaj) uretir.
    5. Sonuclari hafizaya ve rapor dosyalarina yazar.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from agency.agents.base import BaseAgent
from agency.backtest.validator import validate_strategy_source
from agency.schemas import BacktestMetrics, BacktestReport, ErrorMatrix, StrategyBuild
from agency.utils import iso_now, write_json

ERROR_HINTS = {
    "KeyError": "Indikator kolonu uretilmemis olabilir; spec'teki indikator anahtarlarini kontrol et.",
    "NameError": "Tanimlanmamis degisken; kural ifadesi yanlis operand kullaniyor olabilir.",
    "TypeError": "Tip uyumsuzlugu; operand veya parametre tipi hatali olabilir.",
    "ZeroDivisionError": "Sifira bolme; indikator periyodu veya fiyat verisi sifir olabilir.",
    "IndexError": "Veri araligi disina cikildi; warmup/bar sayisi yetersiz olabilir.",
    "AttributeError": "Strateji sozlesmesi ihlal edilmis (eksik metod/alan).",
    "ValueError": "Gecersiz deger; parametre araliklari veya veri kalitesi sorunlu.",
    "ImportError": "Modul yuklenemedi; strateji dosyasi bozuk veya bagimlilik eksik.",
    "TimeoutExpired": "Simulasyon zaman asimina ugradi (sonsuz dongu veya cok yavas kod).",
}


class BacktestAgent(BaseAgent):
    """Sandbox backtest calistirici ve hata matrisi uretici."""

    name = "backtest"
    role = "Geriye donuk test ve hata yakalama"

    # ------------------------------------------------------------------ run
    def run_suite(self, build: StrategyBuild, frame: pd.DataFrame, *, cycle: int) -> BacktestReport:
        """Train + out-of-sample (test) kosularini yapar ve birlestirir."""
        cfg = self.settings.backtest
        split_index = int(len(frame) * float(cfg.train_split))
        split_index = max(200, min(split_index, len(frame) - 120))
        train_frame = frame.iloc[:split_index].reset_index(drop=True)
        # Test dilimine indikatorlerin oturmasi icin 150 bar oncul eklenir
        test_start = max(0, split_index - 150)
        test_frame = frame.iloc[test_start:].reset_index(drop=True)

        self.info(
            "backtest_start",
            f"{build.version} icin train/test simulasyonu",
            train_bars=len(train_frame),
            test_bars=len(test_frame),
        )

        train_report = self._run_split(build, train_frame, cycle=cycle, split="train")
        if not train_report.ok:
            self.warn(
                "backtest_train_failed",
                "Train kosusu basarisiz",
                summary=train_report.errors.summary,
            )
            self._persist(train_report, cycle=cycle)
            return train_report

        test_report = self._run_split(build, test_frame, cycle=cycle, split="test")
        folds = self._run_folds(build, frame, cycle=cycle)
        combined = BacktestReport(
            strategy_version=build.version,
            name=build.name,
            status="ok" if test_report.ok else "failed",
            split="train+test",
            metrics=train_report.metrics,
            oos_metrics=test_report.metrics if test_report.ok else None,
            fold_metrics=folds,
            errors=test_report.errors,
            runtime_seconds=round(train_report.runtime_seconds + test_report.runtime_seconds, 3),
            equity_curve=test_report.equity_curve or train_report.equity_curve,
            trades=test_report.trades or train_report.trades,
            data_source=test_report.data_source,
            bars=train_report.bars + test_report.bars,
            period_start=train_report.period_start,
            period_end=test_report.period_end,
            created_at=iso_now(),
        )
        self._persist(combined, cycle=cycle)
        self.info(
            "backtest_done",
            f"{build.version}: train {train_report.metrics.short()} | "
            f"oos {test_report.metrics.short()} | "
            f"folds {[round(m.net_profit_pct, 2) for m in folds]}",
            version=build.version,
            oos_ok=test_report.ok,
            status=combined.status,
        )
        return combined

    def _run_folds(
        self, build: StrategyBuild, frame: pd.DataFrame, *, cycle: int, n_folds: int = 3
    ) -> List[BacktestMetrics]:
        """Walk-forward dogrulama: veriyi sirali dilimlere boler.

        Tek bir OOS penceresi tek rejime baglidir; 3 dilimlik yuruyus ileri
        testi stratejinin rejim degisimlerine dayanikliligini olcer.
        """
        if len(frame) < 600:
            return []
        window = len(frame) // n_folds
        metrics: List[BacktestMetrics] = []
        for index in range(n_folds):
            start = max(0, index * window - 150)
            end = len(frame) if index == n_folds - 1 else (index + 1) * window
            segment = frame.iloc[start:end].reset_index(drop=True)
            report = self._run_split(build, segment, cycle=cycle, split=f"fold{index + 1}")
            if report.ok:
                metrics.append(report.metrics)
            else:
                self.warn(
                    "fold_failed",
                    f"{build.version} fold{index + 1} basarisiz: {report.errors.summary[:120]}",
                )
        return metrics



    # -------------------------------------------------------------- internals
    def _run_split(
        self, build: StrategyBuild, frame: pd.DataFrame, *, cycle: int, split: str
    ) -> BacktestReport:
        started = time.perf_counter()
        source_path = Path(build.module_path)
        workdir = Path(self.settings.runtime.reports_dir) / f"cycle_{cycle:03d}"
        workdir.mkdir(parents=True, exist_ok=True)

        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            errors = ErrorMatrix()
            errors.add(
                "io", "FileNotFoundError", f"Strateji dosyasi okunamadi: {exc}", file=str(source_path)
            )
            return self._error_report(build, errors, split, started)

        issues = [
            issue
            for issue in validate_strategy_source(source, filename=str(source_path))
            if issue.severity == "error"
        ]
        if issues:
            errors = ErrorMatrix()
            for issue in issues:
                errors.add(
                    "validation",
                    issue.code,
                    issue.message,
                    file=str(source_path),
                    line=issue.line,
                    hint="Derleyici/dogrulayici ciktisini kontrol et.",
                )
            self.warn(
                "validation_failed",
                "Kod dogrulamadan gecemedi",
                issues=[i.message for i in issues[:5]],
            )
            return self._error_report(build, errors, split, started)

        payload = self._execute_sandbox(build, frame, workdir=workdir, split=split, cycle=cycle)
        runtime = round(time.perf_counter() - started, 3)

        if payload.get("status") != "ok":
            errors = ErrorMatrix()
            raw = payload.get("error") or {}
            error_type = str(raw.get("type", "BacktestError"))
            message = str(raw.get("message", "bilinmeyen hata"))
            errors.add(
                str(payload.get("phase", "run")),
                error_type,
                message,
                file=raw.get("file"),
                line=raw.get("line"),
                hint=ERROR_HINTS.get(error_type, "Hata matrisini ve loglari incele."),
            )
            errors.raw_stderr = str(raw.get("traceback", ""))[-4000:]
            self.warn(
                "backtest_error",
                f"{build.version} {split} kosusu hata verdi: {error_type}: {message[:180]}",
                phase=payload.get("phase"),
            )
            report = self._error_report(build, errors, split, started)
            report.runtime_seconds = runtime
            return report

        metrics = BacktestMetrics.from_dict(payload.get("metrics") or {})
        report = BacktestReport(
            strategy_version=build.version,
            name=build.name,
            status="ok",
            split=split,
            metrics=metrics,
            errors=ErrorMatrix(),
            runtime_seconds=runtime,
            equity_curve=payload.get("equity_curve") or [],
            trades=payload.get("trades") or [],
            data_source=payload.get("source", "unknown"),
            bars=int(payload.get("bars") or 0),
            period_start=payload.get("period_start"),
            period_end=payload.get("period_end"),
            created_at=iso_now(),
        )
        self.info(
            "backtest_split_ok",
            f"{build.version} {split}: {metrics.short()}",
            split=split,
            trades=metrics.trades,
            buy_hold_pct=payload.get("buy_hold_pct"),
        )
        return report


    def _execute_sandbox(
        self,
        build: StrategyBuild,
        frame: pd.DataFrame,
        *,
        workdir: Path,
        split: str,
        cycle: int,
    ) -> Dict[str, Any]:
        csv_path = workdir / f"{build.version}_{split}.csv"
        job_path = workdir / f"{build.version}_{split}_job.json"
        out_path = workdir / f"{build.version}_{split}_result.json"

        export = frame.copy()
        export["timestamp"] = pd.to_datetime(export["timestamp"], utc=True)
        export.to_csv(csv_path, index=False)

        cfg = self.settings.backtest
        job = {
            "strategy_path": str(Path(build.module_path).resolve()),
            "data_path": str(csv_path.resolve()),
            "split": split,
            "engine": {
                "initial_capital": float(cfg.initial_capital),
                "fee_bps": float(cfg.fee_bps),
                "slippage_bps": float(cfg.slippage_bps),
                "allow_short": bool(cfg.allow_short),
            },
        }
        write_json(job_path, job)

        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.settings.project_root)
        env["PYTHONHASHSEED"] = "0"
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

        timeout = int(cfg.run_timeout_seconds)
        memory_bytes = int(cfg.max_memory_mb) * 1024 * 1024

        def _limits() -> None:  # pragma: no cover - alt surecte calisir
            try:
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
                resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 5))
            except (ValueError, OSError):
                pass

        command = [
            sys.executable,
            "-m",
            "agency.backtest.runner",
            "--job",
            str(job_path),
            "--out",
            str(out_path),
        ]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.settings.project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout + 15,
                preexec_fn=_limits if hasattr(resource, "RLIMIT_AS") else None,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "phase": "timeout",
                "error": {
                    "type": "TimeoutExpired",
                    "message": f"Simulasyon {timeout}s icinde bitmedi",
                    "traceback": "",
                },
            }
        except OSError as exc:
            return {
                "status": "error",
                "phase": "spawn",
                "error": {"type": type(exc).__name__, "message": str(exc), "traceback": ""},
            }

        if out_path.is_file():
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                payload = {
                    "status": "error",
                    "phase": "result",
                    "error": {
                        "type": "JSONDecodeError",
                        "message": f"Sonuc dosyasi okunamadi: {exc}",
                        "traceback": completed.stderr[-2000:],
                    },
                }
        else:
            payload = {
                "status": "error",
                "phase": "subprocess",
                "error": {
                    "type": "RuntimeError",
                    "message": (completed.stderr or completed.stdout or "Cikti uretilmedi")[-800:],
                    "traceback": (completed.stderr or "")[-4000:],
                },
            }

        if payload.get("status") == "ok":
            payload.setdefault("source", self.settings.data.provider)
            payload["runtime_seconds"] = round(time.perf_counter() - started, 3)
        elif not (payload.get("error") or {}).get("traceback"):
            payload.setdefault("error", {})["traceback"] = (completed.stderr or "")[-4000:]
        return payload

    def _error_report(
        self, build: StrategyBuild, errors: ErrorMatrix, split: str, started: float
    ) -> BacktestReport:
        return BacktestReport(
            strategy_version=build.version,
            name=build.name,
            status="error",
            split=split,
            metrics=BacktestMetrics(),
            errors=errors,
            runtime_seconds=round(time.perf_counter() - started, 3),
            data_source=self.settings.data.provider,
            created_at=iso_now(),
        )

    def _persist(self, report: BacktestReport, *, cycle: int) -> None:
        workdir = Path(self.settings.runtime.reports_dir) / f"cycle_{cycle:03d}"
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / f"{report.strategy_version}_{report.split.replace('+', '_')}.json"
        write_json(path, report.to_dict())
        self.memory.save_backtest(report, cycle=cycle, report_path=str(path))


