#!/usr/bin/env python3
"""Kripto Al-Sat Botu Gelistirme Ajans Sistemi - CLI giris noktasi.

Kullanim ornekleri
------------------
    python main.py run --cycles 2                  # 2 dongu calistir
    python main.py run                             # 24/7 (max_cycles=0 ise sinirsiz)
    python main.py run --cycles 5 --sleep 30        # donguler arasi 30s bekle
    python main.py once                            # tek dongu
    python main.py run --backend langgraph          # LangGraph state machine
    python main.py status                          # hafiza/panel ozeti
    python main.py doctor                          # ortam kontrolu

Tum secenekler ortam degiskenleri ile de override edilebilir (bkz. .env.example).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agency import __version__  # noqa: E402
from agency.config import ConfigError, load_settings  # noqa: E402
from agency.orchestrator import AgencyOrchestrator  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agencysystem",
        description="Kendi kendini gelistiren kripto al-sat botu ajans sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, default=None, help="Ayarlar YAML yolu")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--symbol", type=str, default=None, help="Islem cifti (BTC/USDT)")
        sub.add_argument("--timeframe", type=str, default=None, help="Zaman dilimi (1h)")
        sub.add_argument(
            "--data-provider",
            choices=["auto", "ccxt", "synthetic"],
            default=None,
            help="Veri kaynagi (auto: CCXT dene, olmazsa sentetik)",
        )
        sub.add_argument(
            "--llm",
            choices=["auto", "openai", "anthropic", "offline"],
            default=None,
            help="LLM saglayicisi (offline: deterministik planlayici)",
        )
        sub.add_argument(
            "--backend",
            choices=["native", "crewai", "langgraph"],
            default=None,
            help="Orkestrasyon backend'i",
        )
        sub.add_argument("--quiet", action="store_true", help="Konsol loglarini sustur")

    run_parser = subparsers.add_parser("run", help="Otonom donguyu baslat")
    add_common(run_parser)
    run_parser.add_argument("--cycles", type=int, default=None, help="Dongu sayisi (0 = sinirsiz)")
    run_parser.add_argument("--sleep", type=float, default=None, help="Dongular arasi bekleme (sn)")
    run_parser.add_argument(
        "--max-runtime", type=float, default=None, help="Toplam calisma suresi limiti (sn)"
    )

    once_parser = subparsers.add_parser("once", help="Tek dongu calistir")
    add_common(once_parser)

    status_parser = subparsers.add_parser("status", help="Hafiza ve panel ozeti")
    status_parser.add_argument("--limit", type=int, default=10, help="Listelenecek dongu sayisi")

    doctor_parser = subparsers.add_parser("doctor", help="Ortam ve bagimlilik kontrolu")
    add_common(doctor_parser)

    return parser


def _overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if getattr(args, "symbol", None):
        overrides.setdefault("exchange", {})["symbol"] = args.symbol
    if getattr(args, "timeframe", None):
        overrides.setdefault("exchange", {})["timeframe"] = args.timeframe
    if getattr(args, "data_provider", None):
        overrides.setdefault("data", {})["provider"] = args.data_provider
    if getattr(args, "llm", None):
        overrides.setdefault("llm", {})["provider"] = args.llm
    if getattr(args, "backend", None):
        overrides.setdefault("runtime", {})["backend"] = args.backend
    return overrides


def attach_backend(orchestrator: AgencyOrchestrator) -> str:
    """Konfigurasyona gore backend'i baglar; etkin backend adini dondurur."""
    backend = (orchestrator.settings.runtime.backend or "native").lower()
    if backend == "langgraph":
        from agency.graph import attach_langgraph_backend

        return "langgraph" if attach_langgraph_backend(orchestrator) else "native"
    if backend == "crewai":
        from agency.crew import attach_crewai_backend

        return "crewai" if attach_crewai_backend(orchestrator) else "native"
    orchestrator.logger.info("backend_selected", "Native backend aktif (yerlesik akis)")
    return "native"


def cmd_run(args: argparse.Namespace, *, single: bool = False) -> int:
    settings = load_settings(args.config, _overrides(args))
    cycles = 1 if single else getattr(args, "cycles", None)
    if cycles is None:
        cycles = settings.targets.max_cycles_per_run
    sleep_seconds = getattr(args, "sleep", None)
    if sleep_seconds is None:
        sleep_seconds = settings.targets.cycle_sleep_seconds
    max_runtime = getattr(args, "max_runtime", None)

    orchestrator = AgencyOrchestrator(
        settings,
        console=not args.quiet,
        max_cycles=cycles,
        sleep_seconds=sleep_seconds,
    )
    backend = attach_backend(orchestrator)

    def _handle_signal(signum: int, _frame: Any) -> None:
        orchestrator.logger.warning(
            "signal", f"Sinyal alindi ({signum}); mevcut dongu bitince durdurulacak"
        )
        orchestrator.request_stop()

    for signal_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signal_name):
            try:
                signal.signal(getattr(signal, signal_name), _handle_signal)
            except ValueError:  # pragma: no cover - ana thread degilse
                pass

    timer: Optional[threading.Timer] = None
    if max_runtime and max_runtime > 0:
        timer = threading.Timer(max_runtime, orchestrator.request_stop)
        timer.daemon = True
        timer.start()

    exit_code = 0
    try:
        results = orchestrator.run()
        _print_summary(orchestrator, results, backend)
    except KeyboardInterrupt:  # pragma: no cover - sinyal yakalandi
        orchestrator.request_stop()
    except Exception as exc:
        orchestrator.logger.exception("fatal", f"Kritik hata: {exc}")
        exit_code = 1
    finally:
        if timer is not None:
            timer.cancel()
        try:
            orchestrator.close()
        except Exception as exc:  # pragma: no cover
            print(f"Kapatma uyarisi: {exc}", file=sys.stderr)
    return exit_code


def _print_summary(orchestrator: AgencyOrchestrator, results: list, backend: str) -> None:
    stats = orchestrator.memory.stats()
    print("")
    print("=" * 78)
    print(f" AJANS OZETI (backend={backend}, LLM={orchestrator.llm.describe()})")
    print("=" * 78)
    for result in results:
        promoted = ", ".join(result.promoted) or "-"
        print(
            f" Dongu {result.cycle:03d} | {result.status:<9} | en iyi: "
            f"{result.best_version or '-':<6} skor={result.best_score:5.1f} | terfi: {promoted}"
        )
    print("-" * 78)
    print(
        f" Toplam: dongu={stats['cycles']} surum={stats['versions']} "
        f"terfi={stats['promoted']} snapshot={stats['snapshots']} "
        f"hafiza={stats['memory_strategies']} ders={stats['lessons']}"
    )
    print(f" Raporlar: {orchestrator.settings.runtime.reports_dir}")
    print("=" * 78)


def cmd_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config, _overrides(args))
    from agency.memory.database import Database
    from agency.memory.repositories import MemoryStore

    if not Path(settings.memory.database).is_file():
        print(f"Veritabani bulunamadi: {settings.memory.database}")
        print("Once 'python main.py run --cycles 1' calistirin.")
        return 1

    database = Database(settings.memory.database)
    memory = MemoryStore(database)
    stats = memory.stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\nSon donguler:")
    for row in memory.list_cycles(limit=args.limit):
        print(
            f"  #{row['cycle']:03d} {row['status']:<10} "
            f"en_iyi={row.get('best_version') or '-':<6} "
            f"skor={float(row.get('best_score') or 0):5.1f} "
            f"veri={row.get('data_source') or '-'}"
        )
    print("\nSkor tablosu:")
    for row in memory.leaderboard(limit=args.limit):
        print(
            f"  {row['version']:<6} {row['name'][:28]:<28} {row['family'][:20]:<20} "
            f"{row['status']:<9} skor={float(row['score']):5.1f}"
        )
    memory.close()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = load_settings(args.config, _overrides(args))
    checks: list = []

    def check(name: str, value: str, ok: bool) -> None:
        checks.append((name, value, ok))

    import importlib
    import os

    for module in ("pandas", "numpy", "ccxt", "yaml"):
        try:
            imported = importlib.import_module(module)
            check(f"paket:{module}", str(getattr(imported, "__version__", "?")), True)
        except Exception:  # pragma: no cover
            check(f"paket:{module}", "KURULU DEGIL", False)

    check("OPENAI_API_KEY", "tanimli" if os.environ.get("OPENAI_API_KEY") else "yok", True)
    check("ANTHROPIC_API_KEY", "tanimli" if os.environ.get("ANTHROPIC_API_KEY") else "yok", True)

    from agency.crew import crewai_available
    from agency.graph import langgraph_available

    check("langgraph", "kurulu" if langgraph_available() else "kurulu degil", True)
    check("crewai", "kurulu" if crewai_available() else "kurulu degil", True)
    check("veritabani", str(settings.memory.database), settings.memory.database.parent.is_dir())
    check("strateji kutuphanesi", str(settings.versioning.library_dir), True)
    check("config", str(settings.source_path or "varsayilan"), True)

    print(f"Ajans sistemi v{__version__} - ortam kontrolu\n")
    for name, value, ok in checks:
        print(f"  [{'OK   ' if ok else 'EKSIK'}] {name:<22} {value}")
    print(
        f"\nEtkin ayarlar: {settings.exchange.symbol} {settings.exchange.timeframe} | "
        f"veri={settings.data.provider} | llm={settings.llm.provider} | "
        f"backend={settings.runtime.backend}"
    )
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"
    try:
        if command == "run":
            return cmd_run(args)
        if command == "once":
            return cmd_run(args, single=True)
        if command == "status":
            return cmd_status(args)
        if command == "doctor":
            return cmd_doctor(args)
        parser.print_help()
        return 2
    except ConfigError as exc:
        print(f"Konfigurasyon hatasi: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\nIptal edildi.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())


