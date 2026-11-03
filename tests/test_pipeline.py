"""Uctan uca dongu (pipeline) testleri.

Native ve LangGraph backend'lerinin ayni cikti sozlesmesini urettigini
dogrular: strateji dosyalari, backtest kayitlari, critic kararlari, raporlar.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agency.backtest.validator import validate_strategy_source
from agency.graph import attach_langgraph_backend, langgraph_available
from agency.memory.database import Database
from agency.memory.repositories import MemoryStore
from agency.orchestrator import AgencyOrchestrator


def _open_memory(settings) -> MemoryStore:
    return MemoryStore(Database(settings.memory.database))


def test_full_native_cycle(settings) -> None:
    orchestrator = AgencyOrchestrator(
        settings, console=False, max_cycles=1, sleep_seconds=0
    )
    try:
        results = orchestrator.run()
    finally:
        orchestrator.close()

    assert len(results) == 1
    result = results[0]
    assert result.status == "completed", result.error
    assert result.candidate_count >= 1
    assert result.versions, "en az bir strateji surumu uretilmeli"
    assert result.duration_seconds >= 0

    # 1) Strateji dosyalari diske yazildi ve dogrulayicidan geciyor
    library = Path(settings.versioning.library_dir)
    written = sorted(library.glob("v*.py"))
    assert len(written) == len(result.versions)
    for path in written:
        issues = [
            i for i in validate_strategy_source(path.read_text(encoding="utf-8")) if i.severity == "error"
        ]
        assert issues == [], f"{path.name}: {issues}"

    # 2) Arsiv kopyalari puanlandi
    archive = sorted(Path(settings.versioning.archive_dir).glob("*.py"))
    assert len(archive) >= len(result.versions)
    assert all("score" in item.name for item in archive)

    # 3) Raporlar ve durum dosyasi
    report_path = Path(settings.runtime.reports_dir) / f"cycle_{result.cycle:03d}" / "report.md"
    assert report_path.is_file()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Strateji Denemeleri" in report_text
    assert (Path(settings.runtime.reports_dir) / "INDEX.md").is_file()
    assert Path(settings.runtime.state_file).is_file()

    # 4) Veritabani kayitlari
    memory = _open_memory(settings)
    try:
        stats = memory.stats()
        assert stats["cycles"] == 1
        assert stats["versions"] == len(result.versions)
        assert stats["backtests"] >= len(result.versions)
        assert stats["snapshots"] > 0
        assert stats["lessons"] > 0
        stored_cycle = memory.get_cycle(result.cycle)
        assert stored_cycle["status"] == "completed"
        for version in result.versions:
            row = memory.get_strategy(version)
            assert row is not None
            assert row["status"] in {"promoted", "rework", "rejected"}
            assert memory.backtest_history(version)
            assert memory.review_history(version)
    finally:
        memory.close()


def test_second_cycle_continues_numbering(settings) -> None:
    orchestrator = AgencyOrchestrator(settings, console=False, max_cycles=2, sleep_seconds=0)
    try:
        results = orchestrator.run()
    finally:
        orchestrator.close()
    assert [r.cycle for r in results] == [1, 2]
    versions = [v for r in results for v in r.versions]
    assert len(versions) == len(set(versions)), "surum numaralari benzersiz olmali"


@pytest.mark.skipif(not langgraph_available(), reason="langgraph kurulu degil")
def test_langgraph_backend_cycle(settings) -> None:
    orchestrator = AgencyOrchestrator(settings, console=False, max_cycles=1, sleep_seconds=0)
    try:
        assert attach_langgraph_backend(orchestrator) is True
        assert orchestrator.cycle_executor is not None
        results = orchestrator.run()
    finally:
        orchestrator.close()

    assert results[0].status == "completed", results[0].error
    assert results[0].versions
    memory = _open_memory(settings)
    try:
        assert memory.stats()["versions"] >= 1
    finally:
        memory.close()


def test_backend_falls_back_without_package(settings, monkeypatch) -> None:
    from agency import crew

    monkeypatch.setattr(crew, "crewai_available", lambda: False)
    orchestrator = AgencyOrchestrator(settings, console=False, max_cycles=1, sleep_seconds=0)
    try:
        assert crew.attach_crewai_backend(orchestrator) is False
        assert orchestrator.cycle_executor is None  # native akis
    finally:
        orchestrator.close()
