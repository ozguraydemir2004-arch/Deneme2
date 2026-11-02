"""Dongu raporlari (Markdown) ve kontrol paneli.

Her dongu icin ``runtime/reports/cycle_NNN/report.md`` ve tum donguleri
ozetleyen ``runtime/reports/INDEX.md`` uretilir.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional

from agency.config import Settings
from agency.memory.repositories import MemoryStore
from agency.schemas import CycleResult, ResearchReport
from agency.utils import write_text_atomic


class CycleReporter:
    """Markdown rapor uretici."""

    def __init__(self, settings: Settings, memory: MemoryStore):
        self.settings = settings
        self.memory = memory
        self.reports_dir = Path(settings.runtime.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write_cycle(
        self,
        result: CycleResult,
        research: Optional[ResearchReport],
        evaluations: List[Mapping[str, object]],
    ) -> Path:
        directory = self.reports_dir / f"cycle_{result.cycle:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "report.md"

        lines: List[str] = [f"# Dongu {result.cycle:03d} Raporu", ""]
        lines.append(f"- **Durum:** {result.status}")
        lines.append(f"- **Baslangic:** {result.started_at}")
        lines.append(f"- **Bitis:** {result.finished_at}")
        lines.append(f"- **Sure:** {result.duration_seconds:.1f}s")
        lines.append(f"- **Veri kaynagi:** {result.data_source}")
        lines.append(
            f"- **En iyi surum:** {result.best_version or '-'} (skor {result.best_score:.1f})"
        )
        lines.append(f"- **Terfi eden:** {', '.join(result.promoted) or '-'}")
        lines.append("")

        if research is not None:
            market = research.market
            lines.append("## Piyasa Durumu")
            lines.append("")
            lines.append(f"- Rejim: **{market.regime}** / trend {market.trend_strength}")
            lines.append(
                f"- Volatilite rejimi: {market.volatility_regime} (ATR% {market.atr_pct:.2f})"
            )
            lines.append(
                f"- RSI: {market.rsi:.1f} | ADX: {market.adx:.1f} | "
                f"Hacim: {market.volume_ratio:.2f}x"
            )
            lines.append(f"- Son fiyat: {market.last_price:,.2f} | {market.bars} bar")
            lines.append("")
            if research.insights:
                lines.append("### Arastirma Notlari")
                lines.extend(f"- {item}" for item in research.insights)
                lines.append("")
            if research.top_matches:
                lines.append("### Benzer Tarihsel Basari Anlari (vektor hafiza)")
                lines.append("")
                lines.append("| Zaman | Benzerlik | Ileri getiri % | RSI | ADX | ATR% |")
                lines.append("|---|---|---|---|---|---|")
                for match in research.top_matches:
                    features = match.get("features") or {}
                    lines.append(
                        f"| {match.get('ts')} | {match.get('similarity')} | "
                        f"{match.get('forward_return_pct')} | {features.get('rsi')} | "
                        f"{features.get('adx')} | {features.get('atr_pct')} |"
                    )
                lines.append("")

        lines.append("## Strateji Denemeleri")
        lines.append("")
        if not evaluations:
            lines.append("_Bu dongude degerlendirilen strateji yok._")
        for item in evaluations:
            build = item.get("build")
            report = item.get("report")
            decision = item.get("decision")
            archive = item.get("archive")
            if build is None or report is None or decision is None:
                continue
            lines.append(f"### {build.version} - {build.name} ({build.spec.family})")
            lines.append("")
            lines.append(f"- Tez: {build.spec.thesis or '-'}")
            lines.append(f"- Karar: **{decision.verdict.upper()}** (skor {decision.score:.1f}/100)")
            lines.append(f"- Train: {report.metrics.short()}")
            if report.oos_metrics is not None:
                lines.append(f"- Out-of-sample: {report.oos_metrics.short()}")
            lines.append(
                "- Esikler: "
                + ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in decision.metric_gates.items())
            )
            lines.append(f"- Kod: `{build.module_path}`")
            if archive is not None:
                lines.append(f"- Arsiv: `{archive.archive_path}`")
            if decision.directives:
                lines.append("- Direktifler:")
                lines.extend(f"  - {d}" for d in decision.directives)
            if decision.reasons:
                lines.append("- Gerekceler:")
                lines.extend(f"  - {r}" for r in decision.reasons)
            if report.errors.entries:
                lines.append("- Hata Matrisi:")
                for entry in report.errors.entries[:6]:
                    location = f":{entry.line}" if entry.line else ""
                    lines.append(
                        f"  - [{entry.phase}] {entry.type}: {entry.message[:200]} "
                        f"({entry.file or '-'}{location})"
                    )
            lines.append("")

        if result.lessons:
            lines.append("## Donguden Cikan Dersler")
            lines.extend(f"- {lesson}" for lesson in result.lessons)
            lines.append("")
        if result.next_directives:
            lines.append("## Sonraki Dongu Direktifleri")
            lines.extend(f"- {directive}" for directive in result.next_directives)
            lines.append("")


    # --------------------------------------------------------------- index
    def write_index(self, limit: int = 30) -> Path:
        path = self.reports_dir / "INDEX.md"
        cycles = self.memory.list_cycles(limit=limit)
        stats = self.memory.stats()
        leaderboard = self.memory.leaderboard(limit=10)

        lines = ["# Ajans Kontrol Paneli", "", "## Ozet", ""]
        lines.append(f"- Dongu sayisi: **{stats['cycles']}**")
        lines.append(
            f"- Uretilen surum: **{stats['versions']}** "
            f"(terfi: {stats['promoted']}, red: {stats['rejected']})"
        )
        lines.append(f"- En iyi skor: **{stats['best_score']}** | ortalama: {stats['avg_score']}")
        lines.append(f"- Snapshot: {stats['snapshots']} (winner: {stats['winner_snapshots']})")
        lines.append(
            f"- Kalici hafiza stratejisi: {stats['memory_strategies']} | ders: {stats['lessons']}"
        )
        lines.append("")
        lines.append("## Donguler")
        lines.append("")
        lines.append("| Dongu | Durum | En iyi surum | Skor | Aday | Veri | Bitis |")
        lines.append("|---|---|---|---|---|---|---|")
        for cycle in cycles:
            lines.append(
                f"| {cycle['cycle']} | {cycle['status']} | {cycle.get('best_version') or '-'} | "
                f"{float(cycle.get('best_score') or 0):.1f} | {cycle.get('candidate_count') or 0} | "
                f"{cycle.get('data_source') or '-'} | {cycle.get('finished_at') or '-'} |"
            )
        lines.append("")
        lines.append("## Skor Tablosu (Top 10)")
        lines.append("")
        lines.append("| Surum | Isim | Aile | Durum | Skor | Dongu |")
        lines.append("|---|---|---|---|---|---|")
        for row in leaderboard:
            lines.append(
                f"| {row['version']} | {row['name']} | {row['family']} | {row['status']} | "
                f"{float(row['score']):.1f} | {row['cycle']} |"
            )
        lines.append("")
        write_text_atomic(path, "\n".join(lines) + "\n")
        return path

    def summary_text(self) -> str:
        stats = self.memory.stats()
        return (
            f"dongu={stats['cycles']} surum={stats['versions']} terfi={stats['promoted']} "
            f"en_iyi_skor={stats['best_score']} snapshot={stats['snapshots']} "
            f"hafiza={stats['memory_strategies']}"
        )
