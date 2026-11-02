"""CrewAI tabanli orkestrasyon backend'i (opsiyonel).

ONEMLI TASARIM KARARI
    CrewAI, LLM zorunlu kilan bir cercevedir. Bu projede **kod uretimi ve
    backtest deterministik** kalmak zorundadir (guvenlik + tekrarlanabilirlik).
    Bu nedenle CrewAI backend'i bir **danisma konseyi** olarak calisir:

        Arastirmaci -> Stratejist -> Backtest Muhendisi -> Risk Yoneticisi

    Konsey nitel degerlendirme uretir; uygulama (kod + simulasyon) yerlesik
    deterministik hat ile yapilir. LLM'in urettigi bozuk kod asla dogrudan
    calistirilmaz.

Kurulum:
    pip install "crewai>=0.80"     # agir bagimliliklar icerir
    export OPENAI_API_KEY=...      # veya ANTHROPIC_API_KEY

Anahtar yoksa ya da paket kurulu degilse backend sessizce native akisa duser.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from agency.exceptions import AgencyError


class CrewUnavailableError(AgencyError):
    """crewai paketi veya LLM anahtari yok."""

    phase = "backend"


AGENT_ROLES = (
    {
        "name": "Arastirmaci",
        "role": "Kripto piyasa arastirmacisi",
        "goal": (
            "Piyasa rejimini, basarili tarihsel anlari ve haber akisini birlestirip "
            "aday stratejilerin zayif noktalarini belirle."
        ),
        "backstory": "Kripto piyasalarinda kantitatif arastirma ve rejim analizi uzmani.",
    },
    {
        "name": "Stratejist",
        "role": "Algoritmik strateji mimari",
        "goal": (
            "Aday stratejilerin giris/cikis mantigini, indikator secimlerini ve risk "
            "parametrelerini elestirel olarak degerlendir."
        ),
        "backstory": "Yuksek frekans ve trend takip sistemleri gelistirmis bir mimar.",
    },
    {
        "name": "BacktestMuhendisi",
        "role": "Backtest ve istatistik muhendisi",
        "goal": (
            "Metrikleri (net kar, win rate, drawdown, sharpe) yorumla; overfit ve veri "
            "sizintisi risklerini isaretle."
        ),
        "backstory": "Binicelerce simulasyon kosusu kalibre etmis bir test muhendisi.",
    },
    {
        "name": "RiskYoneticisi",
        "role": "Portfoy risk yoneticisi",
        "goal": (
            "Nihai karari ver: terfi mi, yeniden calisma mi, arsiv mi? Somut parametre "
            "direktifleri yaz."
        ),
        "backstory": "Sermaye koruma odakli, disiplinli bir risk yoneticisi.",
    },
)


def crewai_available() -> bool:
    try:
        import crewai  # noqa: F401

        return True
    except Exception:  # pragma: no cover - kurulum eksikligi
        return False


def build_crew(orchestrator: Any) -> Any:
    """CrewAI konseyini olusturur."""
    if not crewai_available():
        raise CrewUnavailableError("crewai kurulu degil. Kurulum: pip install 'crewai>=0.80'")
    if not orchestrator.llm.available:
        raise CrewUnavailableError(
            "CrewAI icin LLM anahtari gerekli (OPENAI_API_KEY / ANTHROPIC_API_KEY). "
            "Anahtar yoksa native backend kullanilir."
        )

    from crewai import Agent, Crew, Process, Task  # yerel import

    model = orchestrator.settings.llm.model
    agents = [
        Agent(
            role=spec["role"],
            goal=spec["goal"],
            backstory=spec["backstory"],
            verbose=False,
            allow_delegation=False,
            llm=model,
        )
        for spec in AGENT_ROLES
    ]

    tasks = [
        Task(
            description=(
                "Piyasa ozeti: {market_summary}\n"
                "Aday stratejiler: {candidates}\n"
                "Gecmis dersler: {lessons}\n"
                "Hedefler: {targets}\n\n"
                "Rejim uyumunu ve adaylarin zayif noktalarini 5 madde ile ozetle."
            ),
            expected_output="Maddeler halinde rejim ve aday degerlendirmesi.",
            agent=agents[0],
        ),
        Task(
            description=(
                "Aday stratejilerin indikator ve kural secimlerini elestir; hangi aday "
                "hangi rejimde daha iyi calisir, belirt."
            ),
            expected_output="Aday bazli guclu/zayif yon listesi.",
            agent=agents[1],
        ),
        Task(
            description=(
                "Backtest metriklerini ve onceki dersleri yorumla: {metrics}\n"
                "Overfit, asiri islem, stop mesafesi ve cikis kurgusu risklerini degerlendir."
            ),
            expected_output="Metrik yorumu ve risk uyarilari.",
            agent=agents[2],
        ),
        Task(
            description=(
                "Tum degerlendirmeleri birlestirerek nihai kararini ver: promote / rework / "
                "archive ve en fazla 4 somut parametre direktifi. Cikti formati: "
                "'VERDICT: ...' satiri ve ardindan 'DIREKTIF:' maddeleri."
            ),
            expected_output="VERDICT satiri + DIREKTIF maddeleri.",
            agent=agents[3],
        ),
    ]

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        memory=False,
    )


def make_crew_executor(orchestrator: Any, crew: Any) -> Any:
    """Konseyi calistirip ardindan deterministik degerlendirmeyi yapar."""

    def execute(*, cycle: int, research: Any, frame: Any, plan_ctx: Any) -> Dict[str, Any]:
        candidates = [
            {
                "name": spec.name,
                "family": spec.family,
                "thesis": spec.thesis,
                "indicators": [f"{i.kind}:{i.resolved_params()}" for i in spec.indicators],
                "stop_loss_atr": spec.stop_loss_atr,
                "take_profit_atr": spec.take_profit_atr,
            }
            for spec in research.candidates
        ]
        inputs = {
            "market_summary": research.market.regime_summary,
            "candidates": json.dumps(candidates, ensure_ascii=False)[:6000],
            "lessons": "\n".join(research.insights[:8]) or "-",
            "targets": json.dumps(orchestrator.critic_agent.target_dict(), ensure_ascii=False),
            "metrics": "Backtest henuz kosulmadi; onceki dongu ozetleri lesson tablosunda.",
        }
        try:
            result = crew.kickoff(inputs=inputs)
            text = getattr(result, "raw", None) or str(result)
            orchestrator.ctx.state["crew_advice"] = str(text)[:4000]
            orchestrator.logger.info(
                "crewai_advice",
                f"[crewai] konsey tamamlandi ({len(str(text))} karakter)",
                data={"head": str(text)[:400]},
            )
        except Exception as exc:  # pragma: no cover - LLM/ag hatasi
            orchestrator.logger.warning(
                "crewai_failed",
                f"[crewai] konsey calistirilamadi: {exc}; deterministik hatta devam",
            )
        return orchestrator._native_evaluation_phase(
            cycle=cycle, research=research, frame=frame, plan_ctx=plan_ctx
        )

    return execute


def attach_crewai_backend(orchestrator: Any) -> bool:
    """CrewAI backend'ini baglar; kurulamazsa native akisa duser."""
    try:
        crew = build_crew(orchestrator)
    except Exception as exc:
        orchestrator.logger.warning(
            "backend_fallback", f"CrewAI backend kullanilamadi ({exc}); native akisa dusuluyor"
        )
        orchestrator.cycle_executor = None
        return False
    orchestrator.cycle_executor = make_crew_executor(orchestrator, crew)
    orchestrator.logger.info(
        "backend_selected",
        "CrewAI backend aktif (danisma konseyi + deterministik uygulama hatti)",
        data={"agents": [spec["name"] for spec in AGENT_ROLES]},
    )
    return True

