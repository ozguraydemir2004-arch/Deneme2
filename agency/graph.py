"""LangGraph tabanli orkestrasyon backend'i.

Graf:

    coder -> backtest -> critic -> (rework ise coder | sonraki aday ise coder | END)

Bu backend, ``AgencyOrchestrator`` icindeki ayni ajan primitiflerini
(``_start_candidate``, ``_test_and_review``, ``coder_agent.repair``) kullanir;
boylece native akis ile ayni sonuclari uretir, sadece kontrol akisi
LangGraph state machine'e devredilir.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from agency.exceptions import AgencyError


class GraphUnavailableError(AgencyError):
    """langgraph paketi kurulu degil."""

    phase = "backend"


class CycleState(TypedDict, total=False):
    cycle: int
    research: Any
    frame: Any
    plan_ctx: Any
    candidates: List[Any]
    index: int
    attempt: int
    build: Any
    report: Any
    decision: Any
    archive: Any
    evaluations: List[Dict[str, Any]]
    versions: List[str]
    promoted: List[str]
    archived: List[str]
    best_score: float
    best_version: Optional[str]
    route: str


def langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401
        from langgraph.graph import StateGraph  # noqa: F401

        return True
    except Exception:  # pragma: no cover - kurulum eksikligi
        return False


def build_cycle_graph(orchestrator: Any) -> Any:
    """Dongu icin derlenmis LangGraph uygulamasi dondurur."""
    if not langgraph_available():
        raise GraphUnavailableError(
            "langgraph kurulu degil. Kurulum: pip install 'langgraph>=0.2'"
        )
    from langgraph.graph import END, StateGraph

    max_attempts = int(orchestrator.settings.targets.max_rework_attempts)

    def coder_node(state: CycleState) -> Dict[str, Any]:
        cycle = int(state["cycle"])
        plan_ctx = state["plan_ctx"]
        decision = state.get("decision")
        build = state.get("build")
        attempt = int(state.get("attempt", 0))
        market_summary = state["research"].market.regime_summary

        if decision is not None and decision.verdict == "rework" and attempt < max_attempts:
            attempt += 1
            orchestrator.logger.info(
                "graph_rework",
                f"[langgraph] {build.version} onarimi (deneme {attempt}/{max_attempts})",
                data={"directives": decision.directives[:3]},
            )
            new_build = orchestrator.coder_agent.repair(
                build,
                report=state["report"],
                decision=decision,
                cycle=cycle,
                attempt=attempt,
                market_summary=market_summary,
                directives=decision.directives,
                plan_ctx=plan_ctx,
            )
            return {"build": new_build, "attempt": attempt, "route": "rework"}

        index = int(state.get("index", -1)) + 1
        candidates = state["candidates"]
        candidate = candidates[index]
        new_build = orchestrator._start_candidate(
            candidate, cycle=cycle, market_summary=market_summary, plan_ctx=plan_ctx
        )
        orchestrator.logger.info(
            "graph_candidate",
            f"[langgraph] aday {index + 1}/{len(candidates)} -> {new_build.version}",
        )
        return {"build": new_build, "attempt": 0, "index": index, "route": "new_candidate"}

    def backtest_node(state: CycleState) -> Dict[str, Any]:
        report = orchestrator.backtest_agent.run_suite(
            state["build"], state["frame"], cycle=int(state["cycle"])
        )
        return {"report": report, "route": "backtested"}

    def critic_node(state: CycleState) -> Dict[str, Any]:
        build = state["build"]
        report = state["report"]
        attempt = int(state.get("attempt", 0))
        decision = orchestrator.critic_agent.review(
            build,
            report,
            cycle=int(state["cycle"]),
            market_summary=state["research"].market.regime_summary,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        archive = orchestrator.versioning.archive(build, report, decision)

        evaluations = list(state.get("evaluations") or [])
        evaluations.append(
            {"build": build, "report": report, "decision": decision, "archive": archive}
        )
        versions = list(state.get("versions") or []) + [build.version]
        promoted = list(state.get("promoted") or [])
        archived = list(state.get("archived") or [])
        if decision.verdict == "promote":
            promoted.append(build.version)
        elif decision.verdict != "rework":
            archived.append(build.version)

        best_score = float(state.get("best_score", -1.0))
        best_version = state.get("best_version")
        if decision.score > best_score:
            best_score = decision.score
            best_version = build.version

        return {
            "decision": decision,
            "archive": archive,
            "evaluations": evaluations,
            "versions": versions,
            "promoted": promoted,
            "archived": archived,
            "best_score": best_score,
            "best_version": best_version,
            "route": "reviewed",
        }

    def route_after_critic(state: CycleState) -> str:
        decision = state.get("decision")
        attempt = int(state.get("attempt", 0))
        index = int(state.get("index", 0))
        candidates = state.get("candidates") or []
        if decision is not None and decision.verdict == "rework" and attempt < max_attempts:
            return "coder"
        if index + 1 < len(candidates):
            return "coder"
        return "done"

    graph = StateGraph(CycleState)
    graph.add_node("coder", coder_node)
    graph.add_node("backtest", backtest_node)
    graph.add_node("critic", critic_node)
    graph.set_entry_point("coder")
    graph.add_edge("coder", "backtest")
    graph.add_edge("backtest", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"coder": "coder", "done": END})
    return graph.compile()


def make_langgraph_executor(orchestrator: Any) -> Any:
    """Orchestrator'a takilabilecek ``cycle_executor`` fonksiyonu uretir."""
    app = build_cycle_graph(orchestrator)
    max_attempts = int(orchestrator.settings.targets.max_rework_attempts)

    def execute(*, cycle: int, research: Any, frame: Any, plan_ctx: Any) -> Dict[str, Any]:
        initial: CycleState = {
            "cycle": int(cycle),
            "research": research,
            "frame": frame,
            "plan_ctx": plan_ctx,
            "candidates": list(research.candidates),
            "index": -1,
            "attempt": 0,
            "build": None,
            "report": None,
            "decision": None,
            "archive": None,
            "evaluations": [],
            "versions": [],
            "promoted": [],
            "archived": [],
            "best_score": -1.0,
            "best_version": None,
            "route": "start",
        }
        recursion_limit = max(
            50, 10 + 6 * len(research.candidates) * (max_attempts + 1)
        )
        final = app.invoke(initial, config={"recursion_limit": recursion_limit})
        return {
            "evaluations": final.get("evaluations", []),
            "versions": final.get("versions", []),
            "promoted": final.get("promoted", []),
            "archived": final.get("archived", []),
            "best_score": float(final.get("best_score", -1.0)),
            "best_version": final.get("best_version"),
        }

    return execute


def attach_langgraph_backend(orchestrator: Any) -> bool:
    """Backend'i orchestrator'a baglar; basarisizsa False doner."""
    try:
        orchestrator.cycle_executor = make_langgraph_executor(orchestrator)
    except Exception as exc:  # pragma: no cover - kurulum/calisma hatasi
        orchestrator.logger.warning(
            "backend_fallback",
            f"LangGraph backend kullanilamadi ({exc}); native akisa dusuluyor",
        )
        orchestrator.cycle_executor = None
        return False
    orchestrator.logger.info(
        "backend_selected", "LangGraph backend aktif (coder -> backtest -> critic dongusu)"
    )
    return True


