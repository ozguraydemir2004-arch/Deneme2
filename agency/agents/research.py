"""VERI ARASTIRMA VE VERI MADENCILIGI AJANI (Research Agent).

Dongu basina:
    1. BTC/USDT OHLCV verisini CCXT'den (yoksa sentetik) ceker.
    2. Indikator/ozellik matrisini ve piyasa rejimini hesaplar.
    3. ILERI GETIRIYE gore basarili (winner) anlari madencilikle bulur ve
       vektorlu snapshot olarak hafizaya yazar.
    4. Mevcut piyasa vektorune en benzeyen tarihsel basari anlarini getirir.
    5. Kripto haberlerini tarar (ag varsa), sentiment ozeti cikarir.
    6. Bilgi tabani + hafiza + dersler + LLM'i birlestirip strateji adaylari uretir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from agency.agents.base import BaseAgent
from agency.agents.news import NewsScanner
from agency.agents.planner import PlanContext
from agency.agents.prompts import RESEARCH_SYSTEM, research_prompt
from agency.data.mining import (
    build_market_context,
    compute_features,
    feature_dict,
    mine_snapshots,
    summarize_snapshot,
)
from agency.exceptions import DataError
from agency.schemas import ResearchReport, StrategySpec
from agency.utils import iso_now


class ResearchAgent(BaseAgent):
    """Piyasa verisi + haber + hafiza -> strateji adaylari."""

    name = "research"
    role = "Veri arastirma ve veri madenciligi"

    # ------------------------------------------------------------------ run
    def run(self, cycle: int) -> ResearchReport:
        self.info("research_start", f"Dongu {cycle}: veri toplama basladi")
        data = self._load_data()
        frame = data.frame
        self.ctx.state["last_frame"] = frame
        self.ctx.state["last_data_source"] = data.source
        features = compute_features(frame)
        order_book = self._order_book()

        market = build_market_context(
            frame,
            features,
            symbol=data.symbol,
            timeframe=data.timeframe,
            source=data.source,
            order_book=order_book,
        )
        market.fetched_at = iso_now()
        self.info(
            "market_context",
            market.regime_summary,
            data_source=data.source,
            bars=data.bars,
            warnings=data.warnings,
        )

        snapshots = self._mine_snapshots(cycle, data.symbol, data.timeframe, frame, features)
        matches = self._similar_winners(cycle, features)
        news_items = self._scan_news(cycle)
        playbook = self._load_playbook()
        lessons = [row.get("message", "") for row in self.memory.recent_lessons(limit=10)]
        directives = self.memory.pending_directives(limit=8)
        elite = self.memory.memory_strategies(limit=3, regime=market.regime)

        plan_ctx = PlanContext(
            cycle=cycle,
            timeframe=data.timeframe,
            regime=market.regime,
            atr_pct=market.atr_pct,
            rsi=market.rsi,
            adx=market.adx,
            winner_features=[m.get("features") or {} for m in matches],
            lessons=lessons,
            directives=directives,
            elite_specs=[
                {**item.get("spec", {}), "_version": item.get("version")}
                for item in elite
                if item.get("spec")
            ],
            avoid_fingerprints=self._recent_fingerprints(),
        )

        candidates = self._generate_candidates(
            plan_ctx, market, matches, directives, news_items, playbook
        )
        insights = self._build_insights(
            market, matches, news_items, snapshots, data.warnings, order_book
        )

        report = ResearchReport(
            cycle=cycle,
            market=market,
            candidates=candidates,
            news_digest=[
                {"title": n["title"], "sentiment": n["sentiment"], "keywords": n["keywords"]}
                for n in news_items
            ],
            snapshot_count=len(snapshots),
            winner_count=sum(1 for s in snapshots if s.get("label") == "winner"),
            top_matches=[
                {
                    "ts": m.get("ts"),
                    "similarity": m.get("similarity"),
                    "forward_return_pct": m.get("forward_return_pct"),
                    "label": m.get("label"),
                    "features": m.get("features"),
                }
                for m in matches[:5]
            ],
            insights=insights,
            playbook_refs=playbook.get("refs", []),
            generated_at=iso_now(),
            data_source=data.source,
            bars_used=data.bars,
        )
        self.memory.save_research(report)
        self.info(
            "research_done",
            f"{len(candidates)} aday uretildi | snapshot={len(snapshots)} | winner={report.winner_count}",
            candidates=[c.name for c in candidates],
            source=data.source,
        )
        return report

    # --------------------------------------------------------------- data
    def _load_data(self):
        try:
            return self.ctx.data_provider.fetch_ohlcv()
        except DataError as exc:
            self.error("data_error", f"Veri cekilemedi: {exc}")
            raise
        except Exception as exc:  # pragma: no cover - savunma
            self.error("data_error", f"Beklenmeyen veri hatasi: {exc}")
            raise DataError(str(exc)) from exc

    def _order_book(self) -> Dict[str, Any]:
        try:
            summary = self.ctx.data_provider.fetch_order_book_summary()
            if summary.get("source") == "ccxt":
                self.info(
                    "order_book",
                    "Emir defteri dengesizligi alindi",
                    imbalance=summary.get("imbalance"),
                    spread_bps=summary.get("spread_bps"),
                )
            return summary
        except Exception as exc:  # pragma: no cover
            self.warn("order_book_error", f"Emir defteri alinamadi: {exc}")
            return {}


    def _mine_snapshots(
        self, cycle: int, symbol: str, timeframe: str, frame, features
    ) -> List[Dict[str, Any]]:
        memory_cfg = self.settings.memory
        snapshots = mine_snapshots(
            frame,
            features,
            horizon=memory_cfg.winner_horizon_bars,
            winner_threshold_pct=memory_cfg.winner_forward_return_pct,
            max_snapshots=max(10, memory_cfg.max_snapshots_per_cycle // 2),
            vectorizer=self.memory.vectorizer,
        )
        if not snapshots:
            self.warn("no_snapshots", "Basarili an bulunamadi; esik dusuruluyor")
            snapshots = mine_snapshots(
                frame,
                features,
                horizon=memory_cfg.winner_horizon_bars,
                winner_threshold_pct=max(0.5, memory_cfg.winner_forward_return_pct / 2),
                max_snapshots=50,
                vectorizer=self.memory.vectorizer,
            )
        saved = self.memory.save_snapshots(cycle, symbol, timeframe, snapshots)
        winners = sum(1 for s in snapshots if s.get("label") == "winner")
        self.info(
            "snapshots_saved",
            f"{saved} snapshot kaydedildi ({winners} winner)",
            threshold_pct=memory_cfg.winner_forward_return_pct,
            horizon=memory_cfg.winner_horizon_bars,
        )
        if snapshots:
            self.log.debug("snapshot_example", summarize_snapshot(snapshots[0]))
        return snapshots

    def _similar_winners(self, cycle: int, features) -> List[Dict[str, Any]]:
        current = feature_dict(features.iloc[-1])
        if not current:
            return []
        vector = self.memory.vectorizer.embed_dict(current)
        matches = self.memory.similar_snapshots(
            vector,
            top_k=self.settings.memory.top_k_snapshots,
            label="winner",
            exclude_cycle=cycle,
        )
        self.info(
            "vector_search",
            f"{len(matches)} benzer basarili an bulundu",
            best_similarity=matches[0]["similarity"] if matches else None,
        )
        return matches

    # --------------------------------------------------------------- news
    def _scan_news(self, cycle: int) -> List[Dict[str, Any]]:
        cfg = self.settings.research
        scanner = NewsScanner(
            endpoint=cfg.news_endpoint,
            timeout=cfg.news_timeout_seconds,
            max_items=cfg.max_news_items,
            enabled=cfg.news_enabled,
        )
        items = scanner.fetch()
        if not items:
            self.warn(
                "news_unavailable",
                "Haber akisi alinamadi (ag kapali olabilir); haber sinyali atlandi",
                error=scanner.last_error,
            )
            return []
        self.memory.save_news(cycle, items)
        bullish = sum(1 for i in items if i["sentiment"] == "bullish")
        bearish = sum(1 for i in items if i["sentiment"] == "bearish")
        self.info(
            "news_scanned",
            f"{len(items)} haber tarandi",
            bullish=bullish,
            bearish=bearish,
            neutral=len(items) - bullish - bearish,
        )
        return items

    def _load_playbook(self) -> Dict[str, Any]:
        path = Path(self.settings.research.playbook)
        if not path.is_file():
            self.warn("playbook_missing", f"Bilgi tabani bulunamadi: {path}")
            return {"ideas": [], "refs": []}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self.warn("playbook_error", f"Bilgi tabani okunamadi: {exc}")
            return {"ideas": [], "refs": []}
        ideas = [
            f"{item.get('name')}: {item.get('idea')}"
            for item in payload.get("patterns", [])
            if item.get("idea")
        ]
        refs = [str(item.get("name")) for item in payload.get("patterns", []) if item.get("name")]
        return {"ideas": ideas, "refs": refs, "raw": payload}

    def _recent_fingerprints(self) -> List[str]:
        fingerprints: List[str] = []
        for row in self.memory.list_strategies(limit=15):
            detail = self.memory.get_strategy(row["version"])
            if detail:
                try:
                    fingerprints.append(StrategySpec.from_dict(detail["spec"]).fingerprint())
                except Exception:
                    continue
        return fingerprints


    # --------------------------------------------------------- candidates
    def _generate_candidates(
        self,
        plan_ctx: PlanContext,
        market,
        matches: List[Mapping[str, Any]],
        directives: List[str],
        news_items: List[Mapping[str, Any]],
        playbook: Mapping[str, Any],
    ) -> List[StrategySpec]:
        count = int(self.settings.targets.candidates_per_cycle)
        candidates: List[StrategySpec] = []
        if self.llm.available:
            prompt = research_prompt(
                market_summary=market.regime_summary,
                targets=self.settings.targets.__dict__,
                winner_matches=matches,
                lessons=plan_ctx.lessons,
                directives=directives,
                news_headlines=[n.get("title", "") for n in news_items],
                playbook_ideas=playbook.get("ideas", []),
                count=count,
            )
            payload = self.ask_json(RESEARCH_SYSTEM, prompt, label="research_ideas")
            candidates = self.planner.from_llm_ideas(payload, plan_ctx)
            self.info(
                "llm_candidates",
                f"LLM'den {len(candidates)} gecerli aday alindi",
                names=[c.name for c in candidates],
            )
        else:
            self.info(
                "offline_planner",
                "LLM yok: rejim + winner snapshot temelli deterministik planlayici kullaniliyor",
            )

        seen = {c.fingerprint() for c in candidates}
        for spec in self.planner.plan(plan_ctx, count=count):
            if spec.fingerprint() in seen:
                continue
            seen.add(spec.fingerprint())
            candidates.append(spec)
            if len(candidates) >= count:
                break
        return candidates[:count]

    def _build_insights(
        self,
        market,
        matches: List[Mapping[str, Any]],
        news_items: List[Mapping[str, Any]],
        snapshots: List[Mapping[str, Any]],
        warnings: List[str],
        order_book: Mapping[str, Any],
    ) -> List[str]:
        insights = [market.regime_summary]
        winners = [s for s in snapshots if s.get("label") == "winner"]
        if winners:
            avg_ret = sum(float(s.get("forward_return_pct") or 0.0) for s in winners) / len(winners)
            insights.append(
                f"{len(winners)} basarili an bulundu; ortalama ileri getiri %{avg_ret:.2f}"
            )
        if matches:
            insights.append(
                f"En benzer tarihsel basari anina benzerlik: cosine={matches[0].get('similarity')}"
            )
        if news_items:
            sentiment: Dict[str, int] = {}
            for item in news_items:
                key = str(item.get("sentiment"))
                sentiment[key] = sentiment.get(key, 0) + 1
            insights.append(f"Haber sentiment dagilimi: {sentiment}")
        if order_book.get("imbalance"):
            insights.append(f"Emir defteri dengesizligi: {order_book.get('imbalance')}")
        insights.extend(warnings)
        return insights


