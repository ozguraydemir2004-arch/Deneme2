"""Ajan prompt'lari (tek merkezde).

Tum prompt'lar ayni JSON sozlesmesini kullanir; boylece LLM cevabi
dogrulanabilir ve derlenebilir kalir. LLM yoksa ajanlar bu prompt'lari
kullanmaz, deterministik planlayiciya duser.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from agency.data.indicators import catalog_for_prompt

SPEC_SCHEMA = """{
  "name": "kisa_snake_case_isim",
  "family": "trend_following | mean_reversion | breakout | momentum | volatility_compression | hybrid",
  "thesis": "stratejinin tek cumlelik gerekcesi",
  "timeframe": "1h",
  "indicators": [
    {"key": "ema_fast", "kind": "ema", "output": "value", "params": {"period": 21}}
  ],
  "entry_long":  [{"left": "ema_fast", "op": "cross_above", "right": "ema_slow"}],
  "exit_long":   [{"left": "ema_fast", "op": "cross_below", "right": "ema_slow"}],
  "filters":     [{"left": "adx_14", "op": ">", "right": 18}],
  "stop_loss_atr": 2.0,
  "take_profit_atr": 4.0,
  "trailing_stop_atr": 0.0,
  "risk_per_trade_pct": 1.5,
  "cooldown_bars": 0,
  "max_holding_bars": 0
}"""

RESEARCH_SYSTEM = f"""Sen bir kripto kantitatif arastirma ajanisin (Research Agent).
Gorevin: verilen piyasa rejimi, basarili tarihsel anlar (winner snapshot'lar),
haber ozeti ve onceki dongu derslerini birlestirerek YENI, test edilebilir
strateji fikirleri uretmek.

ZORUNLU KURALLAR
- Yalnizca asagidaki indikator katalogundaki indikatorleri kullan.
- Kurallarda operand olarak yalnizca tanimladigin indikator anahtarlari,
  open/high/low/close/volume veya sayisal sabit kullan.
- Her aday icin 2-6 indikator, 1-4 giris kurali, 0-3 cikis kurali, 0-3 filtre.
- Ayni aileden birden fazla aday uretme; cesitlilik hedeflenir.
- Cikti SADECE su JSON semasinda olsun:

{{
  "insights": ["gozlem 1", "gozlem 2"],
  "strategy_ideas": [ {SPEC_SCHEMA} ]
}}

INDIKATOR KATALOGU
{catalog_for_prompt()}
"""

CODER_SYSTEM = f"""Sen bir kidemli algoritmik ticaret yazilimcisisin (Coder Agent).
Gorevin: arastirma fikirlerini veya critic direktiflerini, derlenebilir ve
backtest edilebilir bir strateji SPEC'ine donusturmek.

ZORUNLU KURALLAR
- Cikti SADECE su JSON olsun: {{"strategy": {SPEC_SCHEMA}, "notes": ["kisa not"]}}
- Indikatorler katalogdan secilmeli; parametreler makul araliklarda olmali.
- Giris kosulu en az 1 kural; stop_loss_atr 0.5-6; take_profit_atr 0-12;
  risk_per_trade_pct 0.25-3.
- Direktiflerde istenen degisiklikleri (stop mesafesi, trend filtresi, risk
  dusurme vb.) somut parametre degisikliklerine cevir.
- Asla kod metni dondurme; yalnizca JSON spec.

INDIKATOR KATALOGU
{catalog_for_prompt()}
"""

CRITIC_SYSTEM = """Sen bir portfoy risk yoneticisi ve optimizasyon uzmanisin (Critic & Optimizer Agent).
Gorevin: backtest metriklerini hedef esiklerle karsilastirmak, basarisizligin
kok nedenini belirlemek ve Coder Agent'a uygulanabilir, SOMUT direktifler
vermek.

ZORUNLU KURALLAR
- Cikti SADECE su JSON olsun:
  {"verdict": "promote|rework|archive",
   "reasons": ["neden 1", "neden 2"],
   "directives": ["stop_loss_atr 2.0 -> 1.5", "adx filtresi 18 -> 22"]}
- Direktifler olculebilir olmali: hangi parametre, hangi yone, hangi deger.
- Ayni hatayi tekrar oneriyorsan farkli bir kaldiraç (filtre, zamanlama,
  cikis kurali) oner.
- Karar esiklerini kullanici hedefleri belirler; sen onlara uy.
"""

RESEARCH_JSON_KEYS = ("insights", "strategy_ideas")
CODER_JSON_KEYS = ("strategy", "notes")
CRITIC_JSON_KEYS = ("verdict", "reasons", "directives")


def _bullets(items: Sequence[str], limit: int = 12) -> str:
    if not items:
        return "- (kayit yok)"
    return "\n".join(f"- {item}" for item in list(items)[:limit])


def _compact_features(features: Mapping[str, Any]) -> str:
    if not features:
        return "(ozellik yok)"
    keys = ("rsi", "adx", "atr_pct", "bb_width", "bb_pctb", "rel_volume", "trend_score")
    parts = [f"{k}={features[k]}" for k in keys if k in features]
    return ", ".join(parts) if parts else str(dict(list(features.items())[:6]))


def research_prompt(
    *,
    market_summary: str,
    targets: Mapping[str, float],
    winner_matches: Sequence[Mapping[str, Any]],
    lessons: Sequence[str],
    directives: Sequence[str],
    news_headlines: Sequence[str],
    playbook_ideas: Sequence[str],
    count: int,
) -> str:
    """Research Agent icin kullanici prompt'u."""
    matches_text = "\n".join(
        f"- benzerlik={m.get('similarity')} fwd={m.get('forward_return_pct')}% "
        f"({_compact_features(m.get('features') or {})})"
        for m in winner_matches[:6]
    ) or "- benzer basarili an bulunamadi (ilk dongu olabilir)"
    return f"""PIYASA DURUMU
{market_summary}

HEDEFLER
- min net kar: %{targets['min_net_profit_pct']}
- min basari orani: %{targets['min_win_rate']}
- maks drawdown: %{targets['max_drawdown_pct']}
- min sharpe: {targets['min_sharpe']}

GECMIS BASARILI ANLAR (vektor benzerligi)
{matches_text}

ONCEKI DONGU DERSLERI
{_bullets(lessons)}

AKTIF DIREKTIFLER
{_bullets(directives)}

HABER BASLIKLARI
{_bullets(news_headlines, 8)}

BILGI TABANI FIKSLERI (kanitlanmis kaliplar)
{_bullets(playbook_ideas, 10)}

GOREV: Yukaridaki verilere dayanarak {count} adet FARKLI strateji adayi uret.
JSON semasina uy; her adayin gerekcesini (thesis) veriye dayandir.
"""


def coder_prompt(
    *,
    market_summary: str,
    idea: Mapping[str, Any] | None,
    directives: Sequence[str],
    repair_context: Mapping[str, Any] | None,
    elite_specs: Sequence[Mapping[str, Any]],
    attempt: int,
) -> str:
    """Coder Agent icin kullanici prompt'u."""
    idea_text = (
        f"Adi: {idea.get('name')}\nAile: {idea.get('family')}\nTez: {idea.get('thesis')}\n"
        f"Taslak: {idea}"
        if idea
        else "(yeni fikir yok; arastirma cikisina gore kendi fikrini kur)"
    )
    repair_text = (
        "(ilk deneme)"
        if not repair_context
        else (
            f"Metrikler: {repair_context.get('metrics')}\n"
            f"Hatalar: {repair_context.get('errors')}\n"
            f"Karar: {repair_context.get('verdict')}\n"
            f"Deneme: {attempt}"
        )
    )
    elite_text = _bullets([f"{s.get('name')} ({s.get('family')})" for s in elite_specs], 5)
    return f"""PIYASA
{market_summary}

ARASTIRMA FIKRI
{idea_text}

UYGULANACAK DIREKTIFLER
{_bullets(directives)}

ONCEKI DENEME / HATA BAGLAMI
{repair_text}

KALICI HAFIZADAKI BASARILI STRATEJILER (ilham)
{elite_text}

GOREV: Bu bilgilerle, hedefleri kovalayan tek bir strateji SPEC'i uret.
Ozellikle son denemede patlayan noktalari (asiri islem, stop yakinligi, trend
filtresi eksikligi) duzelt. Sadece JSON dondur.
"""


def critic_prompt(
    *,
    market_summary: str,
    metrics: Mapping[str, Any],
    oos_metrics: Mapping[str, Any] | None,
    gates: Mapping[str, bool],
    targets: Mapping[str, float],
    trades_summary: str,
    errors: Sequence[str],
    attempt: int,
    max_attempts: int,
) -> str:
    """Critic Agent icin kullanici prompt'u."""
    gate_text = ", ".join(f"{k}={'GECTI' if v else 'KALDI'}" for k, v in gates.items())
    return f"""PIYASA
{market_summary}

TEST METRIKLERI
- net kar %: {metrics.get('net_profit_pct')}
- basari orani %: {metrics.get('win_rate')}
- maks drawdown %: {metrics.get('max_drawdown_pct')}
- sharpe: {metrics.get('sharpe')}
- profit factor: {metrics.get('profit_factor')}
- islem sayisi: {metrics.get('trades')}
- ortalama islem %: {metrics.get('avg_trade_pct')}
- al-tut getirisi %: {metrics.get('buy_hold_pct')}
- out-of-sample: {oos_metrics or 'yok'}

ESIK KONTROLU
{gate_text}

HEDEFLER
{dict(targets)}

ISLEM OZETI
{trades_summary}

HATA MATRISI
{_bullets(errors, 6)}

DENEME: {attempt}/{max_attempts}
GOREV: Karar ver ve somut direktifler yaz. Sadece JSON dondur.
"""


def idea_to_dict(spec: Any) -> Dict[str, Any]:
    return spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)


def summarize_specs(specs: Sequence[Any], limit: int = 5) -> List[str]:
    lines: List[str] = []
    for spec in list(specs)[:limit]:
        try:
            lines.append(spec.describe())
        except AttributeError:
            lines.append(str(spec))
    return lines

