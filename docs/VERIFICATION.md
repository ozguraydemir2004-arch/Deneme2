# Dogrulama Kaydi (Verification)

Bu dosya, sistemin calistigina dair **gozlemlenen kanitlari** ozetler.
Komutlar bu depoda oldugu gibi calistirilabilir (ag erisimi degiskenlik
gosterebilir).

## 1. Statik kontrol

```bash
$ python3 -m pyflakes agency main.py tests
$ echo $?
0
```

## 2. Test paketi

```bash
$ python -m pytest
........................................................................ [ 75%]
........................                                                 [100%]
119 passed in 13.82s
```

Kapsam ozeti:

| Alan | Test dosyasi | Kritik iddia |
|------|--------------|--------------|
| Indikatorler | `tests/test_indicators.py` | Donchian kanali mevcut bari icermez (look-ahead regresyonu) |
| Backtest | `tests/test_engine.py` | Trailing stop ayni barin high/low'u ile tetiklenmez; komisyon/TP/SL/zaman stopu |
| Skorlama | `tests/test_metrics.py` | Drawdown, sharpe, profit factor, gate ve skor siralamasi |
| Hafiza | `tests/test_memory.py` | Cycle/snapshot/vektor/terfi/ders dongusu |
| Derleyici | `tests/test_compiler.py` | Spec -> OOP kod; AST yasakli cagri/import kontrolu |
| Ajanlar | `tests/test_planner_coder.py`, `tests/test_critic.py` | Onarim uretir, kesif mutasyonu, terfi/rework/arsiv |
| Uctan uca | `tests/test_pipeline.py` | Native + LangGraph dongu; dosya, DB ve rapor dogrulamasi |
| Bakim | `tests/test_maintenance.py`, `tests/test_news_and_cache.py` | Decay check/demote, onbellek kaynak sadakati, haber sentiment |
| LLM | `tests/test_llm.py` | JSON ayiklama (string icindeki parantezler dahil), istemci secimi |

## 3. Gercek veriyle dongu (BTC/USDT 1h, 90 gun)

Veri zinciri: `binance` (engelli) -> `kraken` (721 bar) -> **CoinGecko REST
(1500+ bar)**. Loglardan gozlemlenen:

```
market_context  BTC/USDT 1h rejim=squeeze/moderate vol=normal RSI=46.3 ADX=20.4
                data_source=coingecko bars=1500
                warnings=["binance kullanilamadi; kraken borsasindan veri alindi.",
                          "CCXT 721 bar dondurdu; CoinGecko 1500 bar ile gecmis genisletildi."]
...
review_done     v011 -> promote | skor=88.1 | gates=10/15
                reasons=["net kar %6.81 (esik %2.0) | win %66.7 (esik %40.0)",
                         "drawdown %3.1 (limit %20.0) | sharpe 4.13 | PF 4.63",
                         "Walk-forward dilimleri: %2.57, %3.39, %-0.72"]
cycle_done      DONGU 002 BITTI (completed) | en iyi=v011 skor=88.1 | terfi=v011
```

Kalici hafiza kaydi (`memory_strategies`):

```
{'version': 'v011', 'name': 'donchian_breakout_20', 'family': 'breakout',
 'score': 88.15, 'regime': 'squeeze', 'times_reused': 0}
```

Ayni kosuda uretilen artefaktlar:

```
runtime/reports/INDEX.md
runtime/reports/cycle_002/report.md
runtime/reports/cycle_002/v011_train.csv / _test.csv / _fold1..3.csv (+ job/result JSON)
runtime/archive/v011_donchian_breakout_20_score088.1_promote.py
agency/strategies/library/v011_donchian_breakout_20.py
```

Ornek uretilen strateji kodu (`v011`):

```python
def generate_signal(self, row, prev, position):
    atr_value = safe_float(row.get("atr"), 0.0)
    if position is None:
        if ((row["rel_volume"] > 1.2) and (row["adx_14"] > 17.5)) and \
           ((row["close"] > row["dc_upper"] and prev["close"] <= prev["dc_upper"])):
            return {"action": "buy", "reason": "giris kosullari saglandi ..."}
        return {"action": "hold", "reason": "giris kosulu saglanmadi"}
    if (row["close"] < row["dc_lower"] and prev["close"] >= prev["dc_lower"]):
        return {"action": "sell", "reason": "cikis kosulu saglandi"}
    return {"action": "hold", "reason": "pozisyon korunuyor"}
```

## 4. LangGraph backend'i (ayni dongu, graf uzerinden)

```bash
$ python main.py run --cycles 1 --backend langgraph
backend_selected   LangGraph backend aktif (coder -> backtest -> critic dongusu)
graph_candidate    [langgraph] aday 1/3 -> v015
graph_rework       [langgraph] v016 onarimi (deneme 1/3)
graph_candidate    [langgraph] aday 2/3 -> v016
...
cycle_done         DONGU 003 BITTI (completed) | en iyi=v021 skor=56.7
```

## 5. Reddetme (risk disiplini) davranisi

Ayni veride yuksek skor alan ancak **son (unseen) pencerede derin kayip**
yazan bir strateji terfi ettirilmedi:

```
v009 -> rework | skor=84.4
failed gates: ['oos_net_profit','oos_win_rate','oos_sharpe','oos_profit_factor','oos_not_catastrophic']
Out-of-sample net %-3.68 (limit > -1.5), Walk-forward: %2.35, %2.86, %-2.12
```

Bu, sistemin "yuksek skor = terfi" gibi naif bir kural kullanmadigini
gosterir: en guncel gorulmemis pencere ve walk-forward dilimleri birlikte
degerlendirilir.

## 6. Strateji bakimi (decay check)

```bash
$ python main.py validate --limit 3
Yeniden dogrulama: 1 terfi etmis strateji
------------------------------------------------------------------------------
 v020   bb_reversion_24_22_r1_p1   BASARISIZ  net= -6.48% win= 36.7% dd= 6.84% sharpe=-5.14 islem= 30
------------------------------------------------------------------------------
 Rapor: runtime/reports/REVALIDATION.md
```

Sentetik veride terfi eden strateji, gercek veri penceresinde esikleri
gecemedi ve raporlandi. `--demote` verildiginde kalici hafizadan cikarilir,
`status='demoted'` isaretlenir ve `lessons` tablosuna gerekce yazilir
(`tests/test_maintenance.py` bunu dogrular).

## 7. Notlar

- Veri penceresi zamanla kayar; ayni komut farkli dongude farkli sonuclar
  uretebilir (piyasa verisi degisir).
- `runtime/` git'e dahil edilmez; yukaridaki artefaktlar her kosuda yeniden
  uretilir.
- Sentetik mod (`--data-provider synthetic`) ag olmadan calisir ancak
  sonuclar **gercek piyasa performansi degildir**.
