# Kripto Al-Sat Botu Gelistirme Ajans Sistemi

Kendi kendini **surekli gelistiren**, 24/7 calisan coklu ajan (multi-agent) bir
algoritmik ticaret Ar-Ge sistemi. Dort uzman ajan bir dongu icinde calisir:

| # | Ajan | Sorumluluk |
|---|------|-----------|
| 1 | **Research Agent** | CCXT'den BTC/USDT verisi ceker, indikator/haber tarar, **basarili tarihsel anlari madencilikle** bulur ve vektorlu hafizaya yazar |
| 2 | **Coder Agent** | Arastirma ciktisini **OOP Python stratejisine** derler (versiyonlanir, AST ile dogrulanir) |
| 3 | **Backtest & Debug Agent** | Kodu **ayri bir surecte (sandbox)** gecmis veri uzerinde simule eder; net kar, win rate, max drawdown, sharpe, profit factor ve **Hata Matrisi** uretir |
| 4 | **Critic & Optimizer Agent** | Metrikleri esiklerle karsilastirir, skorlar, **terfi / onarim / arsiv** karari verir, dersleri ve direktifleri hafizaya yazar |

Dongu biter -> dersler hafizaya yazilir -> bir sonraki dongu bu derslerle
baslar. Basarili stratejiler **kalici hafizaya** alinir ve ayni piyasa rejimi
tekrar olustugunda tohum olarak kullanilir.

---

## 1. Mimari

```
                        +-----------------------------+
                        |      ORCHESTRATOR           |
                        |  (24/7 dongu, state, rapor) |
                        +--------------+--------------+
                                       |
        +------------------------------+------------------------------+
        |                              |                              |
+-------v--------+          +----------v---------+          +---------v-------+
| RESEARCH       |          | CODER              |          | BACKTEST        |
| - CCXT OHLCV   |  spec    | - spec -> OOP kod  |  build   | - sandbox run   |
| - indikator    |--------->| - AST dogrulama    |--------->| - metrikler     |
| - winner mine  |          | - versiyonlama     |          | - hata matrisi  |
| - haber/vektor |          +----------^---------+          +---------+-------+
+-------+--------+                     |                              |
        |                              | direktif / onarim            | rapor
        |                    +---------+------------------------------v-------+
        +------------------->| CRITIC & OPTIMIZER                             |
             dersler         | - gate tablosu, 0-100 skor, walk-forward        |
                             | - terfi -> KALICI HAFIZA, aksi halde rework     |
                             +----------------+-------------------------------+
                                              |
                       +----------------------v----------------------+
                       | SQLite: cycles, snapshots(vektor), versions, |
                       | backtests, reviews, memory_strategies, lessons|
                       +-----------------------------------------------+
```

### Orkestrasyon backend'leri

| Backend | Aciklama | Dogrulama |
|---------|----------|-----------|
| `native` | Bagimliliksiz yerlesik akis (varsayilan). Her ortamda calisir. | ✅ uctan uca test |
| `langgraph` | Ayni ajan primitifleri **LangGraph state machine** ile yonetilir: `coder -> backtest -> critic -> (rework ? coder : END)` | ✅ uctan uca test |
| `crewai` | CrewAI **danisma konseyi** (Arastirmaci/Stratejist/Backtest/Risk). Kod uretimi ve backtest deterministik kalir (LLM'in yazdigi bozuk kod asla dogrudan calismaz). Anahtar/paket yoksa native'e duser. | opsiyonel |

`backend` secimi `config/settings.yaml -> runtime.backend` veya
`python main.py run --backend langgraph` ile yapilir.

---

## 2. Dosya Yapisi

```
.
├── main.py                     # CLI (run / once / status / doctor)
├── config/settings.yaml        # Tum parametreler (env ile override)
├── requirements.txt
├── agency/
│   ├── config.py               # Dataclass tabanli ayar yukleyici
│   ├── schemas.py              # Ajanlar arasi veri sozlesmeleri
│   ├── exceptions.py           # Yapilandirilmis hata tipleri
│   ├── utils.py                # Atomik yazma, slug, zaman yardimcilari
│   ├── logging_utils.py        # JSONL + insan-okur cift kanal log
│   ├── llm.py                  # OpenAI/Anthropic/Offline istemcileri
│   ├── orchestrator.py         # 24/7 dongu, state, raporlama
│   ├── graph.py                # LangGraph backend'i
│   ├── crew.py                 # CrewAI backend'i (opsiyonel)
│   ├── versioning.py           # Arsiv/puanlama yoneticisi
│   ├── reporting.py            # Markdown rapor + INDEX panosu
│   ├── agents/
│   │   ├── base.py             # AgentContext + BaseAgent
│   │   ├── research.py         # 1) Research Agent
│   │   ├── coder.py            # 2) Coder Agent
│   │   ├── backtester.py       # 3) Backtest & Debug Agent
│   │   ├── critic.py           # 4) Critic & Optimizer Agent
│   │   ├── planner.py          # Deterministik spec uretimi/onarimi
│   │   ├── prompts.py          # Tum ajan prompt'lari (JSON sozlesmeli)
│   │   └── news.py             # Haber tarayici + sentiment
│   ├── data/
│   │   ├── market_data.py      # CCXT + borsa fallback + CoinGecko + sentetik
│   │   ├── indicators.py       # Indikatorler + INDICATOR_KINDS katalogu
│   │   └── mining.py           # Ozellik cikarimi, rejim, winner snapshot
│   ├── strategies/
│   │   ├── base.py             # BaseStrategy (OOP sozlesmesi)
│   │   ├── compiler.py         # StrategySpec -> Python kodu
│   │   ├── registry.py         # Uretim/versiyon/yukleme/arsiv
│   │   └── library/            # Uretilen vNNN_*.py dosyalari (runtime)
│   ├── backtest/
│   │   ├── engine.py           # Bar bazli simulator
│   │   ├── metrics.py          # Metrik + skor + gate
│   │   ├── runner.py           # Sandbox calistirici (ayri surec)
│   │   └── validator.py        # AST guvenlik/dogrulama
│   ├── memory/
│   │   ├── database.py         # SQLite semasi + baglanti
│   │   ├── repositories.py     # MemoryStore (tum sorgular)
│   │   └── vector_store.py     # Hash-tabanli vektor hafiza
│   └── knowledge/
│       └── strategy_playbook.json  # Kanitlanmis kaliplar (bilgi tabani)
├── tests/                      # 119 test (unit + uctan uca)
└── runtime/                    # Calisma zamani (git'e girmez)
    ├── agency.db               # SQLite hafiza
    ├── logs/                   # agency-YYYY-MM-DD.log/.jsonl
    ├── reports/cycle_NNN/      # report.md + backtest artefaktlari
    ├── reports/INDEX.md        # Kontrol panosu
    ├── archive/                # Puanli strateji kopyalari
    └── state.json              # Dongu durumu (resume)
```

---

## 3. Hizli Baslangic

```bash
# 1) Bagimliliklar (cekirdek: pandas, numpy, ccxt, pyyaml)
pip install -r requirements.txt
#    Opsiyonel LangGraph backend'i icin
pip install "langgraph>=0.2" "langchain-core>=0.3"

# 2) Ayarlar (opsiyonel LLM anahtari)
cp .env.example .env
#    Anahtar yoksa sistem "offline" deterministik beyin ile calisir.

# 3) Ortam kontrolu
python main.py doctor

# 4) Tek dongu (veri: CCXT -> borsa fallback -> CoinGecko -> sentetik)
python main.py once

# 5) Iki dongu, aralarinda 30 sn
python main.py run --cycles 2 --sleep 30

# 6) 24/7 mod (max_cycles_per_run: 0 => sinirsiz; Ctrl+C guvenli cikis)
python main.py run

# 7) Durum / pano
python main.py status
cat runtime/reports/INDEX.md
```

---

## 4. Bir Donguda Ne Olur?

1. **Research**
   - OHLCV cekilir (`data.limit` = 2160 bar; CCXT borsasi 720 ile sinirliysa
     otomatik olarak CoinGecko REST ile 90 gunluk gecmis genisletilir).
   - Ozellik matrisi (RSI/ADX/ATR%/BB/volume/trend skoru) ve **piyasa rejimi**
     (trend_up, range, squeeze, volatile, ...) hesaplanir.
   - `horizon` barlik **ileri getiri** esigini asan anlar `winner`, ters yonde
     asanlar `loser` etiketiyle snapshot'lanir; her snapshot 256 boyutlu
     vektoruyle SQLite'a yazilir.
   - Mevcut piyasa vektorune en yakin **tarihsel basari anlari** getirilir.
   - Haberler taranir (anahtar kelime tabanli sentiment), bilgi tabani ve
     gecmis dersler okunur; LLM varsa adaylar LLM'den, yoksa deterministik
     planlayicidan uretilir.
2. **Coder**: her aday icin `StrategySpec` -> derlenmis OOP strateji
   (`vNNN_isim.py`), AST guvenlik dogrulamasi, DB'ye `draft` kaydi.
3. **Backtest**: train / out-of-sample / 3 dilimlik walk-forward kosulari
   **ayri sureclerde** (zaman + bellek limitli) calisir; metrikler ve hata
   matrisi kaydedilir.
4. **Critic**: gate tablosu (net kar, win rate, drawdown, sharpe, PF, islem
   sayisi), 0-100 skor, walk-forward onayi; karar:
   - `promote` -> `promoted`, kalici hafizaya eklenir,
   - `rework` -> coder'a **olculebilir direktifler** (orn.
     `stop_loss_atr 2.0 -> 1.5`), yeni surum uretilir,
   - `archive` -> `rejected`, ders kaydi birakilir.
5. **Dongu kapanisi**: `report.md` + `INDEX.md`, state dosyasi, DB kayitlari,
   arsiv kopyasi (`..._scoreXX.X_verdict.py`).

---

## 5. Neden "Spec -> Kod"?

LLM'e ham kod yazdirmak yerine **bildirimsel spec** urettiriyoruz:

- **Guvenlik**: serbest kod dosya/sistem/ag erisimi deneyebilir. Burada kod
  her zaman deterministik derleyiciden gecer ve AST dogrulayicisindan (import
  whitelist, yasakli cagrilar, `while` yasagi, `BaseStrategy` sozlesmesi)
  gecmeden diske yazilmaz.
- **Tekrarlanabilirlik**: ayni spec -> ayni kod.
- **Onarim kolayligi**: dusuk skor -> parametre duzeyinde direktif -> yeni
  surum.
- **LLM'siz calisma**: anahtar olmadan da sistem tam kapasite calisir.

---

## 6. Backtest Dogrulugu (look-ahead korumalari)

Testlerle korunan uc kritik tuzak:

1. **Donchian kanali** `shift(1)` ile hesaplanir (mevcut bar dahil degil).
   Aksi halde `close > kanal_ust` matematiksel olarak imkansizdir.
2. **Trailing stop** ayni barin high'i ile yukseltilip ayni barin low'u ile
   tetiklenmez; once o an gecerli stop kontrol edilir, sonra guncellenir.
3. **Sikisma filtresi** (`bb_width_prev`) onceki barda kontrol edilir; breakout
   barinda bant genisligi zaten acilmistir.

Ayrica: komisyon (7.5 bps) + kayma (2 bps), risk yuzdesine gore pozisyon
boyutlandirma (kaldiracsiz spot), ATR tabanli SL/TP/trailing, zaman stopu
(`max_holding_bars`), cooldown ve out-of-sample dogrulama uygulanir.

### Skorlama ve terfi politikasi

```
skor = 0.35*net_kar + 0.20*win_rate + 0.20*drawdown + 0.15*sharpe + 0.10*profit_factor
      (her bilesen hedefe gore 0-1 normalize; islem < min_trades ise x0.6)
```

- `promote`: tum cekirdek esikler + `score >= min_score` + **walk-forward
  onayi** (dilimlerin >= %60'i pozitif veya OOS pozitif; OOS'de yeterli islem
  varsa PF >= ~1.1).
- `rework`: skor alt esigin uzerindeyse ve deneme hakki varsa; coder yeni bir
  surum uretir (`parent_version` ile soy agaci).
- `archive`: deneme hakki bitti veya skor cok dusuk.
- `score >= elite_score` (varsayilan 70) olan terfi edilmis strateji "elite"
  sayilir ve sonraki dongulerde mutasyon tohumu olur.

---

## 7. Hafiza Modeli

| Tablo | Icerik |
|-------|--------|
| `cycles` | Dongu ozeti, en iyi surum, terfi listesi, direktifler, rapor yolu |
| `research_reports` | Piyasa baglami + adaylar + haber ozeti (JSON) |
| `market_snapshots` | Basarili/kayip anlar: ozellikler + **vektor** + ileri getiri |
| `strategy_versions` | `version`, `parent_version`, `code_path`, `spec_json`, `status`, `score` |
| `backtests` | Train/OOS/fold metrikleri + hata matrisi + sure |
| `reviews` | Critic karari (gerekce + direktifler) |
| `memory_strategies` | **Kalici hafiza**: terfi eden stratejiler, rejim etiketi, kullanim sayisi |
| `lessons` | Dersler ve cagrilacak direktifler (sonraki donguya tasinir) |

Vektor hafiza harici embedding servisi gerektirmez: ozellikler "signed hashing
trick" ile 256 boyuta izdusurulur, kosinus benzerligi ile en yakin tarihsel
basari anlari bulunur (`agency/memory/vector_store.py`).

### Strateji bakimi (decay check)

Terfi eden bir strateji, piyasa rejimi degistiginde bozulabilir. Bu yuzden
kalici hafiza tek yonlu degildir:

```bash
python main.py validate            # rapor uretir (runtime/reports/REVALIDATION.md)
python main.py validate --demote   # basarisizlari kalici hafizadan cikarir
```

- Yeniden dogrulama, promoted stratejileri **guncel veri** uzerinde kosar ve
  ayni gate tablosuyla degerlendirir (taze pencere kisa oldugu icin islem
  esigi yarilanir).
- Basarisiz strateji `--demote` ile `memory_strategies`'ten cikarilir,
  `status='demoted'` isaretlenir ve `lessons` tablosuna gerekce yazilir.
  Boylece hafiza yalnizca hala gecerli olan stratejileri tasir.
- Ciktilar: `runtime/reports/REVALIDATION.md` (tablo + gerekceler).

---

## 8. Veri Kaynaklari

1. **CCXT** (birincil): `exchange.id` (varsayilan `binance`), basarisizsa
   `data.fallback_exchanges` sirasi (varsayilan `kraken`, `coinbase`). Sembol
   borsada yoksa `BTC/USDT -> BTC/USD` alternatifi denenir. Borsalar genellikle
   720 bar ile sinirli oldugundan gecmis sayfalanarak toplanir.
2. **CoinGecko REST** (fallback/genisletme): anahtarsiz, 90 gunluk saatlik
   veri. OHLCV fiyat noktalarindan **muhafazakar** turetilir (wick yok).
3. **Sentetik**: ag yoksa deterministik, rejim degistiren GBM uretimi
   (test/demo). Bu modda sonuclar gercek piyasa performansi degildir.

Veri kaynagi her raporda ve DB'de acikca yazilir (`data_source`). Onbellek
(CSV) bir tasima bicimidir; `.meta.json` yan dosyasi sayesinde kaynak
(`ccxt:kraken`, `coingecko`, `synthetic`) onbellekten okurken de korunur.

---

## 9. Konfigurasyon (ozet)

| Anahtar | Varsayilan | Aciklama |
|---------|-----------|----------|
| `exchange.symbol` / `timeframe` / `limit` | BTC/USDT, 1h, 2160 | Islem cifti, mum, gecmis bar |
| `data.provider` | `auto` | auto / ccxt / synthetic |
| `data.fallback_exchanges` | `[kraken, coinbase]` | CCXT borsa zinciri |
| `backtest.fee_bps` / `slippage_bps` | 7.5 / 2.0 | Islem maliyeti |
| `backtest.risk_per_trade_pct` | 1.5 | Islem basina risk |
| `backtest.train_split` | 0.7 | Train/OOS bolunmesi |
| `backtest.run_timeout_seconds` / `max_memory_mb` | 120 / 1536 | Sandbox limitleri |
| `targets.min_net_profit_pct` | 2.0 | Terfi esigi (net kar) |
| `targets.min_win_rate` | 40.0 | Terfi esigi (win rate) |
| `targets.max_drawdown_pct` | 20.0 | Terfi esigi (max DD) |
| `targets.min_sharpe` / `min_profit_factor` | 0.25 / 1.10 | Terfi esikleri |
| `targets.min_score` | 50.0 | Terfi icin minimum skor |
| `targets.max_rework_attempts` | 3 | Aday basina onarim dongusu |
| `targets.max_cycles_per_run` | 0 | 0 = sinirsiz (24/7) |
| `targets.cycle_sleep_seconds` | 60 | Dongular arasi bekleme |
| `memory.winner_forward_return_pct` | 1.5 | Basari snapshot esigi |
| `memory.winner_horizon_bars` | 12 | Ileri getiri ufku |
| `versioning.elite_score` | 70.0 | Elite (mutasyon tohumu) esigi |
| `runtime.backend` | `native` | native / crewai / langgraph |
| `llm.provider` | `auto` | auto / openai / anthropic / offline |

Tum alanlar `AGENCY_*` ortam degiskenleriyle override edilebilir
(`.env.example`).

---

## 10. CLI

| Komut | Aciklama |
|-------|----------|
| `python main.py run` | Otonom dongu (varsayilan 24/7) |
| `python main.py run --cycles 3 --sleep 15` | 3 dongu, aralarinda 15 sn |
| `python main.py run --backend langgraph` | LangGraph state machine |
| `python main.py once` | Tek dongu |
| `python main.py status` | DB istatistikleri + skor tablosu |
| `python main.py validate [--demote]` | Terfi eden stratejileri **guncel veriyle** yeniden dogrula (decay check) |
| `python main.py doctor` | Paket/anahtar/backend/ortam kontrolu |
| `--symbol/--timeframe/--data-provider/--llm/--backend` | Tek seferlik override |
| `--max-runtime 300` | Toplam sure limiti (sn), sonra guvenli durus |

`SIGINT/SIGTERM`: mevcut dongu tamamlanir, state ve raporlar yazilir, cikilir.

---

## 11. LLM Kullanimi (opsiyonel)

```bash
export OPENAI_API_KEY=sk-...          # veya ANTHROPIC_API_KEY=...
python main.py run --llm auto         # auto: anahtar varsa OpenAI/Anthropic
```

- LLM yalnizca **JSON spec / direktif / gerekce** uretir; kod derleyiciden
  gecer (bkz. Bolum 5).
- `llm.provider: offline` (veya anahtar yok) -> deterministik planlayici:
  rejim + winner snapshot istatistikleri + dersler ile spec uretir/onarir.
  Bu modda sistem **tam kapasite** calisir; sadece fikir cesitliligi sinirlidir.
- LLM cagrilari `runtime/logs/*.jsonl` icinde `llm_call` olayi olarak
  (prompt boyutu, gecikme, hata) izlenir; prompt/cevap govdesi loglanmaz.

---

## 12. Genisletme

**Yeni indikator**: `agency/data/indicators.py` -> fonksiyon + `INDICATOR_KINDS`
kaydi + `agency/strategies/compiler.py` -> `INDICATOR_TEMPLATES` ve
`FUNCTION_FOR_KIND`. Katalog tabanli oldugu icin prompt'lar otomatik guncellenir.

**Yeni strateji ailesi**: `agency/agents/planner.py` -> `TEMPLATES` +
`FAMILIES` + `REGIME_PREFERENCE`. Uretilen spec'ler ayni dogrulama/derleme
hattindan gecer.

**Yeni LLM saglayicisi**: `agency/llm.py` -> `LLMClient` alt sinifi
(`complete`), `build_llm` icinde secim.

**PostgreSQL'e gecis**: `agency/memory/database.py` icindeki DDL/`connect`
degistirilir; `MemoryStore` arayuzu ayni kalir.

---

## 13. Test ve Dogrulama

```bash
python -m pytest              # 119 test
python -m pytest tests/test_pipeline.py -v   # uctan uca dongu
```

| Test dosyasi | Kapsam |
|--------------|--------|
| `test_indicators.py` | Indikator dogrulugu + **Donchian look-ahead regresyonu** + snapshot madenciligi |
| `test_schemas.py` | Spec/kural dogrulama, serilestirme, fingerprint |
| `test_compiler.py` | Spec -> kod, AST guvenlik, versiyon/arsiv |
| `test_engine.py` | SL/TP/trailing/**ayni-bar look-ahead**, komisyon, zaman stopu |
| `test_metrics.py` | Drawdown, sharpe, profit factor, skorlama, gate'ler |
| `test_memory.py` | SQLite dongu/snapshot/vektor/terfi/ders akisi |
| `test_planner_coder.py` | Deterministik uretim, onarim, kesif mutasyonu, direktif ayristirma |
| `test_critic.py` | Terfi / rework / arsiv kararlari + ders kaydi |
| `test_pipeline.py` | **Native ve LangGraph** uctan uca dongu; dosya + DB + rapor dogrulamasi |
| `test_cli.py`, `test_config.py` | CLI, ayar yukleme/override/dogrulama |

Bu calisma ortaminda yapilan gercek kosu (BTC/USDT 1h, 90 gun, komisyon dahil):

```
Dongu 001 | completed | en iyi: v004  skor= 82.2 | terfi: -
Dongu 002 | completed | en iyi: v011  skor= 88.1 | terfi: v011
Toplam: dongu=2 surum=13 terfi=1 snapshot=270 hafiza=1 ders=129
```

`v011` (Donchian breakout, train net +6.8%, win %66.7, DD %3.1, sharpe 4.1)
kalici hafizaya alindi; `runtime/archive/` altinda puanli kopyasi ve
`runtime/reports/cycle_002/report.md` icinde tam gerekce/direktif listesi var.

CI: `docs/ci-workflow.yml` dosyasini `.github/workflows/ci.yml` olarak
kopyalayin (pyflakes + pytest).

Ayrintili kanit kaydi (test ciktilari, log alintilari, uretilen kod ornegi):
[`docs/VERIFICATION.md`](docs/VERIFICATION.md).

---

## 14. Sinirlamalar ve Risk Uyarisi

- Bu sistem bir **Ar-Ge / strateji kesif** aracidir; **yatirim tavsiyesi
  degildir**. Gercek emir gonderimi (canli trading) bilincli olarak
  uygulanmamistir; ciktilar paper/simulasyon sonuclaridir.
- Simulasyon **long-only spot** varsayar; short, funding, likidite ve borsa
  kesintileri modellenmez.
- Gecmis performans gelecek getiriyi garanti etmez; walk-forward ve OOS
  dogrulamasi buna ragmen asiri uyumu (overfit) tamamen engelleyemez.
- Mevcut 90 gunluk BTC penceresinde sistem bazi dongulerde **hicbir stratejiyi
  terfi ettirmez**; bu bir hata degil, risk disiplininin sonucudur. Esikler
  `targets.*` altinda degistirilebilir.
- LLM anahtari yokken fikir cesitliligi sinirlidir (6 strateji ailesi +
  parametre uzayi); anahtar varsa arama uzayi genisler.

---

## 15. Sorun Giderme

| Belirti | Cozum |
|---------|-------|
| `CCXT veri cekilemedi` uyarisi | Borsa erisimi engelli olabilir; sistem otomatik olarak Kraken/Coinbase -> CoinGecko -> sentetik sirasina duser. `python main.py doctor` ile kontrol edin. |
| `news_unavailable` | Haber API'si anahtar/limit istiyor olabilir; haber sinyali opsiyoneldir, dongu devam eder. |
| `langgraph backend kullanilamadi` | `pip install "langgraph>=0.2"`; aksi halde native akis kullanilir. |
| `CrewAI icin LLM anahtari gerekli` | `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` tanimlayin veya native backend'de kalin. |
| Hic terfi yok | Piyasa rejimi stratejilere uygun degil veya esikler yuksek. `targets.*` ve `max_rework_attempts` degerlerini ayarlayin; `runtime/reports/INDEX.md` ve dersleri inceleyin. |
| Dongu cok yavas | `exchange.limit`, `targets.candidates_per_cycle` ve `targets.max_rework_attempts` degerlerini dusurun. |

---

## 16. Lisans

Bu proje ornek/egitim amacli bir referans mimaridir. Kullanmadan once kendi
risk yonetimi, yasal uyum ve guvenlik gereksinimlerinizi degerlendirin.




