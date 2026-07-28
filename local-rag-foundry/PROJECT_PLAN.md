# Yerel RAG Asistanı — Proje Planı (Foundry Local)

> Kaynak dokümanlar: `goal_ documents/Building Your First Local RAG Application with Foundry Local...pdf` (mimari referans) ve `goal_ documents/Summer School Foundry Local Plan.docx` (öğrenme hedefleri / gereksinimler — orijinali 5-6 haftalık, biz 12 saate sıkıştırıyoruz).
>
> Ayrıca `github.com/YusufAtakanUnal/local-rag-foundry` (MIT lisanslı, izinle incelendi) adlı referans repo taranarak gerçek dünyada doğrulanmış teknik detaylar plana işlendi — bkz. "Referans Repo İncelemesi" bölümü.

## Referans Repo İncelemesi — Plana Yansıtılan Bulgular

Arkadaş projesinde hem JS (blog'un orijinali) hem de bize doğrudan uyan bir **Python** implementasyonu vardı. İncelemeden çıkan ve planı değiştiren kritik noktalar:

1. **Gerçek Foundry Local Python API'si `createChatClient()` değil.** Blog yazısındaki JS örneği güncel/gerçek SDK davranışını yansıtmıyor. Gerçek kalıp:
   ```python
   from foundry_local import FoundryLocalManager
   import openai
   manager = FoundryLocalManager(model_alias)          # servisi başlatır, modeli indirir/yükler
   client = openai.OpenAI(base_url=manager.endpoint, api_key="not-required")
   client.chat.completions.create(model=manager.get_model_info(model_alias).id, messages=[...], stream=True)
   ```
   Yani Foundry Local, OpenAI-uyumlu bir yerel endpoint açıyor; biz de `openai` paketiyle konuşuyoruz. Bu bulgu olmasaydı Aşama 1'de yanlış API'yi debug ederek ciddi zaman kaybedebilirdik.
2. **Retrieval için embedding değil, TF-IDF + cosine similarity kullanılıyor** — hem blog hem referans repo bunu bilinçli tercih ediyor: sıfır ek model indirmesi, anlık hız, tam şeffaflık (vektörler incelenebilir), 20 dokümanlık küçük/orta ölçek için yeterli isabet. **Planı güncelliyoruz: RTX 5080 gücüne rağmen embedding modelini çekirdek mimariden çıkarıp TF-IDF'i temel retrieval yöntemi yapıyoruz** — 12 saatlik bütçede risk azaltma önceliklidir. Embedding tabanlı hibrit arama, zaman kalırsa Aşama 4'te opsiyonel bir "orta profesyonel" cila olarak eklenir (bkz. aşağıda).
3. **Zarif bozulma (graceful degradation):** Foundry Local kurulu/çalışır değilse uygulama çökmüyor; retrieval kısmı çalışmaya devam ediyor ve en alakalı kaynak pasajını gösteriyor, LLM üretimi olmadan. Demo günü riskini ciddi şekilde azaltan bu deseni `chat_engine.py` ve `foundry_client.py` tasarımına baştan dahil ediyoruz.
4. **Hazır 20 dokümanlık bilgi tabanı** (gaz sahası mühendisliği: sızıntı tespiti, H2S izleme, vana bakımı, acil kapatma, PPE, vb.) — MIT lisanslı, kullanıcının izniyle doğrudan projeye kopyalandı (`data/docs/`). Kendi placeholder dokümanlarımızı yazmak yerine bu zengin, gerçekçi corpus'u kullanıyoruz; Aşama 1'de ciddi zaman kazandırıyor.
5. **Bağımlılık ayak izi küçülüyor:** Ingestion + retrieval için **hiçbir üçüncü parti paket gerekmiyor** (sadece Python stdlib: `sqlite3`, `re`, `math`, `collections`). Sadece cevap *üretimi* için `foundry-local-sdk` ve `openai` gerekiyor. `numpy` ve `python-frontmatter` planımızdan çıkarılıyor.
6. **Test kapsamı örnekleri:** arama sıralamasının doğruluğu, top-k sınırı, alakasız sorguda boş sonuç, doküman silme/yeniden ekleme idempotency, model hazır değilken zarif bozulmanın doğrulanması, "bilgi tabanında yok" fallback'i. Bu senaryolar bizim `tests/` planımıza doğrudan giriyor.

## Hedef

Tamamen yerel, internet bağlantısı gerektirmeyen, kaynak referanslı (source-attribution) bir doküman Soru-Cevap asistanı. Hiçbir ücretli API, hiçbir bulut çağrısı yok. Model indirmesi hariç tüm çalışma zamanı tamamen çevrimdışı.

## Onaylanan Mimari Kararlar

| Karar | Seçim | Gerekçe |
|---|---|---|
| AI Runtime | **Microsoft Foundry Local** (`foundry-local-sdk` + `openai` paketi) | Kaynak dokümanın talep ettiği teknoloji; gerçek SDK OpenAI-uyumlu endpoint sunuyor (referans repoda doğrulandı); RTX 5080 GPU'yu otomatik kullanır; sıfır API anahtarı |
| Dil | **Python 3.11+** | Foundry Local Python SDK + docx müfredatıyla birebir uyumlu |
| Retrieval | **TF-IDF + cosine similarity** (saf Python, 3. parti bağımlılık yok) | Blog + referans repo tarafından doğrulanmış, sıfır ek model indirmesi, anlık hız, tam şeffaflık; 12 saatlik bütçede en düşük riskli seçenek |
| Chat modeli | Foundry Local üzerinden `phi-3.5-mini` | Küçük, hızlı, GPU'da rahat çalışır; RTX 5080'de gecikme sorun olmaz |
| Vektör depolama | **SQLite** (tek dosya, `data/knowledge.db`) | Sıfır altyapı; sadece stdlib `sqlite3` |
| Zarif bozulma | Foundry Local hazır değilse retrieval-only moda düş | Demo günü riskini azaltır — model çökse bile uygulama çalışır durumda kalır |
| Arayüz | **Streamlit** | Python-native, 12 saatlik bütçede hızlı ve profesyonel görünüm |
| Bilgi tabanı | **20 dokümanlık gaz sahası mühendisliği corpus'u** (referans repodan, MIT lisans, izinli) | Gerçekçi, zengin, önceden test edilmiş içerik — kendi placeholder'ımızı yazmaktan çok daha hızlı ve güçlü |
| Testler | `pytest` (tfidf, chunker, vector store, chat engine — referans repo test senaryolarından ilham alınarak) | docx'in "Functional Testing" gereksinimini karşılar |
| Stretch (zaman kalırsa) | Embedding tabanlı hibrit skor (RTX 5080 ile) | TF-IDF + semantik skoru birleştirip "orta profesyonel" seviyeyi minimum gereksinimin üstüne taşımak — **opsiyonel, çekirdek teslimat buna bağımlı değil** |

## Dosya Yapısı (oluşturuldu)

```
local-rag-foundry/
├── PROJECT_PLAN.md          <- bu dosya
├── README.md                 <- kurulum & çalıştırma talimatları + kaynak atfı (Aşama 4'te doldurulacak)
├── requirements.txt           <- foundry-local-sdk, openai, streamlit, pytest (ingestion/retrieval bağımlılıksız)
├── .gitignore
├── config.py                  <- model adı, chunk_size/overlap, top_k, DB yolu (Aşama 1)
├── data/
│   ├── docs/                  <- 20 dokümanlık gaz sahası mühendisliği corpus'u (referans repodan, hazır)
│   └── knowledge.db            <- SQLite deposu (Aşama 2'de üretilir, git'e girmez)
├── src/
│   ├── tfidf.py                <- tokenize / term_frequency / idf / cosine_similarity (saf Python) (Aşama 2)
│   ├── chunker.py                <- front-matter parse + overlapping chunking (Aşama 2)
│   ├── vector_store.py             <- SQLite + ters indeks + search() (Aşama 2)
│   ├── ingest.py                    <- docs/ klasörünü işleyip DB'ye yazan pipeline (Aşama 2)
│   ├── foundry_client.py             <- FoundryLocalManager + openai istemcisi, zarif bozulma (Aşama 1/3)
│   ├── prompts.py                     <- güvenlik odaklı sistem promptu + mesaj kurgusu (Aşama 3)
│   ├── chat_engine.py                  <- retrieve → augment → generate, event-yield deseni (Aşama 3)
│   └── security.py                      <- dosya yükleme doğrulama, path traversal koruması (Aşama 4)
├── app/
│   └── streamlit_app.py         <- Streamlit arayüzü: chat + kaynak paneli + dosya yükleme (Aşama 3)
├── scripts/
│   └── run_ingest.py             <- `python scripts/run_ingest.py [--reset]` CLI komutu (Aşama 2)
└── tests/
    ├── test_tfidf.py
    ├── test_chunker.py
    ├── test_vector_store.py
    └── test_chat_engine.py
```

## 12 Saatlik Zaman Planı — 4 Aşama

### Aşama 1 — Ortam Kurulumu & İskelet (≈2 saat)
- Foundry Local kurulumu doğrulama (`winget install Microsoft.FoundryLocal`), GPU'nun (RTX 5080) tanındığının kontrolü
- `foundry-local-sdk` + `openai` paketlerinin kurulumu; "hello model" testi: `FoundryLocalManager("phi-3.5-mini")` ile servis başlatma + `openai.OpenAI(base_url=manager.endpoint)` üzerinden örnek bir chat tamamlama
- `config.py`: model adı, chunk_size=200, chunk_overlap=25, top_k=3 gibi ayarlar
- `src/foundry_client.py`: model başlatma + zarif bozulma mantığı (SDK yoksa/servis kapalıysa `ready=False`, anlamlı hata mesajı)
- 20 dokümanlık gaz sahası corpus'u zaten `data/docs/`'a kopyalandı ✅
- **Kilometre taşı:** Foundry Local'den örnek bir prompt'a gerçek yanıt alınabiliyor; SDK yoksa da uygulama çökmüyor

### Aşama 2 — Veri Katmanı & Retrieval Pipeline (≈3 saat)
- `src/tfidf.py`: tokenize, term_frequency, idf, cosine_similarity (saf Python, bağımlılıksız)
- `src/chunker.py`: markdown dokümanlarını front-matter'ı ayrıştırarak ~200 kelimelik, 25 kelime overlap'li parçalara böler
- `src/vector_store.py`: SQLite'a chunk + term-frequency map yazar; sorgu zamanında ters indeks ile aday daraltma + cosine similarity sıralama (in-memory cache)
- `src/ingest.py` + `scripts/run_ingest.py`: `python scripts/run_ingest.py [--reset]` ile docs/ klasörünü indeksleme
- Birim testler: sıralama doğruluğu, top-k limiti, alakasız sorguda boş sonuç, doküman silme/yeniden ekleme idempotency
- **Kilometre taşı:** "How do I detect a gas leak?" gibi bir soru için doğru chunk'ların top-3'te geldiği doğrulanmış

### Aşama 3 — LLM Entegrasyonu & Arayüz (≈3.5 saat)
- `src/prompts.py`: güvenlik öncelikli sistem promptu — halüsinasyon yasağı, "bilgi tabanında yok" cevabı, kaynak atıfı zorunluluğu, yapılandırılmış yanıt formatı (Özet/Güvenlik Uyarıları/Adımlar/Referans)
- `src/chat_engine.py`: kullanıcı sorusu → TF-IDF retrieval → prompt oluşturma → Foundry Local chat client (streaming) → event akışı (`sources`/`token`/`done`/`error`); model hazır değilse en alakalı pasajı gösteren zarif bozulma
- `app/streamlit_app.py`: sohbet arayüzü, kaynak/relevance panel, `.md`/`.txt` dosya yükleme (runtime'da yeniden indeksleme), oturum içi sohbet geçmişi
- **Kilometre taşı:** Uçtan uca çalışan, tarayıcıda test edilmiş bir demo — Foundry Local kapalıyken de retrieval çalışıyor

### Aşama 4 — Test, Güvenlik, Dokümantasyon (≈2.5-3 saat) ✅ tamamlandı
- Fonksiyonel test seti: cevaplanabilir / cevaplanamaz / uç durum sorguları (docx'teki "Functional Testing" gereksinimi)
- `security.py`: yüklenen dosyalar için uzantı whitelist (.md/.txt), boyut limiti, path traversal koruması; çalışma zamanında sıfır dış ağ çağrısı olduğunun doğrulanması
- `README.md`: mimari diyagram (metin/ASCII), kurulum adımları, çalıştırma komutları, sınırlamalar, **bilgi tabanı kaynağının atfı** (referans repo + MIT lisans notu)
- **Kilometre taşı:** Proje "orta profesyonel" seviyede teslime hazır — 32 test yeşil

### Aşama 5 — Stretch: Hibrit (TF-IDF + Semantik) Retrieval ✅ tamamlandı
Kullanıcı onayıyla, çekirdek teslimattan sonra eklendi:
- `src/embedder.py`: `qwen3-embedding-0.6b` ile Foundry Local üzerinden embedding üretimi, zarif bozulma (model yoksa `ready=False`, arama otomatik TF-IDF'e döner)
- `src/tfidf.py`: dense vektörler için `dense_cosine_similarity` eklendi
- `src/vector_store.py`: `embedding` kolonu (eski DB'ler için otomatik `ALTER TABLE` migration), `search()` artık `query_embedding` verildiğinde TF-IDF + semantik skoru `HYBRID_TFIDF_WEIGHT`/`HYBRID_EMBEDDING_WEIGHT` ile harmanlıyor; semantik arama, TF-IDF'in ıskaladığı (ortak kelime paylaşmayan ama anlamca yakın) chunk'ları da adayları arasına alıyor
- `ingest.py` / `run_ingest.py --no-embed`, `chat_engine.py`, `streamlit_app.py`: embedder uçtan uca bağlandı; arayüzde kaynaklar artık TF-IDF/semantik skor kırılımını gösteriyor
- 4 yeni test (hibrit eşleşme, saf TF-IDF geriye dönük uyumluluk, skor harmanlama, eski şema migration) — toplam **36 test yeşil**
- Gerçek 20 dokümanlık corpus embedding'lerle yeniden indekslendi ve gerçek model + gerçek embedder ile uçtan uca doğrulandı

## Kapsam Dışı (12 saat için ertelenenler)
- Konuşma geçmişinin kalıcı depolanması (sadece oturum içi tutulacak)
- PWA / mobil paketleme
- Çoklu kullanıcı / kimlik doğrulama (tek kullanıcılı yerel araç olarak tasarlanıyor)
- Büyük ölçekte (binlerce chunk) ANN indeksleme (FAISS vb.) — mevcut hibrit arama tüm chunk'ları tarıyor, bu ölçekte (onlarca chunk) sorun değil

## Güvenlik İlkeleri (baştan itibaren uygulanacak)
- Model indirmesi dışında **hiçbir outbound network çağrısı yok**
- Dosya yükleme: uzantı whitelist + boyut limiti + path traversal koruması
- Sistem promptu: modelin bağlam dışına çıkıp uydurma yapmasını engelleyen açık kurallar
- Sır/gizli bilgi (API anahtarı vb.) yok — proje zaten anahtarsız çalışıyor

---

# Faz 2 — Çok Dilli Destek + PDF/DOCX Yükleme ✅ tamamlandı

Proje ilk sürümüyle push edildikten sonra, kullanıcı isteğiyle eklenen ikinci faz.
Hedef: **eski sistemi bozmadan**, diller-arası arama ve daha zengin doküman
formatı desteği eklemek.

**Sonuç:** Tüm maddeler uygulandı ve gerçek modelle uçtan uca doğrulandı (bkz.
`TEST_REPORT.md`). Test paketi 36 → **49 teste** çıktı. Uygulama sırasında iki
gerçek hata daha bulunup düzeltildi: (1) `st.file_uploader` her rerun'da aynı
dosyayı döndürdüğü için, önceki `st.rerun()` çağrısı sonsuz döngüye giriyordu
— `session_state` ile işlenen dosya takip edilerek çözüldü; (2) küçük modelin
(phi-3.5-mini) bağlamın sonuna gömülü tek bir dil talimatını yeterince
önemsemediği görüldü — talimat hem sistem promptuna hem soruya en yakın
noktaya iki kez yerleştirilerek güvenilirliği artırıldı.

## Araştırma Bulguları

Karar vermeden önce, mevcut embedding modelimizin (`qwen3-embedding-0.6b`,
Foundry Local üzerinden) gerçekten diller arası anlamsal hizalama yapıp
yapmadığı canlı olarak test edildi (referans: "How do I detect a gas leak
near a pipeline?"):

| Karşılaştırılan metin | Cosine benzerlik | Yorum |
|---|---|---|
| Türkçe çevirisi ("Bir boru hattında gaz kaçağını nasıl tespit ederim?") | **0.616** | Yüksek — aynı anlam, farklı dil |
| Türkçe + İngilizce karışık sorgu ("Gas leak durumunda pipeline yakınında ne yapmalıyım?") | **0.686** | En yüksek — code-switching sorun yaratmıyor, ortak kelimeler skoru güçlendiriyor |
| Alakasız Türkçe cümle ("Yarın hava nasıl olacak, yağmur yağacak mı?") | 0.311 | Belirgin şekilde düşük — ayrım net |
| Aynı dil, gerçek doküman parçası | 0.840 | Beklenen üst sınır (referans) |

**Sonuç:** Qwen3-Embedding çok dilli bir model; ek bir çeviri adımına veya
ayrı bir çok-dilli kütüphaneye gerek yok. Mevcut hibrit retrieval mimarimiz
(TF-IDF + embedding, bkz. Aşama 5) **zaten** diller-arası ve karışık-dilli
sorguları anlamlı şekilde ayırt edebiliyor — çünkü TF-IDF kelime örtüşmesi
bulamadığında (örn. Türkçe soru + İngilizce doküman) skor otomatik olarak
embedding tarafına kayıyor. Eklenmesi gereken şey yeni bir "çok dilli motor"
değil, mevcut motorun önündeki engelleri kaldırmak ve bunu kullanıcıya
arayüzden sunmak.

## Kapsam ve Mimari Kararlar

| Konu | Karar | Gerekçe |
|---|---|---|
| Diller-arası arama | Ek kütüphane/çeviri **yok** — mevcut embedding hibrit skoruna güveniliyor | Araştırma bunu doğruladı; basitlik + tamamen yerel kalır |
| TF-IDF tokenizer | Unicode-güvenli hale getiriliyor (şu an ç/ğ/ı/ö/ş/ü gibi karakterleri siliyor — **gerçek bir hata**) | Türkçe metinlerde TF-IDF yarısı şu an bozuk çalışıyor; embedding'in üstüne binen TF-IDF katkısı da düzelmeli |
| Stopword listesi | İngilizce listeye Türkçe stopword'ler eklenecek (birleşik liste, dil algılama yok) | Karışık-dilli sorgularda "hangi dil" kararı vermeye gerek kalmaz, ikisi de doğru filtrelenir |
| Dil seçici (UI) | Sidebar'da "Response language" seçici: Auto / Türkçe / English | Kullanıcının "üstten dil seçme" isteğini karşılar; pratikte en anlamlı kaldıraç **yanıt dili**dir (girdi dili zaten LLM tarafından otomatik anlaşılıyor, retrieval zaten dil-bağımsız) |
| Karışık-dilli sorgu (code-switching) | Ekstra kod gerekmiyor — Unicode tokenizer + çok dilli embedding zaten bunu doğru işliyor (araştırma tablosundaki 0.686 skoru kanıtı) | Over-engineering'den kaçınılıyor |
| PDF yükleme | `pypdf` (saf Python, yerel, ücretsiz) ile metin çıkarımı | Taranmış (OCR gerektiren) PDF'ler kapsam dışı — metin katmanı olmayan PDF'lerde "metin bulunamadı" uyarısı verilecek |
| DOCX yükleme | `python-docx` (yerel, ücretsiz) ile paragraf metni çıkarımı | Basit, yaygın, tablo/görsel çıkarımı kapsam dışı (sadece düz metin) |

## Dosya Bazında Değişiklik Planı

```
config.py                    # ALLOWED_UPLOAD_EXTENSIONS'a .pdf/.docx eklenir
requirements.txt             # pypdf, python-docx eklenir
src/
├── tfidf.py                  # _NON_WORD_RE Unicode-güvenli yapılır; TURKISH_STOPWORDS eklenip STOPWORDS ile birleştirilir
├── chunker.py                  # _FILE_EXT_RE'ye .pdf/.docx eklenir (doc_id türetimi için)
├── document_readers.py          # YENİ: extract_pdf_text(), extract_docx_text(), extract_text_for_upload() dispatcher
├── ingest.py                     # DOC_EXTENSIONS genişler; ingest_file() dosya uzantısına göre doğru extractor'ı çağırır
├── prompts.py                     # build_messages()'a opsiyonel response_language parametresi; sistem promptuna dil talimatı eklenir
└── chat_engine.py                  # ask()'a opsiyonel response_language parametresi eklenir, prompts.build_messages'a iletilir
app/
└── streamlit_app.py            # sidebar'da dil seçici (st.selectbox); file_uploader'a pdf/docx eklenir; upload handler'da document_readers dispatcher kullanılır
tests/
├── test_tfidf.py                # Unicode tokenizer testleri (Türkçe karakterler korunuyor mu)
├── test_document_readers.py      # YENİ: pypdf/python-docx ile testte üretilen küçük dosyaları round-trip okuma testi
└── test_vector_store.py          # (opsiyonel) diller-arası hibrit arama senaryosu testi
README.md                    # çok dilli destek + PDF/DOCX + bilinen sınırlamalar (taranmış PDF, OCR yok) güncellenir
```

## Geriye Dönük Uyumluluk (eski sistem bozulmayacak)

- Tokenizer değişikliği sadece regex'i genişletiyor (daraltmıyor) — mevcut İngilizce testler etkilenmeyecek, sadece Türkçe karakterler artık silinmeyecek.
- `response_language` parametresi **opsiyonel** (varsayılan `None`/"Auto") — mevcut `ChatEngine.ask()` çağrıları (testler dahil) değişiklik yapmadan çalışmaya devam eder.
- PDF/DOCX desteği mevcut `.md`/`.txt` akışına **ek** bir dal; markdown/metin yükleme davranışı değişmiyor.
- Mevcut `data/knowledge.db` şeması değişmiyor (embedding kolonu zaten Faz 1'de eklendi); yeniden migration gerekmiyor.

## Kapsam Dışı (bu fazda yapılmayacak)
- OCR (taranmış/görsel PDF'lerden metin çıkarımı) — ayrı bir vision modeli gerektirir, kapsam dışı
- Otomatik dil algılama ile dinamik hibrit ağırlık değişimi (`HYBRID_*_WEIGHT`) — sabit ağırlıklar zaten işe yarıyor, gereksiz karmaşıklık
- Doküman bazında "bu chunk şu dilde" etiketleme/filtreleme — istenirse ayrı bir istek olarak ele alınabilir

---

## Sonradan Eklenen Not — Sohbet Modeli Değişikliği

Faz 2 testleri sırasında `phi-3.5-mini`'nin çok dilli/RAG yükü altında ciddi
ve tekrarlanabilir bir "repetition collapse" arızası olduğu bulundu. Kapsamlı
prompt/parametre mühendisliği (iki aşamalı üretim, sıcak yeniden deneme,
ceza ayarları, genişletilmiş tekrar algılama, geri-dönüş mekanizmaları) kök
nedeni çözemedi — sadece hasarı sınırladı. Model `qwen2.5-7b`'ye değiştirildi
ve sorun tamamen ortadan kalktı. Tüm detaylar ve karşılaştırma verileri
`TEST_REPORT.md`'de (madde 13).

---

## Proje Kapanışı

Proje, `goal_ documents/`'teki iki kaynak dökümanın (Microsoft blog yazısı +
Summer School müfredatı) gereksinimlerini karşılayacak şekilde tamamlandı.
Kapsamlı bir test-değerlendir-düzelt döngüsünden geçti (bkz. `TEST_REPORT.md`
madde 3-5) ve kalan bilinen sınırlamalar saklanmadan belgelendi. Projenin
son, kapanış hâli için bkz. **`FINAL_REPORT.md`** — iki kaynak dökümanla
madde madde karşılaştırma, tüm mühendislik kararlarının kronolojisi ve
dürüst bir sınırlamalar listesi içeriyor.
