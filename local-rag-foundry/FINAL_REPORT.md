# Local RAG Foundry — Final Proje Raporu

> Bu doküman, projenin baştan sona tüm sürecini, aldığı mimari kararları, iki
> kaynak dökümanın (`goal_ documents/`) gereksinimlerini ne ölçüde
> karşıladığını, karşılaşılan tüm hataları/düzeltmeleri ve **dürüstçe kalan
> sınırlamaları** tek bir yerden özetler. `PROJECT_PLAN.md` (planlama),
> `TEST_REPORT.md` (test kayıtları) ve `README.md` (kullanım) ile birlikte
> okunmalıdır — bu rapor onların üstüne, kapanış amaçlı bir sentezdir.

## 1. Proje Amacı

Tamamen yerel/çevrimdışı çalışan, kaynak referanslı (source-attribution) bir
RAG (Retrieval-Augmented Generation) doküman Soru-Cevap asistanı. Hiçbir
ücretli API, hiçbir bulut çağrısı yok; model indirmesi dışında sıfır dış ağ
trafiği. Microsoft Foundry Local üzerinde çalışan yerel bir dil modeli, yerel
bir SQLite vektör deposu ve kullanıcının kendi dokümanlarıyla çalışır.

## 2. İki Kaynak Döküman ve Aralarındaki Gerilim

- **Döküman 1** — Microsoft Community Hub blog yazısı ("Building Your First
  Local RAG Application with Foundry Local"): JavaScript/Node.js/Express
  tabanlı bir referans mimari; TF-IDF + SQLite + Foundry Local; güvenlik
  odaklı sistem promptu; runtime doküman yükleme; "Ideas for Extending"
  bölümünde hibrit (TF-IDF + embedding) retrieval'i açıkça bir sonraki adım
  olarak öneriyor.
- **Döküman 2** — "Summer School Foundry Local Plan" (.docx): Yeni başlayan
  öğrenciler için 5-6 haftalık bir müfredat; **Python** + Foundry Local
  Python SDK + SQLite + embeddings + Streamlit/Gradio arayüzünü açıkça
  istiyor; embedding modeli örneği olarak **`qwen3-embedding-0.6b`**'yi,
  chat modeli örneği olarak **Phi-3.5 Mini**'yi ismen veriyor; 3 fazlı bir
  yapı (Temel Öğrenme / Uygulama / Test-Dokümantasyon) tanımlıyor.

İki döküman arasındaki dil/teknoloji farkını (JS vs Python) **Döküman 2
lehine** çözdük: proje tamamen Python'da (Streamlit arayüzüyle) yazıldı,
ama Döküman 1'in mimari deseni, sistem promptu yapısı (Özet/Güvenlik
Uyarıları/Adımlar/Referans), retrieval stratejisi ve genişletme fikirleri
(hibrit retrieval) birebir uygulandı. Bu, projenin en başında bilinçli
verilmiş ve tutarlı şekilde sürdürülmüş bir karardır.

## 3. Mimari — Ne İnşa Edildi

```
Soru ─┬─▶ TF-IDF vektörü ──▶ ters-indeks aday bulma ──▶ cosine skoru ─┐
      └─▶ sorgu embedding'i (qwen3-embedding-0.6b) ──▶ semantik skor ─┼─▶ harmanlanmış sıralama (TOP_K=5)
                                                                       ▼
                     en iyi chunk'lar → prompt (sistem + bağlam + geçmiş) → Foundry Local (qwen2.5-7b)
                     → akan yanıt + kaynak atıfları → Streamlit sohbet arayüzü
```

| Katman | Teknoloji | Notlar |
|---|---|---|
| AI modeli | Foundry Local + **Qwen2.5-7B-Instruct** | Bkz. bölüm 6 — başlangıç modeli `phi-3.5-mini`'ydi, güvenilirlik sorunları nedeniyle değiştirildi |
| Embedding modeli | **qwen3-embedding-0.6b** | Döküman 2'de ismen belirtilen model — birebir kullanıldı |
| Retrieval | Hibrit: TF-IDF + semantik embedding, ağırlıklı harman (`0.5`/`0.5`) | Döküman 1'in "Ideas for Extending" bölümündeki hibrit retrieval önerisi gerçekleştirildi |
| Diller-arası arama | Ek çeviri kütüphanesi yok — çok dilli embedding zaten yeterli (canlı test: 0.616-0.840 arası tutarlı benzerlik skorları) | |
| Doküman formatları | `.md`, `.txt`, `.pdf`, `.docx` | Döküman 1'in temel önerisinin (`.md`/`.txt`) ötesine geçildi |
| Vektör deposu | SQLite (`data/knowledge.db`, tek dosya) | |
| Arayüz | Streamlit | Döküman 2'nin "Option B" seçimi birebir uygulandı |
| Testler | `pytest`, 75 test, 7 dosya | |

## 4. Döküman 1 (Blog) — Gereksinim Karşılama Tablosu

| Gereksinim | Durum | Not |
|---|---|---|
| Tamamen offline, API anahtarsız | ✅ | Model indirmesi dışında sıfır dış ağ çağrısı |
| Retrieve → Augment → Generate deseni | ✅ | `chat_engine.py` |
| SQLite vektör deposu (tek dosya) | ✅ | |
| TF-IDF + cosine similarity | ✅ | Ek olarak hibrit semantik katman da eklendi |
| Sistem promptu: Özet/Güvenlik Uyarıları/Adımlar/Referans formatı | ✅ | Birebir aynı yapı |
| Kaynak atıfı + relevance skoru gösterimi | ✅ | Sidebar'da TF-IDF/semantik skor kırılımıyla |
| Runtime doküman yükleme (restart gerektirmez) | ✅ | `.md`/`.txt`'nin ötesinde `.pdf`/`.docx` da eklendi |
| Zarif bozulma (model hazır değilse retrieval-only) | ✅ | `foundry_client.py` / `chat_engine.py` |
| Yapılandırılabilir chunk_size/overlap/top_k/model | ✅ | `config.py` |
| Birim testler | ✅ | 75 test (bloğun "Node test runner" örneğinden çok daha kapsamlı) |
| Öneri: Hibrit (TF-IDF+embedding) retrieval | ✅ **uygulandı** | Blog bunu "ileride yapılabilir" diye önermişti, biz çekirdek mimariye dahil ettik |
| Öneri: Konuşma hafızası (persistent) | ⚠️ Kısmi | Oturum içi tutuluyor, sayfa yenilenince kayboluyor — blog da bunu sadece bir "fikir" olarak listeliyordu, zorunlu gereksinim değil |
| Öneri: Multi-modal, PWA, CAG karşılaştırması | ❌ Yapılmadı | Blog bunları "ileri seviye fikirler" olarak listeliyor, kapsam dışı bırakıldı (bilinçli) |

## 5. Döküman 2 (Summer School Planı) — Faz Bazlı Karşılama

| Faz | Gereksinim | Durum |
|---|---|---|
| Faz 1 — Temel Öğrenme | RAG kavramı, Foundry Local kurulumu, embeddings/vektör benzerliği, SQLite, prompt engineering | ✅ Tümü uygulandı — embedding modeli olarak müfredatın ismen belirttiği `qwen3-embedding-0.6b` kullanıldı |
| Faz 2 — Uygulama: Veri Katmanı | Chunk'lama, embedding üretimi, SQLite'a yazma, `get_top_chunks()` benzeri retrieval fonksiyonu | ✅ `chunker.py`, `vector_store.py` |
| Faz 2 — Uygulama: LLM Entegrasyonu | Foundry Local chat modeli (örnek: Phi-3.5 Mini), `answer_query()` benzeri fonksiyon, bağlamdan cevap üretme | ✅ — model başlangıçta müfredatın önerdiği Phi-3.5 Mini idi, güvenilirlik sorunları nedeniyle Qwen2.5-7B'ye geçildi (bkz. bölüm 6) |
| Faz 2 — Arayüz | Seçenek B: Streamlit/Gradio | ✅ Streamlit seçildi ve uygulandı |
| Faz 2 — Sorumlu çıktı | "Bağlamda yoksa bilmediğini söyle", kaynak atıfı | ✅ — bu proje boyunca en çok test edilen ve düzeltilen davranış oldu (bkz. bölüm 7) |
| Faz 3 — Fonksiyonel Test | Cevaplanabilir/cevaplanamaz/uç durum sorguları, sonuçların belgelenmesi | ✅ 75 pytest + 5 turluk canlı 10-soru testi (v1-v5), hepsi `TEST_REPORT.md`'de kayıtlı |
| Faz 3 — Performans/Debug | Yanıt sürelerinin makullüğü, halüsinasyon/format sorunlarının giderilmesi | ✅ CPU-only inference kabul edildi (GPU execution-provider yoktu), repetition-collapse kök nedeni bulunup çözüldü |
| Faz 3 — Değerlendirme & İyileştirme | Kendi kendine eleştiri, yineleme | ✅ v1→v5 arası 5 iterasyonluk test-düzelt-yeniden test döngüsü |
| Faz 3 — Dokümantasyon | Proje raporu/README, kurulum talimatları, tasarım kararları ve sınırlamalar | ✅ README.md + PROJECT_PLAN.md + TEST_REPORT.md + bu rapor |
| Faz 3 — Final sunum | Demo günü, canlı sunum | N/A — tek kişilik proje, canlı sunum gereksinimi yok; bu rapor o rolü üstleniyor |

**Müfredatın istediğinin ötesine geçilen noktalar:** çok dilli (Türkçe/İngilizce)
destek ve karışık-dilli sorgu toleransı (müfredatta yok), `.pdf`/`.docx`
yükleme (müfredat sadece düz metin/markdown öngörüyordu), 20 gerçek/ham
referans doküman (müfredat "5-10 kısa doküman" öneriyordu, biz 40 dokümana
— 20 zenginleştirilmiş anlatı + 20 ham gerçek OSHA/EPA kaynağı — çıktık),
75 otomatik testlik bir paket (müfredat sadece "test cases" diyordu, sayı
belirtmiyordu).

**Bilinçli olarak kapsam dışı bırakılanlar:** çoklu kullanıcı/kimlik
doğrulama (tek kullanıcılı yerel araç olarak tasarlandı), kalıcı
(oturumlar-arası) konuşma geçmişi, PWA paketleme, CAG mimarisiyle
karşılaştırma — bunların hiçbiri iki dökümanın da **zorunlu** gereksinimi
değildi.

## 6. Proje Boyunca Yaşanan Kritik Mühendislik Kararları (kronolojik)

1. **Foundry Local Python API'sinin gerçek şeklinin doğrulanması** —
   dökümanlardaki JS örneği (`createChatClient()`) yanıltıcıydı; gerçek
   Python API'si (`FoundryLocalManager` + `ChatClientSettings`) referans
   repo incelemesiyle doğrulandı, zaman kaybı önlendi.
2. **`phi-3.5-mini` → `qwen3-8b` (terk edildi) → `qwen2.5-7b` model
   değişimi** — küçük modelin ciddi ve tekrarlanabilir bir "repetition
   collapse" (sonsuz tekrar) arızası vardı. Kapsamlı prompt/parametre
   mühendisliği (iki-aşamalı üretim, sıcak yeniden deneme, ceza ayarları,
   karakter/kelime/cümle seviyeli tekrar algılama) hasarı sınırladı ama kök
   nedeni çözemedi; model değişimi sorunu tamamen ortadan kaldırdı.
   `qwen3-8b` denendi ama varsayılan "thinking" modu aynı soruna düştüğü
   için terk edildi.
3. **Doküman corpus'unun zenginleştirilmesi** — ilk 20 doküman (referans
   repodan, MIT lisans, izinli) kullanıcı tarafından "yapmacık/şablonik"
   bulundu; 3.730 kelimeden 10.766 kelimeye çıkarılarak doğal, açıklayıcı
   düzyazıyla yeniden yazıldı (teknik gerçekler — basınçlar, eşikler,
   prosedürler — değişmedi).
4. **Referans repoyla örtüşen isimlendirmenin ayrıştırılması** — dosya
   adları ve bölüm başlıkları (`Purpose`/`Safety Warnings`/`Procedure`/
   `Reference`) referans repoyla birebir aynıydı; hepsi yeniden adlandırıldı
   (`## Overview`/`## Key Safety Precautions`/`## Working Procedure`/
   `## Source Standard` vb.).
5. **20 gerçek, ham referans dokümanının eklenmesi** — kullanıcı isteğiyle,
   Wikipedia hariç tutularak (kullanıcının açık talimatı), ABD federal
   hükümet kaynaklarından (OSHA/EPA — 17 U.S.C. § 105 gereği kamu malı,
   telif sorunu yok) `curl` + Python `html.parser` ile gerçek ham metin
   çekildi, sadece nav/footer temizlenerek, projenin kendi şablonuna
   **sokulmadan** eklendi — gerçek bir PDF/web yüklemesini simüle etmek
   için.
6. **10 soruluk sistematik Q&A testi ve 3 turluk düzeltme döngüsü (v3→v5)**
   — kullanıcının bir ChatGPT değerlendirmesi üzerine başlatıldı, önce
   bulgular dürüstçe raporlandı (v3), sonra kök nedenler bulunup düzeltildi
   (v4: `TOP_K` 3→5, ret-cümlesi/referans-etiketi prompt netleştirmesi,
   "Auto" modda Türkçe algılama ile iki-aşamalı akışın devreye girmesi),
   son olarak çeviri akıcılığı için satır-satır bir mimari denendi ama işe
   yaramadığı görülüp **dürüstçe geri alındı** (v5) — bkz. bölüm 7.

## 7. Bilinen Sınırlamalar (dürüstçe, gizlenmeden)

Bu proje "her şey mükemmel çalışıyor" iddiasında değil. Aşağıdakiler,
birden fazla iterasyonla uğraşılmış ama tam çözülememiş, kabul edilmiş
sınırlardır:

- **İngilizce→Türkçe çeviri akıcılığı zayıf.** `qwen2.5-7b`, zorunlu
  Türkçe yanıtlarda bazen İngilizce-Türkçe karışık ("code-mixed") cümleler
  üretiyor (örn. "Do confined spacede oxygen levels..."). İki farklı mimari
  denendi (tek-blok çeviri ve satır-satır çeviri) — ikisi de aynı temel
  soruna düştü, satır-satır olan hatta bir örnekte daha kötü sonuç verdi
  (halüsinasyonla üretilmiş "Doğrusuz" gibi var olmayan bir kelime). Bu,
  artık prompt mühendisliğiyle değil, ancak daha büyük/özel bir çeviri
  modeliyle çözülebilecek bir kapasite sınırı olarak kabul edildi.
- **Bazı dar/çok-kavramlı sorularda yanlış-negatif retrieme riski.**
  Örnek: "Boru hattını hangi metal korur?" sorusunda model, doğru bilgi
  (magnesium/zinc) bağlamda mevcut olmasına rağmen bazen sadece reddediyor
  — artık yanlış bilgi uydurmuyor (bu kısım düzeltildi) ama doğru cevabı da
  her zaman çıkaramıyor. Kök neden muhtemelen bu spesifik dokümanın
  chunk'lanma şekli + küçük modelin çok yakın iki kavramı (ölçüm elektrodu
  vs. koruyucu metal) ayırt etmekte zorlanması.
- **CPU-only inference.** RTX 5080 GPU makinede mevcut olmasına rağmen, bu
  makinede test edilen hiçbir model (phi-3.5-mini, qwen2.5-7b,
  qwen3-embedding-0.6b) için Foundry Local kataloğunda bir GPU
  execution-provider varyantı sunulmuyor — sadece CPU. Etkileşimli
  kullanım için yeterince hızlı, ama GPU'nun potansiyeli kullanılmıyor.
- **"Auto" dil modu %100 güvenilir değil.** Türkçe algılama artık aksan
  işaretsiz yazımı da (`hattini`, `icin`) yakalıyor, ama bu hâlâ bir
  sezgisel (heuristic) yöntem — üçüncü bir dil veya beklenmeyen bir yazım
  şekli kaçabilir.
- **Oturumlar arası kalıcı sohbet geçmişi yok** (bilinçli kapsam dışı
  bırakma, iki kaynak dökümanın da zorunlu kılmadığı bir özellik).

## 8. Test Kapsamı Özeti

- **75 pytest birim/entegrasyon testi**, 7 dosyada: tokenizer, chunker,
  doküman okuyucular (PDF/DOCX), güvenlik (path traversal, uzantı/boyut
  limiti), vektör deposu (TF-IDF + hibrit arama), chat engine (event akışı,
  iki-aşamalı çeviri, kesinti/tekrar algılama, geri-dönüş mekanizmaları),
  foundry client (tekrar-döngüsü algılama, sıcak yeniden deneme).
- **5 turluk canlı Q&A test döngüsü** (aynı 10 soru, gerçek model, gerçek
  retrieval): v3 (ilk bulgular) → v4 (kök neden düzeltmeleri, çoğu başarılı)
  → v5 (çeviri mimarisi denemesi, geri alındı). Ham çıktılar
  `qa_test_results.json` / `qa_test_results_v4.json` /
  `qa_test_results_v5.json` içinde saklı, tam detay `TEST_REPORT.md`'de.

## 9. Sonuç

Proje, her iki kaynak dökümanın da temel gereksinimlerini karşılıyor ve
birçok noktada (hibrit retrieval, çok dillilik, ek doküman formatları, test
kapsamı, gerçek referans doküman sayısı) onların önerdiği asgari düzeyin
üzerine çıkıyor. Kalan sınırlamalar (çeviri akıcılığı, nadir yanlış-negatif
retrieme durumları, CPU-only inference) birden fazla iterasyonla
araştırıldı, kısmen iyileştirildi, ve çözülemeyen kısımları bu raporda ve
`TEST_REPORT.md`'de saklanmadan belgelendi. Bu, projenin şu anki, son
(final) durumudur.
