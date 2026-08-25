# Test Raporu

> Docx müfredatının "Phase 3 — Functional Testing" gereksinimini karşılar:
> cevaplanabilir/cevaplanamaz sorgular, sonuçların doğruluğu, ve bulunan
> hataların kaydı. Bu rapor iki bölümden oluşuyor: (1) otomatik `pytest`
> birim test paketi, (2) geliştirme sürecinde gerçek modelle (Foundry Local)
> çalıştırılan canlı sorgu testleri.
>
> **Not:** Sohbet modeli geliştirme sırasında `phi-3.5-mini` → `qwen2.5-7b`
> olarak değiştirildi (bkz. madde 13). Aşağıdaki 1-12 numaralı testler
> orijinal `phi-3.5-mini` ile yapıldı ve bu modelin ciddi bir "repetition
> collapse" (sonsuz tekrar) sorununu belgeliyor; 13. madde bu sorunun nasıl
> çözüldüğünü (model değişikliği) gösteriyor.

## 1. Otomatik Test Paketi (pytest)

```
75 passed in 0.25s
```

| Dosya | Test sayısı | Kapsam |
|---|---|---|
| `test_tfidf.py` | 8 | Tokenizer (İngilizce + Türkçe + karışık-dilli), IDF, cosine similarity |
| `test_chunker.py` | 6 | Front-matter ayrıştırma, chunk bölme/overlap, başlık türetimi |
| `test_document_readers.py` | 8 | PDF/DOCX metin çıkarımı, başlık stili → Markdown H1 dönüşümü, bozuk dosya işleme |
| `test_security.py` | 11 | Uzantı whitelist, path traversal, boyut limiti, dosya adı doğrulama |
| `test_vector_store.py` | 10 | TF-IDF arama, hibrit (TF-IDF+semantik) arama, eski şema migration |
| `test_chat_engine.py` | 15 | Event akışı, zarif bozulma, boş sorgu, iki-aşamalı çeviri, kesinti notu izolasyonu, çeviri geri-dönüşü, talimat-yankı temizleme |
| `test_foundry_client.py` | 17 | Tekrar-döngüsü (karakter/kelime/cümle seviyesi) algılama, sıcak yeniden deneme, gerçek `stream_chat` akışının mock istemciyle testi |
| **Toplam** | **75** | **75/75 yeşil** |

Çalıştırmak için: `.venv\Scripts\python -m pytest tests/ -v`

## 2. Canlı Sorgu Testleri (gerçek Foundry Local modeliyle)

Bu testler otomatik test paketinin *dışında*, geliştirme sırasında gerçek
modelle (simülasyon değil) elle çalıştırıldı ve çıktılar gözlemlendi.

| # | Sorgu | Senaryo | Çıktı Özeti | Doğruluk |
|---|---|---|---|---|
| 1 | "How do I detect a gas leak?" | Temel pipeline testi (TF-IDF) | "Gas Leak Detection Procedure" dokümanından doğru, güvenlik uyarılı, adım adım yanıt; kaynak doğru atıflandı | ✅ Doğru |
| 2 | Embedding hizalama testi: İng. soru vs. Türkçe çevirisi / karışık-dilli / alakasız Türkçe cümle | Diller-arası araştırma (Faz 2 öncesi) | Benzerlik skorları: çeviri 0.616, karışık-dilli 0.686, alakasız 0.311, aynı-dil referans 0.840 | ✅ Model diller arası anlamı doğru ayırt ediyor |
| 3 | "How do I detect a gas leak?" (Streamlit AppTest üzerinden) | Tam UI entegrasyon testi | Sohbet arayüzünde doğru akış: kullanıcı mesajı → streaming yanıt → kaynak paneli | ✅ Doğru, hatasız |
| 4 | "There is a strange odour near the wellhead and I feel dizzy, is this dangerous?" | Hibrit retrieval değer testi (kelime örtüşmesi düşük/yok) | "H2S Monitoring" (semantik 0.475) + "Gas Leak Detection" (**tfidf 0.0**, semantik 0.499) doğru bulundu; yanıt doğru şekilde H2S zehirlenmesi teşhisi koydu, güvenlik adımları verdi | ✅ Doğru — TF-IDF tek başına bu dokümanı **hiç bulamazdı** |
| 5 | Dosya yükleme + `st.rerun()` davranışı | Sağlamlık/regresyon testi (AppTest) | İlk denemede 300s+ zaman aşımı (sonsuz rerun döngüsü tespit edildi) → düzeltme sonrası 0.5s'de tamamlandı, tekrar çalıştırmada yeniden işlemedi | 🐛 **Hata bulundu ve düzeltildi** (session_state ile idempotency) |
| 6 | "What should I do if the compressor is vibrating abnormally?" (İng.) → yüklenen **Türkçe** DOCX belgesine karşı, `response_language=Turkish` | Diller-arası retrieval + zorunlu yanıt dili (düzeltme öncesi) | Doğru belge bulundu (**tfidf 0.0**, semantik 0.52) ama yanıt **İngilizce** geldi — dil talimatı görmezden gelindi | 🐛 **Hata bulundu**: retrieval doğru, dil zorlama başarısız |
| 7 | Aynı sorgu, prompt düzeltmesi sonrası (talimat hem sistem promptunda hem soruya bitişik) | Diller-arası retrieval + zorunlu yanıt dili (düzeltme sonrası) | Yanıt büyük ölçüde **Türkçe** geldi (özet, güvenlik uyarıları, adımlar); bazı başlıklar İngilizce kaldı, gramer küçük modelin sınırları nedeniyle kusurlu | ✅ Düzeltildi — **kısmi kalite notu**: küçük model (phi-3.5-mini) çeviri kalitesi sınırlı, prompt hatası değil |
| 8 | DOCX "Heading 1" stilli başlık çıkarımı | Doküman formatı testi | Word başlığı otomatik `# Başlık` biçimine çevrildi, chunker doğru başlığı türetti (dosya adına düşmedi) | ✅ Doğru |
| 9 | "what is flare system" / "flare sistemi ne demek..." / **"ateşleme sistemi ne işe yarıyor"** | Kullanıcı canlı testi (gerçek kullanım sırasında) | İlk iki soru doğru yanıtlandı (biri İng., biri karışık-dilli); üçüncüsünde model **tekrar döngüsüne girip sonsuz "0" üretmeye başladı** | 🐛 **Hata bulundu**: küçük modelin bilinen "repetition collapse" arızası, `frequency_penalty` ayarlanmamıştı |
| 10 | Aynı "ateşleme sistemi" sorgusu, düzeltme sonrası (`frequency_penalty=0.4` + tekrar-algılama güvenlik ağı) | Regresyon testi | `frequency_penalty` tek başına döngüyü tamamen önlemedi, ama **güvenlik ağı döngüyü ~600 karakterde tespit edip kesti**, kullanıcıya net bir not bıraktı; ikinci sorgu ("flare sistemi ne demek") temiz ve doğru yanıtlandı | ✅ Düzeltildi — döngü artık sınırsız sürmüyor, açıkça işaretleniyor |
| 11 | "sogukhava operasyonlarıne demek ne yapılması lazım" (yazım hataları/bitişik kelimelerle) → sonraki turda "corrosion inspection hakkında bilgi ver" | Kullanıcı canlı testi — çok turlu sohbet | Model kendi geçmişindeki (bir önceki turdan kalan) "(Response stopped early...)" notunu **taklit ederek** ilgisiz bir soruya sahte "(Response stopped early — the user's question ... )" metni üretti; ayrıca aynı turda seçili "Turkish" yanıt dili de uygulanmadı | 🐛 **Hata bulundu**: kesinti notu `full_answer` içine karışıp `session_state.history`'e kaydolmuş, sonraki turlarda modele geçmiş olarak tekrar besleniyordu — hem sahte metin üretimine hem dil talimatının bozulmasına yol açtı |
| 12 | Aynı senaryo, düzeltme sonrası (`notice` ayrı bir event tipi oldu, `model_history` sadece role/content içeriyor) | Regresyon testi | Birim test ile doğrulandı: kesinti notu artık `token` metnine hiç karışmıyor, `build_messages`'a giden geçmiş sadece gerçek asistan cevabını içeriyor | ✅ Düzeltildi (bkz. `test_ask_emits_a_separate_notice_event_when_response_is_truncated`) |
| 13 | "corrosion inspection hakkında bilgi ver" (Turkish), "How do I detect a gas leak?" (Turkish), "flare sistemi ne demek" (English), "ateşleme sistemi ne işe yarıyor" (Auto) — **aynı 4 sorgu, model değişikliği öncesi/sonrası karşılaştırma** | Kök neden araştırması: `phi-3.5-mini` bu sorguların çoğunda **tutarlı/tekrarlanabilir şekilde** (4/4 aynı sonuç) tekrar döngüsüne giriyordu — hem tek-aşamalı hem iki-aşamalı (grounding+çeviri) mimaride, hem düşük hem yüksek `frequency_penalty`/`presence_penalty` değerlerinde, hem "sıcak yeniden deneme" ile hem de değneme yaklaşan çözümlerle. Denenenler: iki aşamalı üretim (grounding ayrı, çeviri ayrı çağrı), erken tekrar tespitinde sıcaklık artırarak yeniden deneme, `frequency_penalty` 0.4→0.8, `presence_penalty` 0→0.4, tekrar-algılamayı karakter→kelime→cümle seviyesine (220 karaktere kadar) genişletme, başarısız çeviride orijinal cevaba geri dönüş, talimat-yankısı temizleme. **Sonuç:** `qwen3-8b` denendi (yeni bir sorun getirdi — varsayılan "thinking" modu, düşünce bloğu da döngüye girdi); `qwen2.5-7b`'ye geçildi. | ✅ **Kök neden çözüldü** — `qwen2.5-7b` ile aynı 4 sorgu + "corrosion inspection" sorgusu 3 kez daha tekrarlandı, **7/7 denemede hiç tekrar döngüsü yaşanmadı** (`NOTICE: None`). Çeviri akıcılığı hâlâ mükemmel değil (kelime sırası bazen garip) ama bu bir *stabilite* değil *akıcılık* sorunu. |
| 14 | Kullanıcı gözlemi: "korozyon incelemesi prosedürlerini sadece anlat" (Turkish, `qwen2.5-7b`) yanıtında kelimeler arasına anlamsız **"ağıro"** kelimesi serpiştirilmiş bulundu — mevcut tekrar-dedektörü bunu **yakalayamadı** (aynı blok değil, değişen içerik arasına serpiştirilmiş tek kelime). Kullanıcı ayrıca kaynak `data/docs/` dosyalarının çok kısa/şablonik (656 satır, 3730 kelime, çoğu 1 chunk) olduğunu ve bunun sorunlara katkıda bulunabileceğini öne sürdü. | Kök neden + iyileştirme | 20 dokümanın tamamı doğal düzyazıyla zenginleştirildi (656→1383 satır, 3730→10766 kelime, 23→65 chunk). Aynı sorgu + 3 önceki sorunlu sorgu tekrar denendi: **"ağıro" bozulması bir daha görülmedi**, hiçbir sorguda tekrar döngüsü oluşmadı. Diller-arası retrieval (H2S/wellhead çapraz sorgusu) zenginleştirilmiş içerikle de kusursuz çalıştı. | ✅ Doğrulandı — zengin içerik genel cevap kalitesini ve modelin kararlılığını gözle görülür şekilde iyileştirdi (kesin nedensellik iddia edilmiyor, ama sonuç olumlu) |

### Bulunan ve Düzeltilen Hatalar (özet)

1. **SQLite thread-safety hatası** (Faz 1) — `st.cache_resource` ile paylaşılan bağlantı farklı thread'lerden çağrılınca çöküyordu → `check_same_thread=False` + `threading.Lock`.
2. **Yanlış `.gitignore` yolu** (Faz 1→2 arası) — `data/rag.db` yazıyordu, gerçek dosya `data/knowledge.db` idi → veritabanı yanlışlıkla commit'e girebilirdi, düzeltildi.
3. **Sonsuz rerun döngüsü** (Faz 2) — dosya yükleme sonrası `st.rerun()`, aynı dosya widget'ta kaldığı sürece kendini tekrar tetikliyordu → `session_state` ile işlenen dosya takibi eklendi.
4. **Dil zorlama talimatı göz ardı ediliyordu** (Faz 2) — tek talimat bağlamın sonunda kayboluyordu → sistem promptu + soru-bitişik ikili pekiştirme ile düzeltildi.
5. **TF-IDF tokenizer Türkçe karakterleri siliyordu** (Faz 2) — `[a-z0-9]` yerine Unicode-güvenli `\w` kullanılarak düzeltildi.
6. **Repetition collapse (sonsuz tekrar) hatası** (kullanıcı canlı testinde bulundu) — model bazı belirsiz/zayıf eşleşen sorgularda aynı karakteri sonsuz tekrarlamaya başlıyordu → `frequency_penalty`/`presence_penalty` eklendi + `src/foundry_client.py`'de tekrar-döngüsü algılanınca akışı erken kesen bir güvenlik ağı eklendi (`_is_runaway_repetition`).
7. **Kesinti notunun sohbet geçmişini kirletmesi** (kullanıcı canlı testinde bulundu, #6'nın doğrudan sonucu) — bizim eklediğimiz "(Response stopped early...)" metni asistan cevabına karışıp geçmişe kaydoluyor, model bunu sonraki turlarda görüp taklit ediyordu (hatta dil talimatını bozacak kadar kafası karışıyordu) → kesinti notu artık ayrı bir `"notice"` event'i, asla `build_messages`'a giden geçmişin parçası olmuyor.
8. **`phi-3.5-mini` modelinin tekrarlanabilir repetition-collapse arızası** (kullanıcı canlı testinde defalarca bulundu) — prompt/parametre mühendisliğiyle sınırlanabildi ama kök nedeni çözülemedi → **model `qwen2.5-7b` ile değiştirildi**, sorun tamamen ortadan kalktı (bkz. madde 13).
9. **Serpiştirilmiş-kelime bozulması ("ağıro")** (kullanıcı canlı testinde bulundu) — tekrar-dedektörünün yakalayamadığı yeni bir bozulma türüydü → doğrudan kod düzeltmesi yerine kaynak dokümanların zenginleştirilmesi (madde 14) bu bozulmanın tekrarlanmamasını sağladı.
10. **Bölüm başlıkları referans repoyla birebir aynıydı** (kullanıcı fark etti) — "Purpose/Safety Warnings/Procedure/Reference" 20 dokümanın tamamında referans repodan aynen kalmıştı → tüm dosyalarda `## Overview` / `## Key Safety Precautions` / `## Working Procedure` / `## Source Standard` / `## Inspection Steps` olarak yeniden adlandırıldı (içerik/anlam değişmedi, sadece başlık metni). Yeniden indekslendi (66 chunk), 75/75 test hâlâ yeşil.
11. **Dosya adları da referans repoyla aynıydı** (kullanıcı istediği) — 20 dosyanın tamamı yeniden adlandırıldı (örn. `corrosion-inspection.md` → `corrosion-assessment.md`).
12. **Kullanıcı isteği: yakın-konulu ~20 gerçek makale ham haliyle eklensin** (normal PDF yükleme gibi, bizim özel formatımıza sokulmadan) — `WebFetch` aracının içeriği özetleyip yorumladığı (gerçek verbatim metin vermediği, hatta ABD federal eserlerini yanlışlıkla "telifli" sanıp reddettiği) tespit edildi → doğrudan `curl` ile ham HTML çekilip Python `html.parser` ile site navigasyonu/footer temizlenerek gerçek metin çıkarıldı. 20 gerçek OSHA/EPA sayfası (ABD federal hükümet eseri, kamu malı, telif sorunu yok) eklendi — `## Overview` gibi bizim şablonumuza **sokulmadı**, sadece `Source:` atıf satırı + ham metin. Toplam corpus: **40 doküman, 189 chunk**. Karma corpus'ta (zenginleştirilmiş + ham) 3 farklı sorgu test edildi, hepsi doğru kaynaktan, döngüsüz cevaplandı.

## 3. 10 Soruluk Sistematik Q&A Testi — v3 (bulgular, düzeltme öncesi)

> **v3 notu:** Bu bölüm, kullanıcının bir ChatGPT konuşmasında dile getirdiği
> "cevaplar yeterince temellendirilmiş mi (grounded)?" eleştirisi üzerine
> canlı sistem karşısında çalıştırılan 10 soruluk bir testin **sadece
> bulgularını** belgeler. Aşağıdaki hatalar bu sürümde **kasıtlı olarak
> düzeltilmedi** — önce mevcut durumun tam ve dürüst bir kaydı tutuluyor, asıl
> düzeltmeler bir sonraki sürümde (v4) yapılacak ve yeniden test edilecek.
> Ham çıktılar `qa_test_results.json` içinde saklanıyor.
> Model: `qwen2.5-7b`, `response_language="Auto"`.

| # | Soru | Dil | Doğruluk | Not |
|---|---|---|---|---|
| 1 | How do I detect a gas leak? | EN | ✅ Doğru | Temiz, tam, doğru kaynak |
| 2 | What should I do if I discover a fire on site? | EN | ✅ Doğru | İçerik doğru; referans satırı "Standard Operations Manual OM-01, Section 9" — kaynak dokümanda böyle bir metin yok, model muhtemelen uydurmuş (bkz. madde 17) |
| 3 | Boru hattini hangi metal korur? | TR | ❌ **Yanlış** | "Copper/copper-sulphate reference electrode korur" dedi — bu potansiyel **ölçüm elektrodu**, korumayı sağlayan kurban metal (magnezyum/çinko) değil. Doğru doküman bulunmuş (skor 0.204/0.166) ama yanlış bilgi çekilmiş; cevabın sonuna ayrıca anlamsızca "bilgi mevcut değil" eklenmiş |
| 4 | What is a blowout preventer used for? | EN | ⚠️ Kısmi | Summary/Guidance doğru; ama "Safety Warnings: This information is not available in the local knowledge base." — ret cümlesi yanlış bölüme, sanki o bölümün cevabıymış gibi yerleştirilmiş |
| 5 | confined space entry icin oksijen limitleri nedir? | TR | ❌ **Yanlış** | Doğru doküman ilk sırada bulundu (skor 0.523, içinde net "19.5%–23.5%" yazıyor) ama model yine de tüm cevap yerine sadece "bilgi mevcut değil" dedi — retrieval doğru, üretim yanlış (false negative) |
| 6 | What are the OSHA exposure limits for hydrogen sulfide? | EN | ✅ Doğru | 20 ppm / 10 ppm rakamları OSHA kaynağıyla birebir eşleşiyor |
| 7 | How often should a gas detector be calibrated? | EN | ⚠️ Kısmi | Doğru ve tam cevap (günlük bump-test, 6 ayda bir tam kalibrasyon) verildikten **sonra** cevabın en sonuna anlamsızca "bilgi mevcut değil" eklenmiş |
| 8 | What is the acceptable pressure drop during a pipeline pressure test? | EN | ✅ Doğru | En temiz cevap, %0.5 rakamı birebir doğru |
| 9 | What is photosynthesis? (kontrol sorusu, kapsam dışı) | EN | ✅ Doğru | Doğru şekilde reddetti, halüsinasyon yok — istenen davranış |
| 10 | What PPE is required in H2S areas? | EN | ✅ Doğru | Doğru, temiz |

**Özet: 10 sorudan 6'sı tam doğru, 2'si içerik doğru ama hatalı yerleşmiş ret
cümlesi içeriyor, 2'si gerçekten yanlış/temelsiz.**

### Tespit Edilen Yeni Hata Kategorileri (v3 — henüz düzeltilmedi)

13. **Yanlış bilgi çıkarımı — kavram karışıklığı** (Q3) — model, doğru kaynak
    dokümanı bulmasına rağmen dokümandaki iki farklı kavramı (ölçüm için
    kullanılan bakır/bakır-sülfat referans elektrodu vs. asıl korumayı
    sağlayan kurban anot metali) birbirine karıştırıp yanlış metali cevap
    olarak verdi.
14. **Yanlış-negatif ret ("false negative")** (Q5) — retrieval doğru belgeyi
    ilk sırada bulmasına rağmen (skor 0.523) model yine de "bu bilgi mevcut
    değil" dedi, doğru cevap (19.5%–23.5%) bağlamda açıkça mevcutken.
15. **Yanlış yere yerleştirilmiş / gereksiz tekrarlanan ret cümlesi** (Q4,
    Q7) — model doğru cevabı verdikten sonra veya bir alt bölümün içine,
    sanki o bölümün içeriğiymiş gibi `"This information is not available in
    the local knowledge base."` cümlesini ekliyor. Sistem promptundaki
    "bağlamda yoksa tam olarak şunu söyle" talimatının **bölüm bazında**
    yanlış tetiklenmesinden kaynaklandığı düşünülüyor.
16. **Türkçe/Auto modda güvenilirlik farkı** — bu 10 soruluk örnekte 2
    Türkçe sorunun **ikisi de** (Q3, Q5) yanlış/temelsizken, 8 İngilizce
    sorunun hiçbiri tamamen yanlış değildi (en kötüsü kısmi/kozmetik hata).
    Örneklem küçük (n=2 Türkçe) ama önceki canlı testlerle tutarlı bir
    örüntü.
17. **Olası referans/başlık sızıntısı** (Q2) — cevaptaki "Standard
    Operations Manual OM-01, Section 9" ifadesinin kaynağı belirsiz; ilgili
    dokümanda böyle bir metin yok. Kök neden henüz araştırılmadı (uydurma
    referans mı, yoksa `## Source Standard` başlığından esinlenen bir
    üretim mi, netleşmedi).

## 4. v3 Bulgularının Düzeltilmesi — v4

> Bu bölüm, madde 3'te (v3) belgelenen 13-17 numaralı hataların kök nedenini
> araştırıp düzelttiğimiz ve **aynı 10 soruyla yeniden test ettiğimiz**
> süreci belgeler. Ham çıktılar `qa_test_results_v4.json` içinde saklanıyor.

### Kök neden analizi

- **Madde 13/16 (yanlış metal / Türkçe-Auto güvenilirlik farkı):** Kod
  incelemesi, "Auto" modda `response_language` zaten `None`'a düştüğü için
  iki-aşamalı (grounding+çeviri) mimarinin hiç devreye girmediğini, tek
  aşamalı üretimin ise dil talimatını `SYSTEM_PROMPT` üzerinden (bağlamdan
  çıkarım yapmakla AYNI ANDA) taşıdığını ortaya çıkardı — tam olarak daha
  önce "zorunlu dil" için terk ettiğimiz, tek geçişte iki görev yükleme
  hatası. Ayrıca doğrudan retrieval testiyle (embedder gerçekten `init()`
  edilerek) doğru parça ("How the System Works", magnesium/zinc) bağlama
  giriyordu ama `TOP_K=3` ile bazı sorgularda kırpılma riski taşıyordu.
- **Madde 14/15 (yanlış-negatif / yanlış yerleşmiş ret cümlesi):**
  `SYSTEM_PROMPT`'taki ret talimatı "bağlamda yoksa tam olarak şunu söyle"
  şeklindeydi ama bunun **tüm cevap için mi yoksa her bölüm için mi**
  geçerli olduğu belirtilmemişti — model bazen bunu bölüm-bazlı bir dolgu
  cümlesi gibi kullanıyordu.
- **Madde 17 (referans sızıntısı):** Modelin "Reference" satırını üstteki
  `[Source N] <title>` etiketinden değil, parça içeriğinin kendi
  `## Source Standard` bölümünden kurduğu, başlığın son kelimesini
  ("Standard") bitişik metinle birleştirdiği tespit edildi.

### Uygulanan düzeltmeler

1. `config.py`: `TOP_K` 3 → 5 (zayıf/dağınık dokümanlarda doğru parçanın
   kırpılma riskini azaltmak için).
2. `src/prompts.py`: `SYSTEM_PROMPT` güncellendi —
   - Ret cümlesinin **tüm cevap** kararı olduğu, asla bölüm dolgusu veya
     cevaptan sonra eklenen bir not olarak kullanılamayacağı açıkça belirtildi.
   - Modelin bir şeyi "yok" demeden önce **tüm bağlamı** taraması gerektiği,
     istenen detayın soru ile birebir aynı başlık altında olmayabileceği
     eklendi.
   - "Reference" satırının parça içeriğinden değil, üstteki `[Source N]
     <title>` etiketinden alınması gerektiği eklendi.
   - Artık grounding aşaması **her zaman İngilizce** yanıt veriyor; dil
     dönüşümü tamamen çeviri aşamasına bırakıldı (aşağıya bakın).
3. `src/chat_engine.py`: "Auto" modu artık soruyu **Türkçe olarak algılarsa**
   (`_looks_turkish()`), zorunlu dil seçimiyle birebir aynı, kanıtlanmış
   iki-aşamalı (grounding + ayrı çeviri) akışından geçiyor — önceden sadece
   açıkça seçilmiş bir dil için çalışan bu akış artık Auto+Türkçe soru için
   de devrede.
4. `src/tfidf.py`: `TURKISH_STOPWORDS` genişletildi (`hangi`, `nedir`,
   `değil`, `var`, `yok`, `kaç`, `mıdır` ... eklendi) ve `_looks_turkish()`
   artık aksan işaretsiz yazılmış Türkçe'yi de (`hattini`, `icin` gibi)
   ASCII-katlama ile yakalıyor — ilk düzeltmenin canlı testte tam da bu
   yüzden (test sorguları aksansız yazılmıştı) devreye girmediği görüldü ve
   ikinci bir iterasyonla düzeltildi.

pytest paketi her adımdan sonra çalıştırıldı: **75/75 yeşil**, regresyon yok.

### v3 → v4 Karşılaştırması

| # | Soru | v3 | v4 | Durum |
|---|---|---|---|---|
| 2 | Fire response | Referansta uydurma "Standard Operations Manual" | Referans artık doğru `[Source N]` başlığından ("Fire Response ...") — biraz tekrarlı ama artık uydurma yok | ✅ Düzeltildi |
| 3 | Boru hattini hangi metal korur? (TR) | Yanlış metal (copper/copper-sulphate) + kendiyle çelişen "mevcut değil" eki | Artık yanlış bilgi **uydurmuyor** — dürüstçe reddediyor, ve Auto+Türkçe artık doğru şekilde çeviri aşamasından geçip Türkçe yanıt veriyor — ama hâlâ doğru cevabı (magnesium/zinc, bağlamda mevcut) çıkaramıyor | ⚠️ **Kısmen düzeltildi** — artık yanlış bilgi vermiyor, ama hâlâ gereksiz yere reddediyor (aşağıya bakın) |
| 4 | Blowout preventer | "Safety Warnings: mevcut değil" (yanlış bölüme dolgu) | Safety Warnings bölümü artık gerçek, bağlamdan gelen bir uyarıyla dolu, ret cümlesi hiç yok | ✅ Düzeltildi |
| 5 | Confined space O2 limits (TR) | Doğru bağlam bulunmasına rağmen tümden "mevcut değil" | Doğru rakamlar (19.5%–23.5%) artık veriliyor, Türkçeye çevriliyor | ✅ **Ana hata düzeltildi** (doğru bilgi artık geliyor) — ama çeviri akıcılığı zayıf, bilinen çeviri-kalitesi sınırına giriyor (aşağıya bakın) |
| 7 | Gas detector calibration | Doğru cevaptan sonra anlamsız "mevcut değil" eki | Ek yok, temiz cevap | ✅ Düzeltildi |

### Kalan/açık noktalar (v4 sonrası)

- **Q3 hâlâ tam çözülmedi.** Model artık yanlış metal uydurmuyor (iyi), ama
  doğru metali (magnesium/zinc) — bu bilgi verilen bağlamda mevcut olmasına
  rağmen — çıkaramıyor, sadece reddediyor. Bunun kalan nedeni muhtemelen
  chunk'lama: ilgili parça ("## How the System Works...") bir önceki
  bölümün ortasından başlayıp ("...before opening the box.") yarım kalan
  bir cümleyle bitiyor (`CHUNK_SIZE=200` kelime sınırı). Daha büyük bir
  model veya farklı bir chunklama stratejisi gerekebilir — bu oturumda daha
  fazla prompt mühendisliğiyle zorlanmadı (bkz. "kalan sınır" notu).
- **Çeviri akıcılığı (Q3, Q5'te görülen İngilizce-Türkçe karışık cümleler,
  örn. "Do confined spacede oxygen levels top, middle... tested olabilir")
  yeni bir hata DEĞİL** — bu, README'de zaten belgelenmiş olan, önceden de
  bilinen çeviri-akıcılığı sınırlaması. Fark şu: Auto modun artık Türkçe
  soruları doğru şekilde çeviri aşamasından geçirmesi sayesinde bu sınırlama
  daha sık görünür hale geldi (önceden bu sorular ya İngilizce kalıyor ya da
  yanlış-negatif reddediliyordu, akıcılık sorunu hiç ortaya çıkmıyordu).
  Kullanıcının daha önce ayrı bir aşama olarak ele almayı tercih ettiği
  ("2. aşama olarak fixlemeye çalışırız") bu konuya bu raporda dokunulmadı.

## 5. Çeviri Akıcılığı Girişimi — v5 (kısmi/karışık sonuç, dürüstçe raporlanıyor)

> Bu bölüm, v4'te "bilinen sınır" olarak bırakılan çeviri-akıcılığı sorununu
> (İngilizce-Türkçe karışık çıktı) doğrudan çözmeye yönelik bir mimari
> değişikliği ve onun **gerçek, dürüst sonucunu** belgeler. Sonuç net bir
> "düzeltildi" değil — bu bölüm bunu olduğu gibi yansıtıyor.

### Denenen değişiklik

`src/chat_engine.py`'ye yeni bir `_translate_draft()` metodu eklendi:
tek seferde bütün cevabı çevirmek yerine, cevabı satır satır bölüp **her
satırı ayrı bir çeviri çağrısıyla** çeviriyor (madde listesi işaretini
`- `/`1. ` ayırıp çeviriye karışmasını önleyerek), başarısız/döngüye giren
bir satırı da tüm cevabı iptal etmek yerine sadece o satırı orijinal
İngilizce haliyle bırakarak. Gerekçe: daha önce grounding+çeviriyi tek
geçişten ikiye ayırmanın işe yaraması ile aynı mantık — küçük modele daha
küçük, tek bir görev vermek. `TRANSLATION_SYSTEM_PROMPT` de tek bir kısa
cümle/ifade çevirisine göre yeniden yazıldı, karışık dil ve "Do" gibi
anlamsız eklerin açıkça yasaklandığı bir talimatla. İki yeni birim test
eklendi (`test_ask_translates_a_multiline_draft_line_by_line`,
`test_ask_falls_back_per_line_when_one_line_fails_to_translate`) — 77/77
pytest yeşil.

### Gerçek sonuç: aynı 10 soru tekrar çalıştırıldı (`qa_test_results_v5.json`)

| # | Soru (TR) | v4 çıktısı | v5 çıktısı | Değerlendirme |
|---|---|---|---|---|
| 3 | Boru hattini hangi metal korur? | "Bu information local knowledge base'de available değil." | "Bu information, local knowledge base'de available değil." | Neredeyse aynı — hâlâ karışık dil, düzelme yok |
| 5 | confined space entry icin oksijen limitleri nedir? | "Do confined spacede oxygen levels top, middle... tested olabilir safe limit within do confirm." | "Doğrusuz bir space for oxygen limit 19,5% to 23,5%." / "...limiti aşıi oxygen levelsi içindeki atık içindeki iskişı an be dangerous." | **Daha kötü** — "Doğrusuz" gibi anlamsız/var olmayan bir kelime üretti, cümle daha da anlaşılmaz |

**Dürüst değerlendirme: satır-satır çeviri mimarisi akıcılığı düzeltmedi —
Q5'te belirgin şekilde daha kötü bir sonuç üretti.** Muhtemel neden: tek
satıra indirgenen kısa parçalar bağlamlarını (önceki/sonraki cümle) kaybediyor,
bu da modelin daha az tutarlı, daha fazla halüsinasyonlu kelime üretmesine yol
açıyor. Bu, prompt/mimari mühendisliğiyle çözülebilecek bir sorun olmaktan çok,
`qwen2.5-7b`'nin İngilizce→Türkçe çeviride **temel bir yetenek sınırı**
olduğunu doğruluyor — hem tek-blok hem satır-satır yaklaşım aynı şekilde
karışık-dil üretimine düşüyor.

**Karar: geri alındı.** Satır-satır mimarisi akıcılığı düzeltmediği ve bir
soruyu ölçülebilir şekilde kötüleştirdiği için, `src/chat_engine.py` ve
`src/prompts.py` tekrar v4'teki tek-blok çeviri çağrısına döndürüldü (iki
geçici birim testi de kaldırıldı, paket tekrar 75/75 yeşil). Sistem artık
**v4'teki en iyi doğrulanmış hâliyle** duruyor: Auto+Türkçe algılama,
`TOP_K=5`, ret-cümlesi netleştirmesi ve referans-etiketi düzeltmesi kalıcı;
çeviri akıcılığı ise README'de zaten belgelenmiş, kabul edilmiş bir bilinen
sınır olarak bırakıldı — bu oturumda iki farklı mimari (tek-blok, satır-
satır) denendi, ikisi de aynı temel kapasiteye çarptı, bu nedenle projenin
şu anki kapsamında (harici çeviri kütüphanesi/modeli eklemeden) daha fazla
prompt mühendisliğiyle zorlanmadı.

## 6. Chunking & Vektörleme Tam Denetimi (kullanıcı talebiyle)

> Kullanıcı, "chunking ve vektörleme işlemlerinin doğru yapılıp yapılmadığını
> tam olarak kontrol etmemi" istedi. `src/chunker.py`, `src/ingest.py`,
> `src/embedder.py`, `src/vector_store.py` satır satır incelendi, canlı
> veritabanı (`data/knowledge.db`, 189 chunk) sorgulanarak gerçek veri
> üzerinde doğrulandı. Dört gerçek bulgu çıktı, hepsi düzeltildi:

1. **Hibrit skorlamada tutarsızlık** (`src/vector_store.py::search`) —
   hibrit modda embedding'i olmayan bir chunk, tam ağırlıklı (1.0×) TF-IDF
   skoru alıyordu; embedding'i olan chunk'lar ise 0.5× ağırlıkla
   kısıtlanıyordu. Canlı veritabanında (tüm 189 chunk embedding'liydi)
   tetiklenmiyordu ama embedder geçici olarak kapalıyken eklenen yeni bir
   doküman bu tutarsızlığı canlıya taşıyabilirdi. → Embedding'i olmayan
   chunk artık semantic_score=0 kabul edilip aynı ağırlıklı formülle
   skorlanıyor (görüntüde hâlâ `None` olarak ayrı işaretleniyor). Regresyon
   testi eklendi.
2. **Chunk'lama paragraf/başlık yapısını düzleştiriyordu** (`chunk_text`) —
   `text.split()` + `" ".join()` tüm satır sonlarını tek boşluğa
   indirgiyordu, yani `## Source Standard\nOperations Manual OM-01...`
   chunk içeriğinde `"## Source Standard Operations Manual OM-01..."` olarak
   TEK SATIRDA depolanıyordu. Bu, daha önce QA testinde bulunan uydurma
   "Standard Operations Manual" referans hatasının kök nedenlerinden biri
   olarak değerlendirildi. → Chunk'lama artık başlıkları ve madde
   işaretli/numaralı liste öğelerini birer atomik blok olarak koruyor,
   bloklar arasına boş satır bırakıyor; bir blok `CHUNK_SIZE`'dan büyükse
   eski kelime-bazlı kesme mekanizmasına geri düşüyor.
3. **(2 numaralı düzeltme sırasında bulunan yeni hata) Çok satırlı liste
   öğeleri cümle ortasında bölünüyordu** — ilk uygulamada regex tabanlı blok
   ayırıcı, sarmalanmış (soft-wrap) bir madde işaretli öğenin sadece ilk
   fiziksel satırını yakalıyor, devam satırlarını ayrı bir "paragraf" bloğu
   olarak yetim bırakıyordu (örn. "- Never extinguish... can also be" / ayrı
   blok: "isolated; an unburned gas cloud..."). → Satır-bazlı bir durum
   makinesine geçildi: bir liste öğesi, sonraki boş satır/yeni
   madde/başlığa kadar tüm devam satırlarını kendi bloğuna topluyor.
4. **Ham referans dokümanlarında site-navigasyonu artığı çok küçük
   chunk'lar** — 8 adet 1-7 kelimelik "chunk" bulundu (örn. `"Overview"`,
   `"Workers' Rights"`, `"Well Control – Blowout Preventers"`) — OSHA
   sayfalarından kazınırken temizlenmemiş kenar-menü metni. Bunlar sadece
   gürültü değildi: **"Boru hattini hangi metal korur?" sorgusunda**
   `"Well Control – Blowout Preventers"` (4 kelime) chunk'ı skor 0.182 ile
   gerçek "Cathodic Protection Survey" chunk'larını (0.169/0.165) **geride
   bırakıyordu** — kısa/genel metinler embedding uzayında bazen yapay olarak
   yüksek benzerlik skoru alabiliyor. → `document_to_chunks`'a
   `_MIN_CHUNK_WORDS=8` eşiği eklendi (bir dokümanın TEK chunk'ıysa asla
   silinmiyor). Yeniden indeksleme: 256 → **248 chunk**, 8 chunk temizlendi.

**Doğrulama:** Tüm düzeltmelerden sonra corpus yeniden indekslendi
(40 doküman, 248 chunk, tamamının embedding'i var). Aynı "Boru hattini
hangi metal korur?" sorgusu tekrar çalıştırıldı: junk chunk artık top-5'te
değil; "Cathodic Protection Survey" chunk 0 ve 1 ilk iki sırada (0.169,
0.165) ve chunk 1 artık `## How the System Works` başlığını VE
"magnesium/zinc" cümlesini aynı, düzgün ayrılmış chunk içinde barındırıyor
— `TOP_K=5` ile bu bilginin modele gitmesi artık çok daha güvenilir
(canlı modelle yeniden test edilmedi, sadece retrieval katmanı doğrulandı).
79/79 pytest yeşil (4 yeni regresyon testi dahil).

## 7. Chunking Düzeltmesinin Tek Başına Etkisi — v6 (değişkeni izole eden test)

> Madde 6'daki chunking düzeltmelerinden sonra, **model değişkenini sabit
> tutarak** (hâlâ kanıtlanmış `qwen2.5-7b` ile) aynı 10 soru tekrar
> çalıştırıldı — amaç, Q3/Q5'teki iyileşmenin chunking'den mi yoksa olası bir
> model değişiminden mi geldiğini karıştırmadan görmek. Ham çıktılar
> `qa_test_results_v6.json`'da.

| # | v4 (eski chunking) | v6 (yeni chunking, aynı model) | Değişim |
|---|---|---|---|
| Q3 | "Bu information local knowledge base'de available değil." | **Birebir aynı metin** | ❌ Değişiklik yok |
| Q5 | Doğru rakam (19.5%-23.5%) ama karışık-dilli çeviri | Doğru rakam, hâlâ karışık-dilli çeviri (farklı ama eşit derecede bozuk cümleler) | ⚖️ Aynı seviyede — ne iyileşti ne kötüleşti |
| Q1,2,4,6,7,8,9,10 | Doğru | Doğru | Değişiklik yok (zaten iyiydi) |

**Dürüst sonuç: chunking düzeltmeleri Q3'ü çözmedi.** Retrieval katmanında
doğrulandığı gibi (madde 6) doğru chunk (`## How the System Works`,
magnesium/zinc içeren) artık top-2'de ve modele gidiyor — ama model yine de
bu bilgiyi çıkaramayıp reddediyor. Bu, retrieval/chunking'in ötesinde,
`qwen2.5-7b`'nin bu spesifik soruyu (birbirine yakın iki farklı "metal"
kavramını ayırt etme) işleme kapasitesiyle ilgili gerçek bir sınır olduğunu
doğruluyor. Chunking düzeltmeleri madde 6'da belirtilen kendi hedeflerinde
(skor tutarlılığı, başlık ayrımı, junk chunk temizliği) başarılıydı ama bu
belirli üretim-katmanı hatasını çözmedi — bu ayrımı net tutmak için burada
ayrıca belirtiliyor.

## 8. 30 Soruluk Retrieval-Only Denetim (kullanıcı talebiyle, sadece retrieval)

> Kullanıcı özellikle **LLM üretimini değil, sadece retrieval'i** test etmek
> istedi: "doğru chunk'lar geliyor mu" sorusuna kesin bir cevap için, tek-
> amaçlı (bileşik olmayan — "agentic katmanımız yok" uyarısıyla) 30 soru
> yazıldı: 10 kolay, 10 orta, 10 zor. Her soru, hedef doküman önce tam
> okunarak, doğru cevabın nerede olduğu bilinerek yazıldı. `VectorStore.search()`
> doğrudan çağrıldı (LLM yok), beklenen dokümanın ilk 5 sonuçta olup olmadığı
> kontrol edildi. Ham sonuçlar `retrieval_audit_results.json`'da.

**İlk sonuç (mevcut `0.5/0.5` ağırlıkla): 28/30** — kolay 10/10, orta 8/10,
zor 10/10.

### Bulunan 2 gerçek sorun

1. **"confined space" (entry kelimesi olmadan) sorgusu yanlış dokümanı
   öne çıkarıyordu.** "confined space icin oksijen limitleri nedir?" sorusu,
   gerçek sayısal cevabı (19.5%–23.5%) içeren kendi "Confined Space Entry"
   dokümanımız yerine, ID/gezinme menüsü tekrarları yüzünden anahtar-kelime
   yoğunluğu yapay olarak yüksek olan ama **gerçek cevabı hiç içermeyen**
   ham "Confined Spaces - Overview (OSHA)" sayfasını üstte gösteriyordu.
   İncelemede: doğru chunk'ın semantik skoru (0.616) aday havuzundaki EN
   YÜKSEK skordu — sorun TF-IDF'in ona verdiği düşük ağırlıktı (0.156),
   0.5/0.5 harmanın bunu yeterince telafi etmemesiydi.
   → **Düzeltme:** `HYBRID_TFIDF_WEIGHT`/`HYBRID_EMBEDDING_WEIGHT`
   `0.5/0.5` → `0.35/0.65`. Aynı 30 soruda hiçbir gerilemeye yol açmadan
   (kolay/zor hâlâ 10/10) orta kategoriyi 8/10 → 9/10 yaptı.
2. **Türkçe embedding kalitesi, aksan işaretlerine ve kullanılan kelimeye
   göre büyük ölçüde değişiyor.** "Yangin sondurucu secimi nasil yapilir?"
   (aksansız) sorusu, doğru chunk ile sadece **0.202** benzerlik skoru
   veriyor. Aynı soru düzgün aksanlarla ("Yangın söndürücü seçimi nasıl
   yapılır?") **0.397**'ye çıkıyor — neredeyse iki katı — ama İngilizce
   eşdeğeri ("How do I select the correct fire extinguisher?") hâlâ
   **0.725** ile çok daha güçlü. Bu, embedding modelinin (`qwen3-embedding-0.6b`)
   diller-arası hizalamasının kelime bazında tutarsız olduğunu gösteriyor —
   "gaz kaçağı" gibi bazı terimler için daha önce (bkz. Faz 2 araştırması)
   0.6-0.8 arası güçlü skorlar ölçülmüştü, ama "yangın söndürücü" gibi daha
   nadir terimler için çok daha zayıf. **Bu, chunking/kod tarafında
   düzeltilebilecek bir şey değil** — embedding modelinin kendi çok-dilli
   eğitim kapsamının bir sınırı; olduğu gibi belgeleniyor.

**Son sonuç (0.35/0.65 ağırlıkla, canlı config): 29/30** — kolay 10/10, orta
9/10, zor 10/10. Kalan tek miss, yukarıdaki 2. madde (Türkçe embedding
sınırı). 79/79 pytest hâlâ yeşil.

## 9. Cross-Encoder Reranker Eklendi

Kullanıcı isteğiyle, hibrit (TF-IDF+embedding) retrieval'in üstüne gerçek bir
cross-encoder reranking aşaması eklendi — bi-encoder (embedding) sorgu ve
chunk'ı **ayrı ayrı** kodlayıp kosinüs benzerliğine bakarken, cross-encoder
ikisini **birlikte** tek girdi olarak modele verip doğrudan bir alaka skoru
üretiyor — daha yavaş (chunk başına bir model çağrısı) ama daha isabetli.

- **Model:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (çok dilli, Türkçe
  dahil mMARCO üzerinde eğitilmiş), `sentence-transformers` üzerinden.
- **Mimari:** `src/reranker.py::CrossEncoderReranker` — hibrit arama önce
  `RERANK_CANDIDATE_K=15` aday getiriyor, reranker bunları `TOP_K=5`'e
  indiriyor. Kütüphane/model yoksa `ready=False`'a düşüp hibrit sıralamayı
  olduğu gibi kullanıyor (zarif bozulma, projenin tüm diğer opsiyonel
  bileşenleriyle aynı desen).
- **Kurulum sırasında bulunan (ve kendi kendine çözülen) engel:** İlk pip
  kurulumu `.venv/Scripts/pip` izin hatasıyla sessizce başarısız oldu
  (arka plan `tail` boru hattı bunu gizledi — gerçek hata fark edilmeden
  "tamamlandı" raporlandı, bir dahaki sefere çıktı içeriğini mutlaka
  doğrulamak gerekiyor). `python -m pip install` ile düzeltildi. Ardından
  gerçek bir Windows engeli çıktı: `scipy.linalg._decomp_interpolative`
  DLL'i **Windows Uygulama Denetimi (Application Control/Smart App
  Control)** tarafından bloke edildi — yeni indirilen, henüz "itibar"
  kazanmamış bir ikili dosya olduğu için. Hiçbir ayar değiştirilmeden,
  bir süre sonra (Windows'un bulut itibar kontrolü arka planda dosyayı
  onayladıktan sonra) kendiliğinden çözüldü — kullanıcı aynı sorunu başka
  bir projede farklı bir scipy sürümüyle yaşamamıştı, bu da ilk-dokunuş
  itibar kontrolü teorisini doğruladı.
- **EN/TR doğrulama:** "How do I detect a gas leak?" → doğru pasaj **4.87**
  skorla açık ara önde (yanlış adaylar -3.27/-4.34); Türkçe eşdeğeri
  ("Gaz kacagini nasil tespit ederim?") → doğru pasaj yine 1. sırada ama
  mutlak skor çok daha düşük (-1.53 vs -4.02/-5.41) — göreceli sıralama
  doğru, çok-dilli mutlak güven embedding'de gördüğümüz gibi yine düşük.

### 30 Soruluk Retrieval Denetimi — Reranker'lı Sonuç

Aynı 30 soru (madde 8), reranker devredeyken tekrar çalıştırıldı:

| | Reranker'sız (0.35/0.65 hibrit) | Reranker'lı |
|---|---|---|
| Kolay | 10/10 | 10/10 |
| Orta | 9/10 | 9/10 |
| Zor | 10/10 | 10/10 |
| **Toplam** | **29/30** | **29/30** |

**İsabet sayısı aynı kaldı ama sıralama kalitesi belirgin şekilde arttı:**
reranker'sız sürümde birden fazla soru doğru dokümanı rank 2/4/5/6'da
buluyordu; reranker'lı sürümde bunların neredeyse tamamı **rank 1**'e
yükseldi (sadece 2 soru rank 2'de kaldı). Kalan tek miss — "Yangin
sondurucu secimi nasil yapilir?" — hem bi-encoder'da hem cross-encoder'da
aynı nedenle başarısız: bu spesifik Türkçe terimin ("söndürücü") kendisi
için çok-dilli hizalama zayıf, sorun retrieval mimarisinde değil, modelin
kelime dağarcığı kapsamında. **Dürüst değerlendirme: reranker isabet
sayısını değiştirmedi ama sıralama güvenilirliğini gerçek anlamda artırdı**
— tam iyileştirme değil ama net bir kazanç.

## 10. qwen3-4b — 10 Soruluk Canlı Test (düzeltmelerle)

Kullanıcı isteğiyle `qwen2.5-7b` → `qwen3-4b` denendi (daha hafif kaynak
kullanımı hedefiyle). İki gerçek düzeltme uygulandıktan sonra tam 10 soru
test edildi:
1. `config.THINKING_MODEL_MAX_TOKENS=3000` (Qwen3 ailesi için, 800'den
   yükseltildi) — Qwen3'ün varsayılan `<think>...</think>` iç akıl yürütme
   izi CEVAPLA AYNI token bütçesini paylaşıyor; SDK'da bunu API seviyesinde
   kapatacak bir `enable_thinking` parametresi yok (`ChatClientSettings`
   incelendi: sadece temperature/max_tokens/penalty/top_p/top_k/
   response_format/tool_choice destekleniyor) — tek erişilebilir yol metin
   içi `"/no_think"` talimatı (zaten vardı) + daha büyük bütçe.
2. `src/chat_engine.py::_strip_thinking_block()` — kapanmış VEYA
   `max_tokens` yüzünden yarıda kalmış açık `<think>` bloklarını nihai
   cevaptan temizliyor.

**Sonuç: 3 tam doğru, 2 kısmi, 5 gerçek başarısızlık — `qwen2.5-7b`'den daha
güvenilir değil, hatta yeni bir gerileme var.**

| # | Soru | Sonuç |
|---|---|---|
| 1 | Gas leak detection | ✅ Mükemmel, düzgün format |
| 2 | Fire on site | ❌ **Tamamen boş cevap** — düzeltmelere rağmen hâlâ oluyor (deterministik değil) |
| 3 | Boru hattı metali (TR) | ❌ Talep edilen tam ret cümlesini kullanmıyor, doğru bilgiyi (magnesium/zinc) yine çıkaramıyor |
| 4 | Blowout preventer | ✅ Doğru içerik |
| 5 | Confined space O2 (TR) | ⚠️ Rakamlar doğru, çeviri karışık-dilli |
| 6 | H2S OSHA limitleri | ✅ Doğru ama NIOSH rakamı eksik |
| 7 | Gas detector calibration | ⚠️ İçerik doğru ama iç kaynak etiketini ("[Source 4]") cevaba sızdırmış + tekrarlı cümleler |
| 8 | Pipeline pressure test | ❌ **Yeni gerileme** — doğru chunk ilk sırada bulunmasına rağmen "bilgi mevcut değil" diyor; bu soru `qwen2.5-7b` ile HER turda sorunsuz doğru cevaplanmıştı |
| 9 | Photosynthesis (kontrol) | ✅ Doğru reddetti |
| 10 | H2S PPE | ❌ Tekrar-döngüsüne girdi (koruma yakaladı) |

Ham çıktılar `qa_test_results_qwen3_4b.json`'da.

### Kanıt kontrolü: hata gerçekten modelde mi?

Kullanıcının haklı bir şüphesi vardı: "hata modelde mi emin değilim, önce
chunk'ları kontrol et." qwen3-4b'nin başarısız olduğu 5 sorunun (2, 3, 7, 8,
10) **gerçekte modele giden top-5 chunk'ları birebir okundu**. Sonuç: **beşinde
de doğru chunk ilk sırada, temiz, tam, tek anlamlı** (örn. Q8 için "Pipeline
Pressure Testing" chunk'ında "**Acceptable pressure drop: less than 0.5%**"
kalın yazıyla net şekilde yazıyor; Q3 için magnesium/zinc cümlesi birebir
orada). Hiçbir chunk'ta kopukluk, karışık başlık, ya da retrieval hatası
yok. **Kesin sonuç: bu beş başarısızlığın tamamı model-tarafı bir üretim
sorunu, veri/chunking/retrieval sorunu değil.**

## 11. qwen3-8b — 30 Soruluk Test (kesin sonuç)

Kullanıcı isteğiyle `qwen3-8b` denendi — `THINKING_MODEL_MAX_TOKENS=3000` ve
`_strip_thinking_block()` gibi mitigasyonlar "qwen3" ön ekine göre otomatik
devreye girdiği için qwen3-4b için yapılan düzeltmeler buraya da miras
kaldı. Retrieval-only denetimdeki aynı 30 soru (10 kolay/10 orta/10 zor) bu
kez **tam üretim** (ChatEngine, reranker dahil) ile çalıştırıldı.

**Sonuç: 30 sorudan 28'i tamamen BOŞ cevap döndürdü** — sadece 2 soru
(Q3, Q9) bir şey üretti. Retrieval her seferinde doğru dokümanı buluyordu
(sources alanı neredeyse hep doğruydu), üretim tarafı tamamen çöktü.

**Kök neden doğrulaması:** Ham model çağrısı (`FoundryClient.stream_chat`,
hiçbir post-processing olmadan) "How do I detect a gas leak?" için tam
olarak şunu döndürdü: `'<think>\n\n'` — **9 karakter**. `last_response_truncated=False`,
yani 3000 tokenlık bütçe hiç tükenmedi — model thinking etiketini açar
açmaz, hiçbir şey üretmeden duruyor. **Bu, token bütçesi veya prompt
mühendisliğiyle çözülebilecek bir şey değil** — Foundry Local'in
`qwen3-8b-generic-cpu` paketlemesinde temel bir sorun olduğunu gösteriyor.

### Üç Model — Nihai Karşılaştırma

| Model | Sonuç |
|---|---|
| **qwen2.5-7b** | Güvenilir — onlarca test turunda hiç tekrar-döngüsü/boş cevap yok. Bilinen sınırlar: çeviri akıcılığı, nadir dar-kapsamlı çıkarım hataları (örn. cathodic protection "hangi metal") |
| **qwen3-4b** | 30 sorunun 10'unda (tam üretim testinde) 3 doğru, 2 kısmi, 5 gerçek hata (boş cevap, yanlış ret cümlesi, tekrar-döngüsü, kaynak-etiketi sızıntısı) — veri kontrol edildi, hepsi model-tarafı |
| **qwen3-8b** | 30 sorunun 28'i tamamen boş — kök neden doğrulandı: model `<think>` etiketini açıp hiçbir şey üretmeden duruyor |

**Karar: `qwen2.5-7b`'ye geri dönüldü** (`config.py`). Üç modelin kapsamlı,
kanıta dayalı karşılaştırması sonucunda `qwen2.5-7b` bu proje için hâlâ en
güvenilir seçenek.

### Thinking Modunu Gerçekten Kapatma Girişimi (iki teknik denendi, ikisi de başarısız)

Kullanıcı, başka bir projesinde Ollama'nın `ollama.chat(..., think=False)`
şeklinde native bir thinking-kapatma parametresi desteklediğini gösterdi —
bu, Qwen3 modellerinin genel olarak API seviyesinde bir kapatma anahtarını
desteklediğini kanıtlıyordu. Foundry Local'in bunu gizli/dokümante
edilmemiş şekilde destekleyip desteklemediği iki farklı teknikle test edildi:

1. **Boş `<think></think>` ile "assistant" mesajı prefill'i** — mesaj
   dizisinin sonuna zaten kapanmış boş bir thinking bloğu içeren bir
   "assistant" turu eklenip modelin doğrudan cevaba devam etmesi
   sağlanmaya çalışıldı (topluluk arasında bilinen bir teknik).
   **Sonuç: Foundry Local'in native motoru bunu hiç desteklemiyor** —
   `FoundryLocalException: Operation was cancelled` hatasıyla tamamen
   reddetti.
2. **`chat_template_kwargs: {"enable_thinking": false}` enjeksiyonu** —
   SDK'nın `ChatClientSettings`'i bu alanı desteklemese de, isteğin son
   JSON'a dönüştüğü `CompletionCreateParamsStreaming`'in bir `TypedDict`
   olduğu (yani runtime'da SIFIR doğrulama yaptığı, fazladan alanların
   sansürsüz JSON'a sızdığı) doğrulanıp bu alan doğrudan enjekte edildi.
   **Sonuç: native motor bu alanı sessizce YOK SAYDI** — enjeksiyonlu ve
   enjeksiyonsuz çıktı birebir aynıydı (aynı uzunluk, aynı ilk 500 karakter).

**Kesin sonuç: Foundry Local'in bu ONNX tabanlı ("generic-cpu") Qwen3
motorunda thinking modunu API seviyesinde kapatacak HİÇBİR erişilebilir
yol yok** — ne resmi SDK alanı, ne prefill hilesi, ne de vLLM/SGLang tarzı
`chat_template_kwargs` enjeksiyonu işe yarıyor. Tek erişilebilir savunma
hattı hâlâ metin-içi `"/no_think"` talimatı + token bütçesi artırma +
çıktıdan `<think>` bloğu temizleme (zaten uygulı) — ve bunlar yeterli
güvenilirliği sağlamadı (bkz. madde 10-11). Bu araştırma hattı artık
tükenmiş sayılabilir; `qwen2.5-7b` kararı bu ek kanıtla daha da güçleniyor.

## 12. Mimari Değişiklik: Foundry Local → Ollama (llama3.1:8b + bge-m3)

Madde 10-11'deki üç Qwen3 modelinin de thinking modunu Foundry Local'de
**API seviyesinde kapatacak hiçbir yol bulunamadı** (ne resmi bir alan, ne
prefill hilesi — hata verdi, ne `chat_template_kwargs` enjeksiyonu — sessizce
yok sayıldı). Kullanıcı, başka bir projesinde Ollama'nın bunu native
`think=False` parametresiyle desteklediğini gösterdi. Bunun üzerine proje
**tamamen Ollama'ya geçirildi**:

- **Chat modeli:** `llama3.1:8b` (Ollama) — Foundry Local kataloğunda
  Llama ailesi hiç yok (muhtemelen Meta lisans kısıtlaması), bu yüzden
  Ollama şart oldu.
- **Embedding modeli:** `bge-m3` (Ollama) — Meta resmi bir "Llama embedding"
  yayınlamıyor; çok-dilli (Türkçe dahil) güçlü bir alternatif seçildi.
- **Yeni kod:** `src/ollama_client.py` (`OllamaClient`) ve
  `src/ollama_embedder.py` (`OllamaEmbedder`) — `FoundryClient`/`LocalEmbedder`
  ile birebir aynı arayüz (`ready`/`message`/`init()`/`stream_chat()`/`embed()`),
  bu yüzden `ChatEngine` hiç değişmeden çalışıyor. Tekrar-döngüsü koruması
  (`_is_runaway_repetition` vb.) `foundry_client.py`'den olduğu gibi
  yeniden kullanıldı, kopyalanmadı.
- **Sağlayıcı seçimi:** `config.LLM_PROVIDER` ("foundry" veya "ollama",
  varsayılan artık "ollama") — `app/streamlit_app.py` ve
  `scripts/run_ingest.py` buna göre doğru istemciyi seçiyor.
- **Disk temizliği:** Kullanıcı isteğiyle, artık kullanılmayan 5 Foundry
  Local modeli (`qwen2.5-7b`, `phi-3.5-mini`, `qwen3-4b`, `qwen3-8b`,
  `qwen3-embedding-0.6b`) `remove_from_cache()` ile silindi — proje-özel
  önbellek (`~/.local-rag-foundry/cache/models`) **18GB → 84KB**'a düştü.
  Bu önbellek sadece bu projeye ait olduğu için başka hiçbir projeyi
  etkilemedi.
- Corpus yeniden indekslendi: 248 chunk, `bge-m3` ile. 89/89 pytest yeşil.

### 10 Soruluk Sonuç — llama3.1:8b + bge-m3

| # | Soru | Sonuç |
|---|---|---|
| 1 | Gas leak detection | ✅ Doğru, temiz |
| 2 | Fire on site | ⚠️ İçerik doğru ama Referans yanlış ("Hot Work Permit" yazıyor, doğrusu "Fire Response") |
| 3 | **Boru hattini hangi metal korur? (TR)** | ✅ **DOĞRU — "Magnezyum ya da çinko"** — bu oturumda bu soruyu doğru cevaplayan İLK model (qwen2.5-7b, qwen3-4b, qwen3-8b hepsi başarısız olmuştu). Akıcı, gerçek Türkçe (Qwen'lerin karışık-dilli çevirisinden çok daha iyi) |
| 4 | Blowout preventer | ✅ Doğru, doğru referans |
| 5 | Confined space O2 (TR) | ✅ Doğru rakamlar (%19,5-%23,5), akıcı Türkçe |
| 6 | H2S OSHA limitleri | ✅ Çok kapsamlı ve doğru — qwen2.5-7b'den bile daha eksiksiz (tüm rakamlar var) |
| 7 | Gas detector calibration | ❌ **Gerçek eksiklik** — doğru chunk bulunmasına rağmen asıl cevabı (6 ayda bir/günlük bump test) vermiyor, "yerel düzenlemelerinize bakın" gibi genel bir cevap veriyor |
| 8 | Pipeline pressure drop | ⚠️ Doğru rakam (0.5%) kendi alıntısında var ama kendiyle çelişen kafa karıştırıcı bir yorum ekliyor ("değerin ne olduğu belirtilmemiş" diyor, oysa aynı cümlede yazıyor) |
| 9 | Photosynthesis (kontrol) | ✅ Doğru reddetti ama gereksiz ek yorum ekledi (talimata tam uymuyor ama halüsinasyon da yok) |
| 10 | H2S PPE | ⚠️ İçerik doğru ama "[Source N]" placeholder'ı doldurulmadan cevaba sızmış, başlık tekrarı var |

**Değerlendirme: 5/10 tam temiz doğru, 1/10 gerçek eksiklik, 4/10 doğru
içerik ama referans/format sorunlu.** Yeni, tekrarlayan bir sorun tespit
edildi: **model bazen `[Source N]` etiketini ya yanlış dokümanla eşleştiriyor
ya da doldurulmamış placeholder olarak bırakıyor** — bu, prompt'ta ek bir
netleştirmeyle muhtemelen düzeltilebilir (henüz denenmedi).

### Dört Modelin Nihai Karşılaştırması

| Model | Motor | Sonuç |
|---|---|---|
| qwen2.5-7b | Foundry Local | Güvenilir, temiz format — ama cathodic protection sorusunu HİÇ çözemedi |
| qwen3-4b | Foundry Local | 3/10 doğru, çoklu gerçek hata |
| qwen3-8b | Foundry Local | 28/30 tamamen boş — temel paketleme arızası |
| **llama3.1:8b** | **Ollama** | **5/10 tam temiz, ilk kez cathodic protection'ı doğru çözdü, Türkçe çevirisi en akıcısı — ama yeni bir referans/citation tutarsızlığı var** |

**Sonuç: llama3.1:8b + bge-m3 (Ollama), önceki üç modelden daha iyi bir
temel gösteriyor** — özellikle Türkçe akıcılık ve daha önce çözülemeyen
cathodic-protection sorusunda. Referans-etiketi tutarsızlığı yeni ve gerçek
bir sorun, ama diğer üç modelin döngü/boş-cevap/format-yoksayma
sorunlarından daha hafif ve muhtemelen prompt düzeltmesiyle iyileştirilebilir.

## 13. BM25'e Geçiş + Referans Düzeltmesi Girişimi + 30 Soruluk Tam Üretim Testi

Kullanıcı isteğiyle üç değişiklik yapıldı, sonra aynı 30 soru (madde 8'deki
retrieval denetimiyle aynı liste) bu kez **tam üretim** (ChatEngine,
llama3.1:8b + bge-m3 + reranker) ile test edildi.

### Değişiklik 1: TF-IDF → BM25

`src/bm25.py` eklendi (Okapi BM25, saf Python, yeniden indeksleme
gerektirmiyor — mevcut `tf` sütunundan hesaplanıyor). `src/vector_store.py`
TF-IDF+cosine yerine BM25 kullanacak şekilde güncellendi; BM25 skoru
sınırsız olduğu için (cosine gibi [0,1] değil) aday kümesi içinde
min-max normalize ediliyor. `config.HYBRID_TFIDF_WEIGHT` →
`HYBRID_BM25_WEIGHT` olarak yeniden adlandırıldı (0.35/0.65 aynen taşındı,
yeniden doğrulanmadı). `src/tfidf.py`'deki artık kullanılmayan
`compute_idf`/`tfidf_vector`/`cosine_similarity` (sparse) silindi — 92/92
pytest yeşil (7 yeni BM25 testi eklendi, 4 eski TF-IDF-cosine testi
kaldırıldı, net +3).

**Retrieval-only doğrulama (30 soru, BM25+bge-m3+reranker): 28/30** — önceki
29/30'dan (TF-IDF+qwen3-embedding-0.6b) hafif düşüş. Aynı "Yangin sondurucu
secimi nasil yapilir?" miss'i devam ediyor (embedding modeli değişse de bu
terimin çok-dilli hizalaması hâlâ zayıf); yeni bir miss de çıktı ("kaynak
isinden sonra yangin gozcusu ne kadar sure beklemeli?" — bge-m3 ile alakasız
dokümanlar geliyor, qwen3-embedding-0.6b ile doğru bulunuyordu). Bu,
embedding modeli değişikliğinin (Ollama'ya geçişin bir parçası) küçük ama
gerçek bir yan etkisi — daha fazla ayarlama yapılmadı.

### Değişiklik 2: Referans/Kaynak-Etiketi Düzeltmesi — BAŞARISIZ

Önceki 10 soruluk testte bulunan sorunları (yanlış kaynak adı, doldurulmamış
"[Source N]" placeholder'ı) düzeltmek için `SYSTEM_PROMPT`'a şu kurallar
eklendi: asla "[Source" metnini birebir yazma, kaynak sadece son "Reference"
satırında, cevabı yazdıktan sonra hangi excerpt'i gerçekten kullandığını
tekrar kontrol et, gövde metninden ("Source Standard" gibi) alıntı yapma.

**30 soruluk testte bu kural büyük ölçüde göz ardı edildi:**
- **Literal `[Source N]` / `[Kaynak N]` köşeli parantez kullanımı hâlâ
  sık** (Q3, Q10, Q11, Q12, Q14 — talimat açıkça yasaklamasına rağmen).
- **Gövde metninden/bölüm başlığından kaynak uydurma devam ediyor** — Q8
  Referans olarak "Field Safety Manual" yazdı (dokümanın kendi "Source
  Standard" satırından, tam olarak yasaklanan davranış); Q18 "Trenching
  Safety Initiative" (bir alt-başlık) yazdı, doğrusu "Trenching and
  Excavation - Overview (OSHA)" olmalıydı; Q23 doğrudan bir bölüm başlığını
  ("Why The Fire Watch Continues After Work Stops") kaynak olarak verdi.
- **Yanlış doküman adı** — Q4 "Hydrogen Sulphide (H2S) Monitoring" dedi,
  beklenen "Personal Protective Equipment Requirements" idi (içerik iki
  dokümanda da örtüştüğü için kısmen mazur görülebilir, ama yine de yanlış).

**Sonuç: prompt-seviyesi talimat bu modelde güvenilir çalışmıyor.** Daha
sağlam bir çözüm (denenmedi, gelecek iş): Ollama'nın yapılandırılmış çıktı
(`format` parametresi ile JSON şeması) desteğini kullanıp "Reference"
alanını modele serbest metin yerine, bağlamda gerçekten verilen başlıklardan
oluşan sabit bir listeden **seçtirmek** — bu, modelin uydurma/yanlış
alıntı yapma ihtimalini yapısal olarak ortadan kaldırır.

### 30 Soruluk Tam Üretim Testi — Genel Bulgular

Kaynak/referans sorunlarının ötesinde, bu daha büyük örneklemde **iki yeni
sorun kategorisi** ortaya çıktı (10 soruluk testte görülmemişti):

1. **Sessiz yarıda kesilme (repetition-guard'ın yakalamadığı bir tür).**
   Q1, Q6, Q15, Q30 "Reference" bölümünü hiç yazmadan bitti; Q24 numaralı
   liste öğelerini boş bırakıp ("2)\n3)") durdu. Bunların hiçbirinde
   `notice` alanı dolu değildi (tekrar-döngüsü değil, sadece erken durma) —
   mevcut güvenlik ağımız bu tür sessiz eksik-cevapları yakalamıyor.
2. **Türkçe çeviri artık sadece akıcılık değil, gerçek anlam hatası da
   üretebiliyor.** Q11'de "gaz kaçağı" (leak) yanlışlıkla "gaz patlaması"
   (explosion) olarak çevrildi VE yanlış doküman adıyla ("Gaz Patlama Tespit
   Prosedürü" — böyle bir doküman yok) birleşince tamamen anlamsız bir
   cevap ortaya çıktı. Q19'da doğru rakamlar (%19,5-%23,5) önce doğru
   verilip birkaç cümle sonra yanlış (%19-%23) olarak kendiyle çelişecek
   şekilde tekrarlandı.
3. **Kötü retrieval, kötü üretime dönüşebiliyor.** Q13 ("Yangin sondurucu"),
   retrieval yanlış dokümanları bulunca model tamamen alakasız bir konuya
   (katodik koruma metal seçimi) sıçrayıp tutarsız bir cevap üretti — iyi
   haber: Q28'de aynı tür bir retrieval-miss'te model dürüstçe "bu bilgi
   bulunamıyor" dedi, halüsinasyon yapmadı; yani davranış tutarsız/model
   içi rastgele.

**Genel doğruluk (referans/format kusurlarını saymadan, sadece asıl
içerik):** ~24/30 makul/doğru, ~4-5/30 gerçekten sorunlu (Q11, Q13, Q19,
Q24, kısmen Q28). Ama **referans satırı sorunsuz olan cevap sayısı sadece
~6/30** — bu, kullanıcının özellikle çözülmesini istediği sorunun BAŞARISIZ
olduğu anlamına geliyor ve dürüstçe böyle raporlanıyor.

## 14. Referans Sorunu — Mimari Çözüm (prompt yerine kod)

Madde 13'teki prompt-seviyesi düzeltme (kurallar ekleyerek modeli doğru
alıntı yapmaya ikna etmeye çalışmak) başarısız olmuştu. Kullanıcı daha
sağlam bir yaklaşım önerdi: **modelden hiç kaynak yazmasını istememek**,
bunun yerine kaynağı retrieval'in zaten sahip olduğu doğru metadata'dan
(her chunk zaten `doc_id`/`title`/`category` taşıyor, bu her doküman
yüklendiğinde otomatik ekleniyor — `chunker.document_to_chunks`) **kod
tarafında** deterministik olarak oluşturmak.

Üç değişiklik:
1. `SYSTEM_PROMPT`'tan tüm "Citation rules" bloğu ve "Reference" satırı
   talebi kaldırıldı — model artık kaynak göstermeye hiç çalışmıyor.
2. `build_context_block()`'taki `"[Source N] <title>"` köşeli parantez
   etiketi tamamen kaldırıldı (`"Excerpt from \"<title>\" (<category>):"`
   ile değiştirildi) — modelin taklit ettiği tam da bu görsel-alıntı-benzeri
   köşeli parantez deseniydi.
3. `src/chat_engine.py::_build_reference_footer()` / `_with_reference_footer()`
   eklendi: `ask()`'ın üç yield noktasında da (çeviri yok/çeviri başarısız/
   çeviri başarılı), modelin ürettiği metne, **retrieval'den gelen gerçek
   `sources` listesinden** oluşturulan `"\n\nReference: title1; title2"`
   satırı kod tarafından ekleniyor — modelin yazdığı hiçbir şeye
   dayanmıyor. Temiz bir ret cevabına (`NO_CONTEXT_REPLY` ile birebir eşit)
   footer eklenmiyor.

**Doğrulama (3 soru, gerçek model):**
- Üçünde de `"[Source"` / `"[Kaynak"` sızıntısı **yok** (önceden sürekli
  oluyordu).
- "How long must the fire watch continue..." sorusunda Reference artık
  doğru: `"Hot Work Permit; Fire Response"` (önceden "Field Safety Manual"
  gövde metnini ya da bir alt-başlığı kaynak gösteriyordu).
- "Yangin sondurucu secimi nasil yapilir?" sorusunda model artık cathodic
  protection'a kaydığını fark edip konuyla "ilgim yok" diyerek dürüstçe
  reddetti — önceki tamamen tutarsız/alakasız halüsinasyon yerine (kaynak
  konusuyla uğraşmak zorunda kalmayınca modelin dikkati daha iyi
  odaklanmış görünüyor, kesin nedensellik iddia edilmiyor).

**Kalan küçük, bilinen sınır:** Ret cümlesi tam olarak İngilizce
`NO_CONTEXT_REPLY` ile eşleşmediğinde (örn. çeviri sonrası Türkçe bir ret,
ya da yukarıdaki gibi modelin kendi ürettiği bir ret cümlesi) footer yine
de ekleniyor — bu, "ret cevabına kaynak gösterme" kuralının sadece tam
eşleşen durumları yakaladığı anlamına geliyor. Küçük bir kozmetik
tutarsızlık, yanlış bilgi değil (o kaynaklar gerçekten kontrol edildi,
sadece yetersiz bulundu). 94/94 pytest yeşil.

**Sonuç: referans sorunu artık prompt mühendisliğiyle değil, mimari olarak
çözüldü** — modelin bunu doğru yapmasına güvenmek yerine, zaten var olan
doğru veriden kod tarafında inşa ediliyor.

### Bilinen Kalite Sınırı (hata değil)

`qwen2.5-7b` + zenginleştirilmiş corpus ile tekrar döngüsü/kelime-serpiştirme
bozulması bir daha görülmedi. Zorunlu dil çevirisinin **akıcılığı** hâlâ
tutarsız: bazen gramer kusurlu, bazen model talimatı görmezden gelip
İngilizce yanıtlıyor, bazen ham `##` başlıkları sızdırıyor — ama bunların
hiçbiri artık sonsuz döngüye dönüşmüyor. "Auto" modu da bazen Türkçe soruya
İngilizce yanıt verebiliyor (dil eşleştirme talimatı %100 güvenilir değil).
Detaylar için README "Known limitations" bölümüne bakın.

## 15. Kapsamlı 98 Soruluk Canlı Test (Ollama + BM25 + Reranker + Deterministik Referans)

`_run_qa_100.py` ile üretilen, üretim yığınının (llama3.1:8b + bge-m3 +
BM25/dense hibrit + cross-encoder reranker + deterministik referans footer)
üzerinden geçen 98 soruluk uçtan uca canlı test. Sorular 40 kaynak
dokümanın tamamını (daha önce hiç test edilmemiş 17 OSHA/EPA referans
dokümanı dahil) kapsayacak şekilde, kolay/orta/zor zorlukta ve
İngilizce/Türkçe/karışık-dilde tasarlandı; 3 tanesi bilgi tabanında
karşılığı olmayan negatif kontrol sorusu. Ham sonuçlar
`qa_test_results_100.json`'da, tam soru-cevap-kaynak dökümü ve analiz
[şeffaflık raporunda](https://claude.ai/code/artifact/c744796f-2da7-4cc9-aecb-254c12f65f63) mevcut.

**Sonuç dağılımı:** 78/98 doğru (%80), 13/98 kısmi/belirsiz (%13),
2/98 dürüst "bulunamadı" (bilinen kapsam boşlukları, halüsinasyon yok),
5/98 gerçek hata (%5). Dil dağılımı: 61 EN / 33 TR / 4 karışık. Toplam
çalışma süresi model+reranker yükleme dahil ~4 dk 48 sn (~2.6 sn/soru) —
bu makinede Ollama'nın GPU hızlandırmalı çalıştığını doğruluyor.

**Bulunan 5 gerçek hata:**
1. **Çapraz-dil retrieval hatası (tekrarlanabilir):** "How long must the
   fire watch continue after hot work is finished?" İngilizce doğru
   cevaplanırken ("Hot Work Permit", 30 dakika), Türkçe paraphrase'i
   ("kaynak isinden sonra yangin gozcusu ne kadar sure beklemeli?")
   ilgili dokümanı retrieval'a hiç sokamadı — sonuç tamamen alakasız
   5 kaynaktan oluşan bir "bulunamadı" cevabıydı.
2. **PPE sorusu için tam retrieval kopması:** "Yuksekte calisirken hangi
   KKD gereklidir?" sorusu PPE/Fall Protection yerine Wellhead Inspection,
   ESD ve Cathodic Protection dokümanlarını getirdi; cevap da buna bağlı
   olarak tamamen konu dışı.
3. **OSHA standart numarası halüsinasyonu:** Hazardous Energy Control
   sorusunda model "1910.Subpart.S" diye bir standart numarası uydurdu;
   gerçek referans 29 CFR 1910.147.
4. **Negatif kontrol sızıntısı:** "Yapay zeka nedir?" sorusu önce doğru
   şekilde reddedildi, ama aynı cevabın devamında alakasız gürültü
   chunk'larından uydurma bir "kulak anatomisi" paragrafı üretildi —
   diğer iki negatif kontrol (fotosentez, Fransa'nın başkenti) temiz
   şekilde reddedildi.
5. **Konu dışı cevap:** Pig launcher'daki mekanik kilidin önemini soran
   soruya, retrieval doğru dokümanı getirmesine rağmen cevap LOTO/motor
   elektrik beslemesi gibi alakasız bir konuya kaydı.

**2 dürüst boşluk (hata değil):** "Yangin sondurucu secimi nasil yapilir?"
(Fire Response dokümanı bu konuyu içermiyor — önceki test turlarında da
görülen, kalıcı bir korpus boşluğu) ve OSHA Oil & Gas eTool'un genel
odağını soran soru (kaynak doküman büyük ölçüde ince bir gezinme sayfası).
Her ikisinde de model halüsinasyon yapmak yerine dürüstçe reddetti.

**Genel değerlendirme:** %80 net doğruluk ve halüsinasyonların büyük
çoğunluğunun (referans footer sayesinde) yanlış kaynak göstermek yerine
yanlış *içerik* üretmekle sınırlı kalması, mimari değişikliklerin
(BM25, reranker, deterministik referans, Ollama geçişi) birlikte gerçek
bir kalite artışı sağladığını gösteriyor. Kalan hataların ortak teması
net: (a) Türkçe sorularda embedding-tabanlı retrieval hâlâ İngilizce
kadar güvenilir değil, (b) model bilmediği spesifik sayısal/kod
referanslarını (standart numaraları gibi) uydurmaya bazen İngilizce'de
de meyilli.

## 16. Fix Round 1: 5 Hatanın Kök Nedeni ve Düzeltmesi

§15'te bulunan 5 hata, modele gitmeden önce chunk/retrieval katmanı
doğrudan incelenerek (bu oturumda daha önce kurulan yöntem) kök nedenine
kadar izlendi. Diagnostik script'lerle (kalıcı tutulmadı, sonuçlar burada
özetlendi) `VectorStore.search()`'ün ham BM25 skorlarını ve
`CrossEncoderReranker`'ın çıktısını doğrudan sorgulayarak iki farklı katmanda
iki farklı kök neden bulundu — **hiçbiri chunking veya embedding modelinde
değildi**; saf semantik arama (bge-m3) her iki başarısız Türkçe sorguda da
doğru chunk'ı kendi başına doğru sıralıyordu.

**Kök neden 1 — BM25 normalizasyon gürültüsü (retrieval katmanı):**
"kaynak isinden sonra yangin gozcusu ne kadar sure beklemeli?" sorgusunda
Türkçe "süre" kelimesi ASCII'ye katlanınca ("sure") İngilizce corpus'taki
yaygın "sure" kelimesiyle (örn. "make sure...") çakışıyor. Bu tek tesadüfi
eşleşme, per-query max-normalizasyonun paydasını belirliyor ve alakasız
onlarca dokümanın BM25 payını yapay olarak 1.0'a şişiriyor — doğru
dokümanın (Hot Work Permit, hiç "sure" içermiyor) hibrit skoru bu yüzden
gerçek semantik gücüne rağmen (`sem=0.60`, saf semantik sıralamada #1)
candidate havuzunun (top-15) dışına düşüyordu.

**Kök neden 2 — Reranker'a körü körüne güven (rerank katmanı):**
"Yuksekte calisirken hangi KKD gereklidir?" sorgusunda doğru chunk
(Personal Protective Equipment Requirements) hibrit aramada zaten 1.
sıradaydı (bm25=0, tamamen semantik skorla). Ama cross-encoder reranker bu
Türkçe ifade için TÜM 15 adaya derin negatif (-3.1 ile -4.0 arası) skor
verdi — yani sorguyla hiçbir chunk'ı gerçekten eşleştiremedi — ve bu
"tümü kötü" gürültü sıralaması doğru adayı top-5'in dışına attı. Kalibrasyon
için gerçekten başarılı Türkçe sorgularda en iyi skorun en kötü ihtimalle
-1.5 civarında kaldığı, gerçek kaybolma durumunda ise en iyi skorun bile
-3.1'in altında olduğu ölçüldü.

**Düzeltmeler:**
- `VectorStore.search_semantic_only()` (yeni): sadece embedding benzerliğine
  göre sıralayan, BM25'e hiç dokunmayan bir "yedek havuz". `ChatEngine.ask()`
  reranking yaparken bu havuzu hibrit sonuçlarla birleştiriyor — BM25
  gürültüsü candidate havuzunu bozsa bile doğru semantik eşleşme her zaman
  reranker'a bir şans daha almış oluyor.
- `config.RERANK_MIN_CONFIDENCE = -2.5` + `CrossEncoderReranker.rerank()`:
  bu turdaki en iyi rerank skoru eşiğin altında kalırsa (reranker'ın gerçek
  sinyali yok demektir), reranker'ın gürültülü sıralamasına değil, hibrit
  aramanın kendi (zaten daha güvenilir ölçülen) sıralamasına dönülüyor.
- `src/prompts.py`: "context'te birebir geçmeyen bir kod/standart numarası
  asla uydurulmayacak" kuralı eklendi (Q52'nin "1910.Subpart.S" halüsinasyonu
  için).
- `src/chat_engine.py`: `_opens_with_refusal()` + `_truncate_after_refusal()`
  (yeni) — model bir ret cümlesiyle açıp (İngilizce kanonik metin veya
  Türkçe paraphrase, anahtar-kelime demetleriyle tespit ediliyor) sonra
  yine de üretmeye devam ederse (Q91'de gözlemlenen halüsinasyon sızıntısı),
  yanıt o ret cümlesinden sonrasını atacak şekilde kesiliyor. Aynı tespit
  fonksiyonu `_with_reference_footer()`'da da kullanılarak §14'te bilinen
  bir kozmetik boşluk (paraphrase edilmiş retlere hâlâ footer eklenmesi) da
  bu vesileyle kapatıldı.
- 10 yeni pytest testi eklendi (94/94 → 104/104 yeşil).

**Doğrulama — aynı 98 soru tekrar çalıştırıldı (`qa_test_results_100_v2.json`):**
5 hatanın **tamamı** düzeldi (canlı çıktılarla tek tek doğrulandı): Q16 artık
doğru "30 dakika" cevabı veriyor, Q22 artık doğru PPE ekipmanını (tam vücut
koşum takımı, sertifikalı kanca) anlatıyor, Q52 artık yanlış standart
numarası uydurmuyor, Q91 artık tek cümlelik temiz bir ret veriyor (uydurma
kulak-anatomisi içeriği yok), Q94 artık mekanik kilidin işlevini doğru
açıklıyor. Otomatik "beklenen kaynak top-5'te mi" ölçümü retrieval
isabetini 91/95'ten 93/95'e çıkardı (net +2); tek bir küçük yan etki
gözlendi (Q82, TR — genişleyen aday havuzu bu bir soruda referans
dokümanları değiştirdi, ama cevabın içeriği hâlâ makul kaldı, halüsinasyon
değil). Nihai dağılım: **87/98 doğru (%89), 9/98 kısmi, 2/98 dürüst boşluk,
0/98 gerçek hata**. Güncel tam soru-cevap dökümü aynı şeffaflık raporunda
(https://claude.ai/code/artifact/c744796f-2da7-4cc9-aecb-254c12f65f63 —
link yerinde kaldı, içerik güncellendi).

**Bu turun kapsamı dışında bırakılan (ayrı, daha derin bir sorun sınıfı):**
kalan 9 kısmi cevabın çoğu (Q9, Q18, Q31, Q37, Q45, Q56, Q66, Q84) retrieval
değil, llama3.1:8b'nin Türkçe çeviri akıcılığı/kesinlik-kaçamağı sınırından
kaynaklanıyor — bu zaten "Bilinen Kalite Sınırı" bölümünde belgeli, ayrı bir
iyileştirme turu gerektirir (örn. çeviri geçişini güçlendirmek, ya da
belirsiz durumlarda modele "bilmiyorum" yerine kaçamak cevap vermeyi
yasaklayan ek bir kural).

## 17. Yerel (LLM'siz) Çeviri Altyapısı: MarianMT Denemesi → NLLB-200'e Geçiş

§16'da kalan 9 kısmi cevabın çoğunun (Q9, Q18, Q31, vb.) retrieval değil,
llama3.1:8b'nin kendi çeviri geçişinin (LLM'e "önce cevapla, sonra Türkçe'ye
çevir" dedirtmenin) akıcılık/tutarlılık sınırından kaynaklandığı belirlenmişti.
Bu bölüm, o ikinci LLM çağrısını tamamen kaldırıp yerine özel bir çeviri
modeli koyma çalışmasını belgeliyor.

**Adım 1 — Helsinki-NLP/opus-mt-tc-big-en-tr (MarianMT, tek dil çifti):**
`sentence-transformers`'ın zaten kurduğu `transformers`/`torch` altyapısı
üzerine `src/translator.py` (`LocalTranslator`) yazıldı — reranker ile aynı
`init()`/`ready` desenini takip ediyor, model yoksa/başarısız olursa
otomatik olarak eski LLM-tabanlı çeviriye düşüyor. Klasik
`Helsinki-NLP/opus-mt-en-tr` artık HuggingFace Hub'da 401 (deprecated)
döndürdüğü için güncel halka açık "tc-big" (Tatoeba-Challenge) sürümü
kullanıldı. Aynı 98 soru bu çeviriciyle tekrar çalıştırıldı
(`qa_test_results_100_v3.json` → düzeltme sonrası `_v4`, ikisi de silindi,
sonuçlar burada özetleniyor).

**Bulunan hata 1 — sonsuz tekrar döngüsü:** Bir soruda (Q35, "vana pozisyon
göstergesi") çeviri modeli aynı iki Türkçe cümleyi 21 kez art arda üretti.
Kök neden doğrudan izole edilebildi: `max_length=512` her satıra girdi
uzunluğundan bağımsız sabitlenmişti; greedy decoder gerçek içeriği
bitirdikten sonra kalan bütçeyi tekrarla doldurdu. Girdi uzunluğuna
orantılı bir `max_length` (`min(512, max(32, girdi_token*3))`) +
`no_repeat_ngram_size=3` + `repetition_penalty=1.3` eklenmesi bunu yan
yana testte tam olarak ortadan kaldırdı (LLM'in kendi tekrar-koruması ile
aynı "çok katmanlı savunma" mantığı, bkz. `_is_runaway_repetition`).

**Bulunan hata 2 — yanlış hedef dile kayma (daha ciddi):** Tekrar düzeltmesi
sonrası 37 TR/karışık cevabın tam taraması yapıldı: ~%60'ı temiz/akıcıydı
(bazıları eski LLM çevirisinden bile daha iyi), ama 5 tanesi hâlâ ciddi
bozukluk gösteriyordu — en çarpıcısı Q40 ("boru hattı temizleme pigi"),
cevabın **tamamen Portekizce** üretilmesiydi ("Domuz lançamento em tubulao
de petrleo..."). Kök neden: iki dilli bir MarianMT modelinin çıktı dili
hiçbir zaman yapısal olarak *zorlanmıyor*, sadece öyle olması bekleniyor;
nadir durumlarda model paylaşılan alt-kelime uzayında başka bir dile
kayabiliyor. Ucuz bir otomatik "bu çıktı bozuk" tespiti (Türkçe durak-kelime
oranı, yabancı harf kontrolü) denendi ama Portekizce'nin "da" gibi
kelimeleri Türkçe ile çakıştığı için güvenilir çıkmadı.

**Çözüm — Meta NLLB-200'e geçiş:** Kullanıcı açıkça "Meta/Facebook tipi bir
dil modeli kullan, başka dil asla olmasın" talimatı verdi.
`facebook/nllb-200-distilled-600M` bu tam ihtiyaç için tasarlanmış bir
mekanizma sunuyor: `forced_bos_token_id` ile üretim, hedef dilin kendi
token'ıyla başlamaya *zorlanıyor* — modelin başka bir dile kayması, ayrı bir
tespit katmanı gerekmeden, yapısal olarak imkânsız hale geliyor. `config.py`
ve `src/translator.py` bu modele geçirildi (`AutoModelForSeq2SeqLM` +
`AutoTokenizer`, `src_lang="eng_Latn"`, `forced_bos_token_id=tur_Latn`);
aynı uzunluk-ölçekli `max_length` + tekrar-koruması aynen korundu.

**Doğrulama — aynı 98 soru üçüncü kez çalıştırıldı
(`qa_test_results_100_v5.json`):** 37 TR/karışık cevabın tam taraması
tekrarlandı. Sonuç: **dil kayması sıfıra indi** (Q40 artık tamamen ve
tutarlı şekilde Türkçe) ve önceki 5 ciddi bozukluk vakasının (Q11, Q20,
Q40, Q54, Q73) tamamı okunabilir/tutarlı hâle geldi — kalan kusurlar artık
"anlamsız kelime salatası" değil, izole yanlış kelime seçimleri (örn. Q20'de
"personal lock" → "kişisel öykü" (kilit yerine "hikaye") gibi tekil
anlam kaymaları) veya bilinen, önceden belgeli "Auto modu bazen İngilizce
yanıt verebiliyor" sınırının iki tekrarı (Q28, Q58). 116/116 pytest yeşil.

**Sonuç:** Referans/kaynak sorunu için daha önce izlenen aynı mimari
ilke burada da doğrulandı — LLM'e güvenmek yerine, işe özel/deterministik
bir araca geçmek gerçek bir kalite artışı sağladı, ama bu araç kendi
başarısızlık moduna sahip olabiliyor (tekrar döngüsü, dil kayması) ve bu
modlar ancak **gerçek canlı test + kök neden izleme** ile ortaya çıkıyor —
tek bir örnek cümleyle "çalışıyor" demek yeterli olmuyordu, 98 sorunun tam
taraması olmasaydı hem tekrar döngüsü hem dil kayması gözden kaçabilirdi.

## 19. Context Window (num_ctx) Düzeltmesi ve Temperature Deneyi — Kapanış Turu

§17'de kalan 5 kısmi cevabın (Q31, Q37, Q56, Q66, Q84) tamamı, modele
gönderilen chunk içeriği doğrudan okunarak "doğrulanmış LLM sınırı" olarak
işaretlenmişti. Bu sonuca varmadan önce iki bağımsız, kapsamlı test daha
yapıldı — ikisi de **tam 98 soruluk testle** doğrulandı, sadece hedeflenen
5 soruyla değil (kısmi bir düzeltmenin başka yerde regresyon yaratıp
yaratmadığını görmek için).

**Bulgu — context window hiç ayarlanmamış:** `src/ollama_client.py`'de
`num_ctx` hiçbir zaman set edilmiyordu, yani Ollama modelin mimari desteğini
(`llama.context_length=131072`) değil, kendi sabit **2048 token**
varsayılanını kullanıyordu. Bu projenin context bloğu (5 chunk + system
prompt) tipik olarak ~1900-1960 token'a denk geliyor; `MAX_TOKENS=800`'lük
çıktı payı da AYNI 2048'lik pencereden düşüldüğü için, bazı sorularda
çıktının kelimenin ortasında kesildiği doğrudan gözlemlendi (Q31'de
"...spec" diye kesiliyordu). `num_ctx=8192` ile aynı soru tam ve doğru
cevap üretti ("en az 6 ayda bir"). Bu, `config.OLLAMA_NUM_CTX=8192` olarak
üretime alındı — kanıtlanmış, yan etkisiz, gerçek bir düzeltme.

**Denenen ve reddedilen — temperature=0.0:** Kesin sayısal/kod çıkarımı
için düşük sıcaklığın standart bir iyileştirme olduğu bilindiğinden,
`TEMPERATURE` 0.2'den 0.0'a düşürülüp aynı 98 soru tekrar çalıştırıldı.
Sonuç **net olarak karışıktı, kazanç yoktu**: hedeflenen 5 sorudan sadece
1-2'sinde marjinal fayda görüldü, ama daha önce tamamen doğru ve temiz olan
bir cevap (Q47, "25 galon") kendi kendiyle çelişen bir belirsizlik ekleyerek
BOZULDU ("bu bilgiyi veriyorum ama aslında eksik" gibi anlamsız bir
öz-şüphe). Bu, projenin FREQUENCY_PENALTY/PRESENCE_PENALTY ayarlarında daha
önce öğrendiği dersi doğruluyor: küçük modellerde parametre ince ayarı bir
yeri düzeltirken başka bir yeri bozuyor. **Temperature 0.2'ye geri
alındı, üretime hiçbir prompt/parametre değişikliği eklenmedi.**

**Sonuç — final tur, aynı 98 soru (`qa_test_results_100_v6.json`):**
num_ctx düzeltmesiyle Q31 tam olarak düzeldi (doğru "en az 6 ay" bilgisi
artık net veriliyor); Q37/Q56/Q66/Q84 değişmedi. Geniş bir örneklem (22
soru, tüm zorluk/dil kombinasyonlarından) yeniden okunarak yeni bir
regresyon olmadığı doğrulandı (kesilme uyarısı yok, boş cevap yok, beklenmeyen
ret yok — otomatik tarama ile de teyit edildi).

**Nihai durum: 92/98 (%94) doğru, 4/98 doğrulanmış LLM kapasite sınırı
(Q37, Q56, Q66, Q84 — iki farklı müdahale türüyle [prompt kuralı VE
temperature] test edilip düzeltilemediği kanıtlandı), 2/98 dürüst korpus
boşluğu, 0/98 halüsinasyon/retrieval hatası/dil kayması.** 116/116 pytest
yeşil. Bu, projenin başladığı 78/98'lik ilk durumdan üç bağımsız düzeltme
turu (retrieval+reranker, çeviri altyapısı, context window) sonrası ulaştığı
son, kapsamlıca doğrulanmış hâl.
