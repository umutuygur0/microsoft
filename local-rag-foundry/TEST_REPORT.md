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

### Bilinen Kalite Sınırı (hata değil)

`qwen2.5-7b` + zenginleştirilmiş corpus ile tekrar döngüsü/kelime-serpiştirme
bozulması bir daha görülmedi. Zorunlu dil çevirisinin **akıcılığı** hâlâ
tutarsız: bazen gramer kusurlu, bazen model talimatı görmezden gelip
İngilizce yanıtlıyor, bazen ham `##` başlıkları sızdırıyor — ama bunların
hiçbiri artık sonsuz döngüye dönüşmüyor. "Auto" modu da bazen Türkçe soruya
İngilizce yanıt verebiliyor (dil eşleştirme talimatı %100 güvenilir değil).
Detaylar için README "Known limitations" bölümüne bakın.
