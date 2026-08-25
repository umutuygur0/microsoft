"""Merkezi konfigürasyon: model adı, chunking/retrieval parametreleri, dosya yolları."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT_DIR / "data" / "docs"))
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "knowledge.db"

# --- Retrieval (TF-IDF) ---
CHUNK_SIZE = 200      # yaklaşık kelime sayısı / chunk
CHUNK_OVERLAP = 25    # chunk'lar arası örtüşen kelime sayısı
TOP_K = 5             # sorgu başına getirilecek chunk sayısı

# --- LLM sağlayıcısı seçimi ---
# "foundry" (Microsoft Foundry Local) veya "ollama". İkisi de tamamen yerel/
# çevrimdışı çalışır; fark sadece hangi runtime'ın modeli servis ettiği.
# qwen3 ailesinin thinking modunu Foundry Local'in SDK'sında API seviyesinde
# kapatacak hiçbir yol bulunamadı (bkz. TEST_REPORT.md — ne resmi bir alan,
# ne prefill hilesi, ne de chat_template_kwargs enjeksiyonu işe yaradı);
# Ollama ise "think=False" parametresini native olarak destekliyor. Bu proje
# bu yüzden Ollama'ya (llama3.1:8b + bge-m3) geçirildi; disk alanını boşaltmak
# için artık kullanılmayan Foundry Local modelleri kaldırıldı.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# --- Ollama (yerel LLM/embedding motoru) ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
# Meta resmi bir "Llama embedding" modeli yayınlamıyor (Llama üretken/
# decoder modeli, embedding ayrı bir mimari gerektirir) — bu yüzden ayrı,
# çok-dilli (Türkçe dahil) bir embedding modeli seçildi.
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
OLLAMA_KEEP_ALIVE = "10m"

# --- Foundry Local (yerel LLM) ---
APP_NAME = "local-rag-foundry"
MODEL_ALIAS = os.getenv("FOUNDRY_MODEL", "qwen2.5-7b")
TEMPERATURE = 0.2
MAX_TOKENS = 800
# Qwen3-family models default to an internal <think>...</think> reasoning
# trace before the real answer, drawn from the SAME token budget as the
# answer itself. The Foundry Local SDK's ChatClientSettings has no
# enable_thinking/extra_body field to disable this at the API level (only
# temperature/max_tokens/penalties/top_p/top_k/response_format/tool_choice
# are supported), so the only accessible lever is a much larger budget for
# these models specifically -- otherwise thinking alone can exhaust
# MAX_TOKENS and leave zero room for the actual answer (observed live: a
# plain English question returned a completely empty response). Paired with
# the "/no_think" prompt suffix (see src/prompts.py) and stripping any
# <think>...</think> block from the final text (see src/chat_engine.py).
THINKING_MODEL_MAX_TOKENS = 3000
# Küçük modellerde bilinen bir arıza modu: aynı token/kelime/cümleyi sonsuz
# tekrarlama ("repetition collapse"). DENEYSEL BULGU: bu değerleri çok
# agresif ayarlamak (0.8/0.4 denendi) tekrarı azaltmak yerine modeli tutarlı
# tekrardan tutarsız "kelime salatasına" düşürüyor — daha kötü bir sonuç.
# Ilımlı bir ayar + src/foundry_client.py'deki tekrar-algılama/geri-dönüş
# güvenlik ağı (asıl güvenilir savunma hattı) birlikte kullanılıyor.
FREQUENCY_PENALTY = 0.5
PRESENCE_PENALTY = 0.1

# --- Hibrit retrieval (opsiyonel, zaman/donanım izin verirse) ---
# Foundry Local veya embedding modeli kullanılamıyorsa arama otomatik olarak
# saf TF-IDF'e düşer (bkz. src/embedder.py, src/vector_store.py).
EMBEDDING_MODEL_ALIAS = os.getenv("FOUNDRY_EMBEDDING_MODEL", "qwen3-embedding-0.6b")
# 30 soruluk (kolay/orta/zor) bir retrieval denetimi sırasında ölçüldü: eşit
# ağırlık (0.5/0.5), içeriği zayıf ama anahtar kelime yoğunluğu yüksek bir
# dokümanın (OSHA nav-menü metni), gerçek/sayısal cevabı barındıran ama daha
# az kelime tekrarı olan doğru chunk'ı geride bırakmasına yol açıyordu — o
# doğru chunk'ın semantik skoru zaten en yüksekti, sorun sadece anahtar-kelime
# skorunun ona verdiği ağırlıktı. 0.35/0.65 aynı 30 soruda regresyon
# yaratmadan isabeti artırdı (bkz. TEST_REPORT.md). Sparse taraf sonradan
# TF-IDF'ten BM25'e yükseltildi (bkz. src/bm25.py) — ağırlık oranı, BM25'in
# kendi normalize edilmiş skoruna karşı yeniden doğrulanmadan aynen taşındı.
HYBRID_BM25_WEIGHT = 0.35
HYBRID_EMBEDDING_WEIGHT = 0.65

# --- Cross-encoder reranking (opsiyonel ikinci aşama) ---
# Hibrit arama (yukarıdaki TF-IDF+embedding) daha fazla aday getirir
# (RERANK_CANDIDATE_K), bir cross-encoder bu adayları sorguyla birlikte
# (query, chunk) çifti olarak yeniden puanlayıp TOP_K'ya indirger — bir
# bi-encoder'ın (embedding) ayrı ayrı kodladığı iki vektörün kosinüsüne göre
# çok daha isabetli bir alaka sıralaması verir. sentence-transformers/torch
# kurulu değilse veya model indirilemezse arama otomatik olarak sadece
# hibrit sıralamaya döner (bkz. src/reranker.py).
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANK_CANDIDATE_K = 15

# --- Dosya yükleme güvenliği ---
# .pdf/.docx metinleri src/document_readers.py ile çıkarılır; ikili dosya
# olduklarından boyut limiti biraz daha yüksek tutuluyor.
ALLOWED_UPLOAD_EXTENSIONS = (".md", ".markdown", ".txt", ".pdf", ".docx")
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
