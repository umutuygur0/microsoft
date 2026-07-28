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

# --- Foundry Local (yerel LLM) ---
APP_NAME = "local-rag-foundry"
MODEL_ALIAS = os.getenv("FOUNDRY_MODEL", "qwen2.5-7b")
TEMPERATURE = 0.2
MAX_TOKENS = 800
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
HYBRID_TFIDF_WEIGHT = 0.5
HYBRID_EMBEDDING_WEIGHT = 0.5

# --- Dosya yükleme güvenliği ---
# .pdf/.docx metinleri src/document_readers.py ile çıkarılır; ikili dosya
# olduklarından boyut limiti biraz daha yüksek tutuluyor.
ALLOWED_UPLOAD_EXTENSIONS = (".md", ".markdown", ".txt", ".pdf", ".docx")
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
