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
TOP_K = 3             # sorgu başına getirilecek chunk sayısı

# --- Foundry Local (yerel LLM) ---
APP_NAME = "local-rag-foundry"
MODEL_ALIAS = os.getenv("FOUNDRY_MODEL", "phi-3.5-mini")
TEMPERATURE = 0.2
MAX_TOKENS = 800

# --- Hibrit retrieval (opsiyonel, zaman/donanım izin verirse) ---
# Foundry Local veya embedding modeli kullanılamıyorsa arama otomatik olarak
# saf TF-IDF'e düşer (bkz. src/embedder.py, src/vector_store.py).
EMBEDDING_MODEL_ALIAS = os.getenv("FOUNDRY_EMBEDDING_MODEL", "qwen3-embedding-0.6b")
HYBRID_TFIDF_WEIGHT = 0.5
HYBRID_EMBEDDING_WEIGHT = 0.5

# --- Dosya yükleme güvenliği ---
ALLOWED_UPLOAD_EXTENSIONS = (".md", ".markdown", ".txt")
MAX_UPLOAD_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB
