"""Ingestion pipeline: read docs/, chunk them, and index them into the vector store."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import config
from src.chunker import document_to_chunks
from src.vector_store import VectorStore

DOC_EXTENSIONS = (".md", ".markdown", ".txt")


def discover_documents(docs_dir: Path = config.DOCS_DIR) -> List[Path]:
    if not docs_dir.exists():
        return []
    return sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in DOC_EXTENSIONS)


def ingest_file(store: VectorStore, path: Path, embedder=None) -> Tuple[str, int, str]:
    """Chunk and (re-)index a single file. Returns (file_name, chunk_count, title).

    ``embedder`` is optional (see ``src/embedder.py``); when it is ready,
    each chunk also gets a stored embedding for hybrid retrieval.
    """
    raw = path.read_text(encoding="utf-8")
    chunks = document_to_chunks(raw, path.name)
    if not chunks:
        return path.name, 0, ""
    store.remove_document(chunks[0].doc_id)  # idempotent: replace, don't duplicate

    embeddings = None
    if embedder is not None and embedder.ready:
        embeddings = embedder.embed([c.content for c in chunks])

    added = store.add_chunks(chunks, embeddings=embeddings)
    return path.name, added, chunks[0].title


def ingest_all(docs_dir: Path = config.DOCS_DIR, reset: bool = False, embedder=None) -> dict:
    """Ingest every supported document in docs_dir. Returns a summary dict."""
    store = VectorStore()
    if reset:
        store.clear()

    files = discover_documents(docs_dir)
    results = [ingest_file(store, path, embedder=embedder) for path in files]
    total_chunks = sum(count for _, count, _ in results)
    store.close()

    return {
        "docs_dir": str(docs_dir),
        "files": results,
        "file_count": len(files),
        "total_chunks": total_chunks,
    }
