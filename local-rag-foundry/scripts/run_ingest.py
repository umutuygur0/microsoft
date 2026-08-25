"""CLI entry point for (re-)indexing the knowledge base.

    python scripts/run_ingest.py             # add/refresh documents (idempotent per file)
    python scripts/run_ingest.py --reset     # wipe the store first, then ingest
    python scripts/run_ingest.py --no-embed  # skip semantic embeddings (TF-IDF only, faster)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.ingest import ingest_all


def main() -> None:
    reset = "--reset" in sys.argv
    no_embed = "--no-embed" in sys.argv

    if not config.DOCS_DIR.exists():
        print(f"docs/ folder not found at {config.DOCS_DIR}")
        sys.exit(1)

    embedder = None
    if not no_embed:
        if config.LLM_PROVIDER == "ollama":
            from src.ollama_embedder import OllamaEmbedder
            embedder = OllamaEmbedder()
        else:
            from src.embedder import LocalEmbedder
            embedder = LocalEmbedder()
        print("Loading embedding model for hybrid retrieval (skip with --no-embed)...")
        embedder.init()
        print(f"  {embedder.message}")

    summary = ingest_all(reset=reset, embedder=embedder)
    if summary["file_count"] == 0:
        print("No .md / .markdown / .txt documents found.")
        sys.exit(1)

    for file_name, chunk_count, title in summary["files"]:
        if chunk_count == 0:
            print(f"  - {file_name}: no content, skipped")
        else:
            print(f"  - {file_name} -> {chunk_count} chunk(s)  [{title}]")

    print(f"\nIndexed {summary['total_chunks']} chunk(s) from {summary['file_count']} document(s).")
    print(f"Database: {config.DB_PATH}")


if __name__ == "__main__":
    main()
