"""SQLite-backed store for document chunks with hybrid BM25 + semantic retrieval.

Each chunk is persisted with its raw term-frequency map (JSON) and, optionally,
a dense embedding vector (JSON list of floats). At query time an in-memory
cache builds an inverted index and BM25 statistics (per-chunk length, corpus
average length, IDF) from those stored term-frequency maps directly — no
extra column or re-ingestion needed when the scoring formula changes.

Retrieval is BM25 by default (replaces the original plain TF-IDF + cosine
similarity — see src/bm25.py's docstring for why). When a caller also
supplies a query embedding (see ``src/embedder.py``) *and* the store has
embeddings on file, the two scores are blended (``config.HYBRID_BM25_WEIGHT``
/ ``HYBRID_EMBEDDING_WEIGHT``) so that semantically related chunks can
surface even without keyword overlap. Without a query embedding, behaviour
is identical to pure BM25.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections import Counter
from typing import Dict, List, Optional

import config
from src.bm25 import bm25_score, compute_bm25_idf
from src.tfidf import dense_cosine_similarity, term_frequency
from src.chunker import Chunk


class VectorStore:
    def __init__(self, db_path: Optional[str] = None):
        path = str(db_path) if db_path is not None else str(config.DB_PATH)
        if path != ":memory:":
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + an explicit lock: a single VectorStore
        # instance is shared across Streamlit script re-runs, which do not
        # all execute on the same thread.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._migrate()
        self._cache: Optional[dict] = None  # rebuilt lazily after every write

    def _migrate(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      TEXT NOT NULL,
                title       TEXT NOT NULL,
                category    TEXT NOT NULL DEFAULT 'General',
                chunk_index INTEGER NOT NULL,
                content     TEXT NOT NULL,
                tf          TEXT NOT NULL,
                embedding   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """
        )
        # Back-compat: databases created before hybrid retrieval was added
        # won't have this column yet.
        existing_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(chunks)")}
        if "embedding" not in existing_cols:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN embedding TEXT")
        self.conn.commit()

    def clear(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM chunks")
            self.conn.commit()
            self._cache = None

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self.conn.commit()
            self._cache = None

    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[List[Optional[List[float]]]] = None) -> int:
        """Insert chunks, optionally with one pre-computed embedding vector each.

        ``embeddings`` (if given) must have the same length as ``chunks``; a
        ``None`` entry means "no embedding for this chunk" and it will simply
        be excluded from the semantic half of hybrid search.
        """
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("embeddings must have the same length as chunks")

        rows = []
        for i, c in enumerate(chunks):
            embedding = embeddings[i] if embeddings is not None else None
            rows.append((
                c.doc_id, c.title, c.category, c.chunk_index, c.content,
                json.dumps(term_frequency(c.content)),
                json.dumps(embedding) if embedding is not None else None,
            ))
        with self._lock:
            self.conn.executemany(
                "INSERT INTO chunks (doc_id, title, category, chunk_index, content, tf, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self.conn.commit()
            self._cache = None
        return len(chunks)

    def count(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]

    def list_documents(self) -> List[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT doc_id, title, category, COUNT(*) AS chunks "
                "FROM chunks GROUP BY doc_id ORDER BY title"
            ).fetchall()
        return [dict(r) for r in rows]

    def _ensure_cache_locked(self) -> dict:
        """Build the in-memory BM25 (+ embedding) cache. Caller must hold ``self._lock``."""
        if self._cache is not None:
            return self._cache

        rows = self.conn.execute(
            "SELECT doc_id, title, category, chunk_index, content, tf, embedding FROM chunks"
        ).fetchall()

        tf_maps = [Counter(json.loads(r["tf"])) for r in rows]
        idf = compute_bm25_idf(tf_maps)
        doc_lens = [sum(tf.values()) for tf in tf_maps]
        avg_doc_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0

        inverted: Dict[str, set] = {}
        records = []
        has_embeddings = False
        for i, row in enumerate(rows):
            tf = tf_maps[i]
            for term in tf:
                inverted.setdefault(term, set()).add(i)
            embedding = json.loads(row["embedding"]) if row["embedding"] else None
            has_embeddings = has_embeddings or embedding is not None
            records.append({
                "doc_id": row["doc_id"],
                "title": row["title"],
                "category": row["category"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "tf": tf,
                "doc_len": doc_lens[i],
                "embedding": embedding,
            })

        self._cache = {
            "records": records,
            "inverted": inverted,
            "idf": idf,
            "avg_doc_len": avg_doc_len,
            "has_embeddings": has_embeddings,
        }
        return self._cache

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K,
        query_embedding: Optional[List[float]] = None,
    ) -> List[dict]:
        """Rank chunks by BM25, blended with semantic similarity when
        ``query_embedding`` is given and the store has embeddings on file.
        """
        query_tf = term_frequency(query)
        if not query_tf and query_embedding is None:
            return []

        with self._lock:
            cache = self._ensure_cache_locked()
            records = cache["records"]
            if not records:
                return []

            bm25_candidates: set = set()
            for term in query_tf:
                bm25_candidates.update(cache["inverted"].get(term, ()))

            # Semantic similarity isn't limited to keyword-overlap candidates —
            # that's the whole point of adding it — so when a query embedding
            # is available, every chunk with a stored embedding is considered.
            use_hybrid = query_embedding is not None and cache["has_embeddings"]
            candidate_indices = bm25_candidates | (set(range(len(records))) if use_hybrid else set())

            # BM25 is unbounded (unlike cosine similarity's [0, 1] range), so
            # blending it with semantic score requires normalizing it first —
            # min-max against the max BM25 score *within this candidate set*
            # (a per-query normalization, not a stored/global one).
            raw_bm25: Dict[int, float] = {}
            if query_tf:
                for idx in candidate_indices:
                    record = records[idx]
                    raw_bm25[idx] = bm25_score(
                        query_tf.keys(), record["tf"], record["doc_len"], cache["avg_doc_len"], cache["idf"]
                    )
            max_bm25 = max(raw_bm25.values(), default=0.0)

            scored = []
            for idx in candidate_indices:
                record = records[idx]
                bm25_norm = (raw_bm25.get(idx, 0.0) / max_bm25) if max_bm25 > 0 else 0.0

                if use_hybrid:
                    # A candidate missing its own embedding (e.g. a chunk added
                    # while the embedder was briefly unavailable) must not be
                    # scored on bm25_score alone at full weight — that would
                    # unfairly outrank chunks that *do* have an embedding and
                    # are correctly discounted to their blended share. Treat a
                    # missing embedding as a semantic score of 0 for scoring
                    # purposes, while still reporting it as None (not 0.0) so
                    # the caller can tell "not embedded" apart from "embedded
                    # but no match".
                    has_embedding = record["embedding"] is not None
                    semantic_score = (
                        max(0.0, dense_cosine_similarity(query_embedding, record["embedding"]))
                        if has_embedding else 0.0
                    )
                    final_score = (
                        config.HYBRID_BM25_WEIGHT * bm25_norm
                        + config.HYBRID_EMBEDDING_WEIGHT * semantic_score
                    )
                    extra = {
                        "bm25_score": round(bm25_norm, 4),
                        "semantic_score": round(semantic_score, 4) if has_embedding else None,
                    }
                else:
                    final_score = bm25_norm
                    extra = {"bm25_score": round(bm25_norm, 4), "semantic_score": None}

                if final_score > 0:
                    scored.append({**record, "score": final_score, **extra})

        scored.sort(key=lambda r: r["score"], reverse=True)
        for r in scored:
            r.pop("tf", None)
            r.pop("doc_len", None)
            r.pop("embedding", None)
        return scored[:top_k]

    def close(self) -> None:
        with self._lock:
            self.conn.close()
