import pytest

import config
from src.chunker import Chunk
from src.vector_store import VectorStore


def sample_chunks():
    return [
        Chunk("leak", "Gas Leak Detection", "Safety", 0,
              "Use a calibrated combustible gas detector to find a leak near flanges and joints."),
        Chunk("valve", "Valve Maintenance", "Maintenance", 0,
              "Inject valve grease through the sealant fitting and cycle the valve open and close."),
        Chunk("ppe", "PPE Requirements", "Safety", 0,
              "A personal H2S monitor is mandatory in sour areas with escape breathing apparatus."),
    ]


@pytest.fixture
def store():
    s = VectorStore(":memory:")
    s.add_chunks(sample_chunks())
    yield s
    s.close()


def test_add_and_count(store):
    assert store.count() == 3


def test_search_ranks_relevant_first(store):
    results = store.search("how do I detect a gas leak", top_k=3)
    assert len(results) >= 1
    assert results[0]["doc_id"] == "leak"
    assert results[0]["score"] > 0


def test_search_respects_top_k(store):
    assert len(store.search("gas safety valve", top_k=1)) == 1


def test_search_unrelated_query_returns_empty(store):
    assert store.search("photosynthesis chlorophyll biology") == []


def test_remove_document_then_readd_is_idempotent(store):
    store.remove_document("valve")
    assert store.count() == 2
    store.add_chunks([sample_chunks()[1]])
    assert store.count() == 3


def test_list_documents_groups_by_doc_id(store):
    docs = store.list_documents()
    assert len(docs) == 3
    assert all(d["chunks"] == 1 for d in docs)


# --- Hybrid (BM25 + semantic) retrieval ---

def test_pure_bm25_search_ignores_embeddings_when_no_query_embedding(store):
    # sample_chunks() has no embeddings at all; behaviour must be unchanged.
    results = store.search("how do I detect a gas leak")
    assert results[0]["doc_id"] == "leak"
    assert results[0]["semantic_score"] is None


def test_hybrid_search_finds_semantic_match_with_no_keyword_overlap():
    store = VectorStore(":memory:")
    chunks = [
        Chunk("a", "Doc A", "General", 0, "zzz yyy xxx totally distinct vocabulary"),
        Chunk("b", "Doc B", "General", 0, "completely unrelated filler content words"),
    ]
    # "a" is embedded close to the query embedding, "b" is orthogonal.
    store.add_chunks(chunks, embeddings=[[1.0, 0.0], [0.0, 1.0]])

    # No shared keywords at all -> pure BM25 finds nothing.
    assert store.search("qqq www") == []

    # With a query embedding aligned to "a", hybrid search finds it anyway.
    results = store.search("qqq www", query_embedding=[0.9, 0.1])
    assert results
    assert results[0]["doc_id"] == "a"
    assert results[0]["semantic_score"] > 0
    store.close()


def test_hybrid_search_blends_scores_when_both_signals_present():
    store = VectorStore(":memory:")
    chunks = [
        Chunk("leak", "Gas Leak Detection", "Safety", 0,
              "Use a calibrated combustible gas detector to find a leak near flanges."),
    ]
    store.add_chunks(chunks, embeddings=[[1.0, 0.0]])

    results = store.search("how do I detect a gas leak", query_embedding=[1.0, 0.0])
    assert results[0]["bm25_score"] > 0
    assert results[0]["semantic_score"] == pytest.approx(1.0)
    # blended score sits between the two weighted components, not just one of them
    assert 0 < results[0]["score"] <= 1.0
    store.close()


def test_hybrid_search_discounts_a_candidate_missing_its_own_embedding():
    # Regression test: a chunk added while the embedder was unavailable (so
    # it has no stored embedding) must not be scored on bm25_score alone at
    # full weight in hybrid mode -- that would unfairly outrank a chunk that
    # *does* have an embedding and is correctly discounted to its blended
    # share. Both chunks share identical text, so any score difference here
    # is purely from how the missing embedding is handled.
    store = VectorStore(":memory:")
    chunks = [
        Chunk("embedded", "Embedded Doc", "General", 0, "detect a gas leak near the flange"),
        Chunk("bare", "Unembedded Doc", "General", 0, "detect a gas leak near the flange"),
    ]
    store.add_chunks(chunks, embeddings=[[1.0, 0.0], None])

    results = store.search("detect a gas leak", query_embedding=[1.0, 0.0])
    by_doc = {r["doc_id"]: r for r in results}

    assert by_doc["bare"]["semantic_score"] is None
    # Same bm25_score for both (identical text) -> the embedded chunk's
    # blended score and the bare chunk's bm25-only-but-discounted score
    # must use the same weight, so they end up equal rather than the bare
    # chunk winning on an un-discounted bm25_score.
    assert by_doc["embedded"]["bm25_score"] == by_doc["bare"]["bm25_score"]
    expected_bare_score = config.HYBRID_BM25_WEIGHT * by_doc["bare"]["bm25_score"]
    assert by_doc["bare"]["score"] == pytest.approx(expected_bare_score, abs=1e-3)
    store.close()


def test_missing_embedding_column_is_added_on_open(tmp_path):
    import sqlite3

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'General',
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            tf TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    # Opening a pre-hybrid-retrieval database must not crash; it should
    # transparently gain the new column.
    store = VectorStore(str(db_path))
    store.add_chunks([Chunk("x", "X", "General", 0, "some content")])
    assert store.count() == 1
    store.close()
