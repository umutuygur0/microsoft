import pytest

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


# --- Hybrid (TF-IDF + semantic) retrieval ---

def test_pure_tfidf_search_ignores_embeddings_when_no_query_embedding(store):
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

    # No shared keywords at all -> pure TF-IDF finds nothing.
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
    assert results[0]["tfidf_score"] > 0
    assert results[0]["semantic_score"] == pytest.approx(1.0)
    # blended score sits between the two weighted components, not just one of them
    assert 0 < results[0]["score"] <= 1.0
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
