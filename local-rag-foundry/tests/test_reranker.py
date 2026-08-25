from src.reranker import CrossEncoderReranker


class _FakeCrossEncoderModel:
    """Scores each (query, content) pair by how many query words appear in
    the content -- deterministic and cheap, just enough to prove reordering
    happens without downloading a real model in the unit test suite."""

    def predict(self, pairs):
        scores = []
        for query, content in pairs:
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            scores.append(float(len(query_words & content_words)))
        return scores


def make_reranker_with_fake_model() -> CrossEncoderReranker:
    reranker = CrossEncoderReranker()
    reranker._model = _FakeCrossEncoderModel()
    reranker.state = "ready"
    return reranker


def test_not_ready_returns_first_top_k_unchanged():
    reranker = CrossEncoderReranker()
    assert not reranker.ready
    candidates = [
        {"content": "a"}, {"content": "b"}, {"content": "c"},
    ]
    result = reranker.rerank("query", candidates, top_k=2)
    assert result == candidates[:2]
    assert "rerank_score" not in result[0]


def test_empty_candidates_returns_empty():
    reranker = make_reranker_with_fake_model()
    assert reranker.rerank("query", [], top_k=5) == []


def test_rerank_reorders_by_cross_encoder_score():
    reranker = make_reranker_with_fake_model()
    candidates = [
        {"doc_id": "off-topic", "content": "completely unrelated filler text"},
        {"doc_id": "on-topic", "content": "detect a gas leak near the flange"},
        {"doc_id": "partial", "content": "gas compressor maintenance schedule"},
    ]
    result = reranker.rerank("how do I detect a gas leak", candidates, top_k=3)

    assert [c["doc_id"] for c in result] == ["on-topic", "partial", "off-topic"]
    assert all("rerank_score" in c for c in result)
    assert result[0]["rerank_score"] >= result[1]["rerank_score"] >= result[2]["rerank_score"]


def test_rerank_truncates_to_top_k():
    reranker = make_reranker_with_fake_model()
    candidates = [{"doc_id": str(i), "content": f"word{i}"} for i in range(10)]
    result = reranker.rerank("word1 word2", candidates, top_k=3)
    assert len(result) == 3
