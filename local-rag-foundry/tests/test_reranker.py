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


class _NoSignalCrossEncoderModel:
    """Simulates the real failure found in a live 98-question audit
    (TEST_REPORT.md §15/§16): every candidate scores deep negative because
    the model has no real signal for this query, and its relative order
    among them is just noise."""

    def predict(self, pairs):
        # Distinct-but-all-deep-negative scores -- there IS a "ranking" here,
        # it's just meaningless, which is exactly the failure mode.
        return [-3.5 - 0.1 * i for i in range(len(pairs))]


def test_rerank_falls_back_to_hybrid_order_when_confidence_is_too_low():
    reranker = CrossEncoderReranker()
    reranker._model = _NoSignalCrossEncoderModel()
    reranker.state = "ready"
    # candidate[0] is the correct, hybrid-search-ranked-first chunk (highest
    # "score"); the fake cross-encoder above would rank it first too by
    # coincidence of list order, so give it the LOWEST cross-encoder score to
    # prove the fallback path (not the cross-encoder path) produced this order.
    candidates = [
        {"doc_id": "correct", "content": "c", "score": 0.9},
        {"doc_id": "wrong-a", "content": "a", "score": 0.4},
        {"doc_id": "wrong-b", "content": "b", "score": 0.2},
    ]
    result = reranker.rerank("belirsiz bir soru", candidates, top_k=3)
    assert [c["doc_id"] for c in result] == ["correct", "wrong-a", "wrong-b"]


def test_rerank_trusts_cross_encoder_when_confidence_clears_the_floor():
    reranker = CrossEncoderReranker()

    class _OneConfidentMatch:
        def predict(self, pairs):
            return [-3.8, 0.5, -3.9]  # one candidate clearly above the floor

    reranker._model = _OneConfidentMatch()
    reranker.state = "ready"
    candidates = [
        {"doc_id": "hybrid-favourite", "content": "a", "score": 0.9},
        {"doc_id": "cross-encoder-favourite", "content": "b", "score": 0.1},
        {"doc_id": "least-relevant", "content": "c", "score": 0.5},
    ]
    result = reranker.rerank("query", candidates, top_k=3)
    assert result[0]["doc_id"] == "cross-encoder-favourite"
