"""Optional cross-encoder reranking pass over hybrid-search candidates.

A bi-encoder (the embedding model in src/embedder.py) scores a query and a
chunk by encoding them *separately* and comparing vectors -- fast, but it
never lets the model actually look at the query and the passage together. A
cross-encoder takes the (query, passage) pair as one input and outputs a
single relevance score, which is slower per pair but meaningfully more
accurate -- the standard "retrieve broad, then rerank precisely" two-stage
pattern. See src/chat_engine.py for how this fits after VectorStore.search().

Graceful degradation: if sentence-transformers or the model download is
unavailable, ``ready`` stays False and callers should just use the hybrid
search's own ordering.
"""
from __future__ import annotations

from typing import List

import config


class CrossEncoderReranker:
    def __init__(self):
        self.state = "idle"
        self.message = "Not started"
        self._model = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def init(self) -> None:
        """Load the configured cross-encoder model. Safe to call more than once."""
        if self.ready:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            self.state = "unavailable"
            self.message = "sentence-transformers is not installed."
            return

        try:
            self.state = "starting"
            self._model = CrossEncoder(config.RERANKER_MODEL)
            self.state = "ready"
            self.message = f"Cross-encoder reranking active · {config.RERANKER_MODEL}"
        except Exception as err:  # noqa: BLE001 - never let this block the rest of the app
            self.state = "unavailable"
            self.message = f"Reranker unavailable ({err}); using hybrid order only."

    def rerank(self, query: str, candidates: List[dict], top_k: int) -> List[dict]:
        """Re-score and re-sort ``candidates`` by cross-encoder relevance to
        ``query``, returning the top ``top_k``.

        Each returned candidate dict gets an added ``rerank_score`` key.
        Falls back to the first ``top_k`` candidates unchanged (no
        ``rerank_score`` added) if the reranker isn't ready or there is
        nothing to rerank -- callers should not assume this key is always
        present.
        """
        if not self.ready or not candidates:
            return candidates[:top_k]

        pairs = [(query, c["content"]) for c in candidates]
        scores = self._model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        # A 98-question live audit (TEST_REPORT.md §15/§16) found this model
        # sometimes has no real signal for a query at all -- every candidate
        # scores deep negative (observed as low as -4) with no clear winner
        # -- and its relative ordering among "all bad options" is then just
        # noise. In one measured case this noise demoted the hybrid search's
        # correctly-ranked #1 candidate out of the top results entirely. When
        # even the *best* candidate doesn't clear config.RERANK_MIN_CONFIDENCE,
        # trust the pre-rerank hybrid order (each candidate's own "score")
        # instead of the cross-encoder's unreliable ranking for this query.
        if max(c["rerank_score"] for c in candidates) < config.RERANK_MIN_CONFIDENCE:
            return sorted(candidates, key=lambda c: c["score"], reverse=True)[:top_k]

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
