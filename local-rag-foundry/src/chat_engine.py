"""RAG orchestrator: retrieve -> augment -> generate.

Yields event dicts so a CLI or web layer can render/stream them:
    {"type": "sources", "sources": [...]}
    {"type": "token", "text": "..."}
    {"type": "done"}
    {"type": "error", "message": "..."}
"""
from __future__ import annotations

from typing import Iterator, List, Optional

import config
from src.prompts import NO_CONTEXT_REPLY, build_messages


class ChatEngine:
    def __init__(self, store, model, embedder=None):
        self.store = store
        self.model = model
        self.embedder = embedder

    def ask(self, question: str, history: Optional[List[dict]] = None) -> Iterator[dict]:
        q = (question or "").strip()
        if not q:
            yield {"type": "error", "message": "Empty question."}
            return

        query_embedding = None
        if self.embedder is not None and self.embedder.ready:
            vectors = self.embedder.embed([q])
            query_embedding = vectors[0] if vectors else None

        sources = self.store.search(q, config.TOP_K, query_embedding=query_embedding)
        yield {"type": "sources", "sources": [self._view(s) for s in sources]}

        if not sources:
            yield {"type": "token", "text": NO_CONTEXT_REPLY}
            yield {"type": "done"}
            return

        if not self.model.ready:
            top = sources[0]
            notice = (
                f"[!] The local model is not ready ({self.model.message}).\n\n"
                f"Most relevant passage found:\n\n\"{top['content']}\"\n\n"
                f"- {top['title']}"
            )
            yield {"type": "token", "text": notice}
            yield {"type": "done"}
            return

        messages = build_messages(q, sources, history)
        try:
            for delta in self.model.stream_chat(messages):
                yield {"type": "token", "text": delta}
            yield {"type": "done"}
        except Exception as err:  # noqa: BLE001 - surface any generation failure without crashing
            yield {"type": "error", "message": str(err)}

    @staticmethod
    def _view(chunk: dict) -> dict:
        view = {
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "category": chunk["category"],
            "chunk_index": chunk["chunk_index"],
            "score": round(chunk["score"], 4),
            "content": chunk["content"],
        }
        if chunk.get("semantic_score") is not None:
            view["tfidf_score"] = chunk.get("tfidf_score")
            view["semantic_score"] = chunk.get("semantic_score")
        return view
