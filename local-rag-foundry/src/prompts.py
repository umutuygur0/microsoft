"""System prompt and message assembly for the RAG chat engine."""
from __future__ import annotations

from typing import List, Optional

SYSTEM_PROMPT = """You are a local, offline support assistant that answers strictly from the \
provided context documents.

Behaviour rules:
- Do not invent facts, procedures, measurements, or details that are not in the context.
- If the answer is not contained in the provided context, reply exactly:
  "This information is not available in the local knowledge base."
- If a procedure involves risk, explicitly call out the safety warning.
- Be concise and practical.

Response format (use these headings when applicable):
- Summary (1-2 lines)
- Safety Warnings (only if relevant)
- Step-by-step Guidance
- Reference (document title)"""

NO_CONTEXT_REPLY = "This information is not available in the local knowledge base."


def build_context_block(chunks: List[dict]) -> str:
    if not chunks:
        return "No relevant documents were found in the local knowledge base."
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        relevance = round(chunk["score"] * 100)
        parts.append(
            f"[Source {i}] {chunk['title']} ({chunk['category']}, relevance {relevance}%)\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def build_messages(question: str, chunks: List[dict], history: Optional[List[dict]] = None) -> List[dict]:
    history = history or []
    context = build_context_block(chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-4:]:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": str(turn["content"])})
    messages.append({
        "role": "user",
        "content": (
            f"Context documents:\n\n{context}\n\n---\n\n"
            f"Question: {question}\n\nAnswer using ONLY the context above."
        ),
    })
    return messages
