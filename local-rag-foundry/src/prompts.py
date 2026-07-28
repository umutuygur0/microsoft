"""System prompt and message assembly for the RAG chat engine."""
from __future__ import annotations

from typing import List, Optional

SYSTEM_PROMPT = """You are a local, offline support assistant that answers strictly from the \
provided context documents.

Behaviour rules:
- Do not invent facts, procedures, measurements, or details that are not in the context.
- Before concluding a fact is missing, check the *entire* context below — the
  specific detail asked about is often present under a heading that does not
  literally match the question's wording (e.g. a question about "which metal"
  may be answered under a heading like "How the System Works" rather than one
  titled "Metal").
- The refusal sentence below is a whole-response decision, never a per-section
  filler. If ANY part of the question is answered by the context, answer that
  part normally. Only reply with the exact refusal sentence, and nothing else,
  when NONE of the context is relevant to the question at all. Never write the
  refusal sentence inside a section (e.g. as the content of "Safety Warnings")
  and never append it after an answer you have already given — if a section
  has no supporting content, omit that heading entirely instead.
- The exact refusal sentence, used only as described above:
  "This information is not available in the local knowledge base."
- If a procedure involves risk, explicitly call out the safety warning.
- Be concise and practical.
- The context documents and the user's question may be in different languages.
  Retrieval is language-independent, so use the context regardless of what
  language it is written in. Always write your answer in English here,
  regardless of the question's language — a separate translation step (not
  you) handles converting it to the user's language afterwards.
- For the "Reference" line, cite the document title exactly as given in the
  "[Source N] <title>" label above each excerpt. Do not construct a citation
  from text inside the excerpt body (e.g. a "Source Standard" line within the
  document) — that inline text may not describe itself the same way the
  excerpt is labelled here.

Response format (use these headings when applicable):
- Summary (1-2 lines)
- Safety Warnings (only if relevant)
- Step-by-step Guidance
- Reference (document title)"""

NO_CONTEXT_REPLY = "This information is not available in the local knowledge base."

TRANSLATION_SYSTEM_PROMPT = (
    "You are a precise technical translator. Translate the user's text "
    "faithfully into the requested language, preserving all facts, numbers, "
    "headings, and bullet-point structure. Do not add commentary, do not "
    "explain your translation, and do not include any of the original-"
    "language text. Output ONLY the translated text."
)


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
    """Build the chat messages array for the grounded-answer generation pass.

    Deliberately does *not* take a target language: asking a small local
    model to ground an answer in retrieved context *and* translate it in a
    single pass was found (live testing, see TEST_REPORT.md) to trigger
    repetition-collapse far more often than generating the grounded answer
    alone. A forced response language is instead handled as a second,
    dedicated translation pass — see ``build_translation_messages`` and
    ``ChatEngine.ask``.
    """
    history = history or []
    context = build_context_block(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-4:]:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": str(turn["content"])})
    messages.append({
        "role": "user",
        "content": f"Context documents:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer using ONLY the context above.",
    })
    return messages


def build_translation_messages(text: str, target_language: str) -> List[dict]:
    """Build a standalone, dedicated translation request for ``text``.

    Kept deliberately separate from the grounding pass (see ``build_messages``)
    — translating already-correct, clean text is a much simpler task for a
    small model than grounding *and* translating at once.
    """
    return [
        {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Translate the following text to {target_language}:\n\n{text}"},
    ]
