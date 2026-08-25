"""System prompt and message assembly for the RAG chat engine."""
from __future__ import annotations

from typing import List, Optional

import config

# Qwen3-family models default to an internal <think>...</think> reasoning
# trace before their real answer. That trace was observed (with qwen3-8b,
# see TEST_REPORT.md) to itself collapse into repetition, exceeding even the
# sentence-level repetition guard. Qwen3's chat template recognises a
# literal "/no_think" directive in the conversation to skip the trace
# entirely. Gated to Qwen3-family models specifically (rather than always
# appended) so it has no effect — and can't confuse — a different
# configured model such as qwen2.5-7b.
_SUPPRESS_THINKING_SUFFIX = " /no_think"


def _maybe_suppress_thinking(text: str) -> str:
    if config.MODEL_ALIAS.lower().startswith("qwen3"):
        return text + _SUPPRESS_THINKING_SUFFIX
    return text

SYSTEM_PROMPT = """You are a local, offline support assistant that answers strictly from the \
provided context documents.

Behaviour rules:
- Do not invent facts, procedures, measurements, or details that are not in the context.
- A specific code, standard, or section number (e.g. "29 CFR 1910.147") must
  only be stated if that exact number appears verbatim somewhere in the
  context. If the context discusses the topic but never states the specific
  number being asked for, say the number is not given in the available
  excerpts instead of guessing one — live testing found this model will
  otherwise fabricate a plausible-looking but wrong number rather than admit
  it is not present.
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
- Be thorough and complete: cover every relevant point, step, and number
  found in the context for this question, in full sentences — do not
  artificially shorten the answer or stop after the first relevant detail
  when the context has more. Being complete is more important than being
  brief.
- The context documents and the user's question may be in different languages.
  Retrieval is language-independent, so use the context regardless of what
  language it is written in. Always write your answer in English here,
  regardless of the question's language — a separate translation step (not
  you) handles converting it to the user's language afterwards.
- Do not cite or name your sources yourself, in any form, anywhere in the
  answer — no "Reference" line, no inline mention of an excerpt number or
  document title, nothing. Source attribution is handled separately, outside
  your answer, from the actual retrieval metadata — not from anything you
  write. Repeated live testing found this model cannot be trusted to name
  its source correctly (it invents titles, cites the wrong excerpt, or
  echoes internal labels verbatim) even when explicitly instructed on
  exactly how to do it, so it is no longer asked to try at all.

Response format (use these headings when applicable):
- Summary (1-2 lines)
- Safety Warnings (only if relevant)
- Step-by-step Guidance"""

NO_CONTEXT_REPLY = "This information is not available in the local knowledge base."
# Hand-written, not machine-translated: this exact fixed sentence is worth
# getting right once rather than trusting either the chat LLM or the local
# MarianMT model (src/translator.py) to phrase it consistently every time.
# Wording deliberately keeps "bulunmuyor"/"bilgi"/"yerel" together so it's
# still recognised by ChatEngine._opens_with_refusal's Turkish heuristic.
NO_CONTEXT_REPLY_TR = "Bu bilgi yerel bilgi tabanında bulunmuyor."

TRANSLATION_SYSTEM_PROMPT = (
    "You are a precise technical translator. Translate the user's text "
    "faithfully into the requested language, preserving all facts, numbers, "
    "headings, and bullet-point structure. Do not add commentary, do not "
    "explain your translation, and do not include any of the original-"
    "language text. Output ONLY the translated text."
)


def build_context_block(chunks: List[dict]) -> str:
    """Render retrieved chunks as plain excerpts for the model to read.

    Deliberately avoids a "[Source N]"-style bracketed label: live testing
    (see TEST_REPORT.md) found the model would echo that exact bracketed
    pattern verbatim into its own answer despite explicit instructions not
    to, presumably because it visually resembles a citation marker worth
    reproducing. This wording still gives the model everything it needs to
    ground its answer, without a token pattern for it to copy — see
    ``SYSTEM_PROMPT``: the model is no longer asked to cite anything itself,
    so it doesn't need a citation label to imitate at all.
    """
    if not chunks:
        return "No relevant documents were found in the local knowledge base."
    parts = []
    for chunk in chunks:
        parts.append(f"Excerpt from \"{chunk['title']}\" ({chunk['category']}):\n{chunk['content']}")
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
        "content": _maybe_suppress_thinking(
            f"Context documents:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer using ONLY the context above."
        ),
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
        {"role": "user", "content": _maybe_suppress_thinking(
            f"Translate the following text to {target_language}:\n\n{text}"
        )},
    ]
