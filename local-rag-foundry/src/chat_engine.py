"""RAG orchestrator: retrieve -> augment -> generate.

Yields event dicts so a CLI or web layer can render/stream them:
    {"type": "sources", "sources": [...]}
    {"type": "token", "text": "..."}
    {"type": "notice", "message": "..."}  # UI-only; must NOT be folded into
                                           # stored conversation content (see
                                           # FoundryClient.last_response_truncated)
    {"type": "done"}
    {"type": "error", "message": "..."}
"""
from __future__ import annotations

import re
from typing import Iterator, List, Optional

import config
from src.prompts import NO_CONTEXT_REPLY, build_messages, build_translation_messages
from src.tfidf import TURKISH_STOPWORDS

_REPETITION_NOTICE = "Response stopped early — the model started repeating itself."

_TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_ASCII_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
_TURKISH_STOPWORDS_FOLDED = {w.translate(_ASCII_FOLD) for w in TURKISH_STOPWORDS}


def _looks_turkish(text: str) -> bool:
    """Lightweight heuristic: Turkish-specific letters or common Turkish words.

    No language-detection library is used anywhere in this project (see
    README) — cross-lingual retrieval already works via multilingual
    embeddings without one. This only decides whether an "Auto" mode question
    should go through the same grounding-then-translate flow already used for
    an explicitly forced language (see ``ask``), instead of the model trying
    to both extract facts from English context *and* answer in Turkish in a
    single pass — live testing showed that combination is unreliable (wrong
    facts, or a spurious "not available" refusal despite good context).

    Matching is done both on the raw words and on an ASCII-folded copy (ç/ğ/
    ı/ö/ş/ü -> c/g/i/o/s/u), because Turkish is very often typed without its
    diacritics ("hattini" for "hattını", "icin" for "için") — live testing
    showed the undiacritic'd form is common enough that skipping it left
    exactly the Turkish questions this exists to catch undetected.
    """
    if any(ch in _TURKISH_CHARS for ch in text):
        return True
    words = _WORD_RE.findall(text.lower())
    if any(w in TURKISH_STOPWORDS for w in words):
        return True
    folded_words = {w.translate(_ASCII_FOLD) for w in words}
    return bool(folded_words & _TURKISH_STOPWORDS_FOLDED)

# A truncated grounding pass isn't necessarily a failure worth abandoning:
# the guard was observed to sometimes trip on a handful of trailing blank
# lines *after* an otherwise complete, well-formed answer. Only skip
# translation when the draft is both truncated *and* too short to be the
# real answer (e.g. failed almost immediately) — a truncated-but-substantial
# draft is stripped of trailing noise and translated normally.
_MIN_USABLE_DRAFT_CHARS = 150


def _strip_echoed_instruction(text: str, target_language: str) -> str:
    """Strip a literal echo of the translation instruction from the model's reply.

    Observed live: when the source text was already (close to) the target
    language, the model sometimes echoed the instruction itself
    ("Translate the following text to English:\\n\\n...") instead of just
    returning the text, rather than recognising there was nothing to do.
    """
    stripped = text.lstrip()
    lower = stripped.lower()
    for prefix in (
        f"translate the following text to {target_language.lower()}:",
        f"translate this text to {target_language.lower()}:",
    ):
        if lower.startswith(prefix):
            return stripped[len(prefix):].lstrip("\n: ").lstrip()
    return text


class ChatEngine:
    def __init__(self, store, model, embedder=None):
        self.store = store
        self.model = model
        self.embedder = embedder

    def ask(
        self,
        question: str,
        history: Optional[List[dict]] = None,
        response_language: Optional[str] = None,
    ) -> Iterator[dict]:
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

        forced_language = response_language.strip() if response_language else None
        if forced_language and forced_language.lower() == "auto":
            forced_language = None

        # "Auto" mode still needs *some* target language when the question
        # isn't English: the grounding pass now always answers in English
        # (see build_messages/SYSTEM_PROMPT), so without this a Turkish
        # question in Auto mode would get an English answer with no
        # translation pass at all. Route it through the same two-pass flow
        # used for an explicitly forced language instead of guessing inside
        # a single overloaded pass.
        target_language = forced_language
        if target_language is None and _looks_turkish(q):
            target_language = "Turkish"

        # Pass 1: ground the answer in the retrieved context. Deliberately
        # never asks for a specific language here — grounding and translating
        # in one pass was found (live testing) to collapse into repetition
        # far more often than grounding alone. See build_messages' docstring.
        messages = build_messages(q, sources, history)
        drafted = ""
        try:
            for delta in self.model.stream_chat(messages):
                drafted += delta
        except Exception as err:  # noqa: BLE001 - surface any generation failure without crashing
            yield {"type": "error", "message": str(err)}
            return
        draft_truncated = getattr(self.model, "last_response_truncated", False)
        drafted_clean = drafted.strip()
        draft_unusable = draft_truncated and len(drafted_clean) < _MIN_USABLE_DRAFT_CHARS

        if not target_language or draft_unusable or not drafted_clean:
            # Nothing to translate, or nothing usable was produced — show the
            # grounded draft as-is.
            if drafted:
                yield {"type": "token", "text": drafted}
            if draft_truncated:
                yield {"type": "notice", "message": _REPETITION_NOTICE}
            yield {"type": "done"}
            return

        # Pass 2: translate the already-correct, already-grounded answer
        # (trailing noise stripped, if the draft was truncated but usable).
        # Collected fully (not streamed live) so a failed translation can be
        # discarded in favour of the original — live testing showed this
        # model's translation pass is the more fragile of the two calls, and
        # a partially-streamed garbled translation can't be taken back once
        # the user has seen it.
        #
        # A line-by-line variant of this pass was tried (translating each
        # line of the draft as a separate call) to address code-mixed,
        # disfluent translations (e.g. "Do confined spacede oxygen levels
        # top, middle... tested olabilir"). It did not help — on a live
        # 10-question re-test it left one case unchanged and made another
        # measurably worse (a hallucinated non-word, "Doğrusuz"). That
        # confirms the disfluency is a genuine translation-capability limit
        # of qwen2.5-7b for English->Turkish, not something the calling
        # pattern can paper over — see TEST_REPORT.md section 5. Reverted to
        # the simpler whole-block call below.
        translation_messages = build_translation_messages(drafted_clean, target_language)
        translated = ""
        try:
            for delta in self.model.stream_chat(translation_messages):
                translated += delta
        except Exception as err:  # noqa: BLE001
            yield {"type": "error", "message": str(err)}
            return

        translation_truncated = getattr(self.model, "last_response_truncated", False)
        translated = _strip_echoed_instruction(translated, target_language)
        translation_clean = translated.strip()
        translation_unusable = (
            not translation_clean
            or (translation_truncated and len(translation_clean) < _MIN_USABLE_DRAFT_CHARS)
        )

        if translation_unusable:
            # Translation was not reliable — fall back to the original,
            # already-correct answer rather than showing garbled text.
            yield {"type": "token", "text": drafted_clean}
            yield {
                "type": "notice",
                "message": (
                    f"Could not reliably translate the answer to {target_language}; "
                    f"showing it in its original language instead."
                ),
            }
            yield {"type": "done"}
            return

        yield {"type": "token", "text": translated}
        if translation_truncated:
            yield {"type": "notice", "message": _REPETITION_NOTICE}
        yield {"type": "done"}

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
