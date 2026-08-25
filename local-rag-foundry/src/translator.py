"""Deterministic (non-LLM) English -> Turkish translation for the response pass.

Grounding is always done in English (see src/prompts.py's SYSTEM_PROMPT), and
the UI only ever offers Turkish as the other response language (see
app/streamlit_app.py's RESPONSE_LANGUAGES) -- so the actual translation need
is a single, fixed direction. Repeated live testing (TEST_REPORT.md §13)
found asking the *same* chat LLM to translate its own already-correct answer
introduces new problems that were never present in the original: garbled or
self-contradictory Turkish, and at least one case of outright factual drift
(a translated number silently changed). A dedicated, purpose-trained
translation model does not carry the chat LLM's instruction-following or
repetition failure modes into this step at all, since it was never asked to
"answer" anything -- only to translate.

Uses Meta's NLLB-200 rather than a dedicated bilingual model: a live 98-
question audit (TEST_REPORT.md §17) found a bilingual EN->TR MarianMT model
can, rarely but for real, drift into translating a sentence into an entirely
different language (one answer came back in Portuguese) -- a bilingual
model's output language is never actually *enforced*, only expected. NLLB is
a many-to-many model built around ``forced_bos_token_id``: generation is
seeded to start with the target language's own token, which structurally
rules out the model drifting into a different language's vocabulary instead
of merely hoping it won't.

Falls back gracefully (``ready`` stays False) if transformers/torch or the
model download is unavailable, exactly like src/reranker.py -- callers
should then use the existing LLM-based translation pass instead.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import config

# The chat LLM's SYSTEM_PROMPT response format is a small, fixed set of
# headings (see src/prompts.py). Translating them via a lookup table rather
# than sending them through the MT model avoids relying on the model to
# translate a bare one-or-two-word heading correctly/consistently.
_HEADING_MAP = {
    "summary": "Özet",
    "safety warnings": "Güvenlik Uyarıları",
    "step-by-step guidance": "Adım Adım Kılavuz",
}
_HEADING_RE = re.compile(r"^(\**)\s*([A-Za-z][A-Za-z \-]*?)\s*(\**):?\s*$")
_LIST_PREFIX_RE = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)")
# The model truncates around 512 sub-word tokens; a plain character cap
# keeps a single over-long line from silently losing its tail instead of
# erroring, without needing to count tokens up front.
_MAX_INPUT_CHARS = 480


class LocalTranslator:
    def __init__(self):
        self.state = "idle"
        self.message = "Not started"
        self._tokenizer = None
        self._model = None
        self._target_token_id = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def init(self) -> None:
        """Load the configured NLLB model. Safe to call more than once."""
        if self.ready:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError:
            self.state = "unavailable"
            self.message = "transformers is not installed."
            return

        try:
            self.state = "starting"
            self._tokenizer = AutoTokenizer.from_pretrained(config.TRANSLATION_EN_TR_MODEL)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(config.TRANSLATION_EN_TR_MODEL)
            self._target_token_id = self._tokenizer.convert_tokens_to_ids(config.TRANSLATION_TGT_LANG_CODE)
            self.state = "ready"
            self.message = f"Local EN->TR translation active · {config.TRANSLATION_EN_TR_MODEL}"
        except Exception as err:  # noqa: BLE001 - never let this block the rest of the app
            self.state = "unavailable"
            self.message = f"Translator unavailable ({err}); falling back to LLM translation."

    def translate_to_turkish(self, text: str) -> Optional[str]:
        """Translate English ``text`` to Turkish, line by line.

        Markdown structure is preserved deterministically instead of being
        left to the translation model: fixed response headings are mapped via
        ``_HEADING_MAP``, list/number prefixes are stripped before translation
        and reattached after, and blank lines pass through untouched. Only
        the actual sentence content is ever sent to the model. Returns
        ``None`` (never raises) if not ready or on any translation failure,
        so callers can fall back to the LLM-based pass.
        """
        if not self.ready or not text.strip():
            return None

        lines = text.split("\n")
        out: List[Optional[str]] = []
        prefixes: Dict[int, str] = {}
        bodies: List[str] = []
        body_positions: List[int] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                out.append(line)
                continue
            heading = _HEADING_RE.match(stripped)
            if heading:
                stars_l, name, stars_r = heading.groups()
                mapped = _HEADING_MAP.get(name.strip().lower())
                if mapped:
                    out.append(f"{stars_l}{mapped}{stars_r}")
                    continue
            prefix_m = _LIST_PREFIX_RE.match(line)
            prefix = prefix_m.group(1) if prefix_m else ""
            idx = len(out)
            prefixes[idx] = prefix
            body_positions.append(idx)
            bodies.append(line[len(prefix):][:_MAX_INPUT_CHARS])
            out.append(None)

        if not bodies:
            return "\n".join(out)  # type: ignore[arg-type]

        try:
            translated = self._translate_batch(bodies)
        except Exception:  # noqa: BLE001 - fall back to the LLM pass, never crash the answer
            return None

        for idx, piece in zip(body_positions, translated):
            out[idx] = prefixes[idx] + piece

        return "\n".join(out)  # type: ignore[arg-type]

    def _translate_batch(self, texts: List[str]) -> List[str]:
        import torch

        batch = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512,
            src_lang=config.TRANSLATION_SRC_LANG_CODE,
        )

        # A real, live-reproduced failure (with the previously-used bilingual
        # MarianMT model): a fixed max_length=512 regardless of input length
        # let the greedy decoder run far past the natural end of a short
        # translation, and once it ran out of real content it started
        # repeating itself for the rest of the budget -- e.g. a single
        # sentence translated as the same two Turkish sentences alternating
        # over 20 times. Scaling the cap to the actual input length
        # reproducibly eliminated this in side-by-side testing
        # (TEST_REPORT.md §17); no_repeat_ngram_size/repetition_penalty are
        # kept as a cheap additional safety net, the same "defence in depth"
        # approach already used for the chat LLM's own repetition guard (see
        # src/foundry_client.py's _is_runaway_repetition).
        max_len = min(512, max(32, int(batch["input_ids"].shape[1] * 3)))
        with torch.no_grad():
            generated = self._model.generate(
                **batch,
                forced_bos_token_id=self._target_token_id,
                max_length=max_len,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
            )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)
