"""Foundry Local integration: loads a local chat model and streams completions.

Degrades gracefully: if Foundry Local or its SDK is unavailable, ``ready``
stays False and the rest of the app keeps serving retrieval-only results
instead of crashing.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

import config

# Repetition-collapse guard: small local models occasionally get stuck
# emitting the same short cycle forever. frequency_penalty (set below) makes
# this rare, but this is the last line of defence that actually stops a
# stuck stream. Three failure shapes were observed live and all must be
# caught:
#   - character-level:   "0000000000000..."                          (period 1)
#   - word/phrase-level:  "gidin, gidin, gidin, gidin, ..."           (period ~7)
#   - sentence-level:     the same ~50-100+ character sentence or
#                         boilerplate line ("This information is not
#                         available in the local knowledge base.") repeated
#                         dozens of times, running to thousands of characters
#                         completely undetected by a period cap of only 20.
# Short periods (1-2 chars, e.g. a repeated digit) need more consecutive
# repeats before flagging, since short legitimate runs *can* occur (e.g. a
# number like "1000000"). Longer periods need very few repeats — the same
# 3+ character chunk appearing twice or three times in a row essentially
# never happens in real prose, in any language, regardless of how long that
# chunk is.
_SHORT_PERIOD_MAX = 2
_SHORT_PERIOD_MIN_REPEATS = 8
_WORD_PERIOD_MAX = 20
_WORD_PERIOD_MIN_REPEATS = 3
_SENTENCE_PERIOD_MAX = 220
_SENTENCE_PERIOD_MIN_REPEATS = 2

# If the loop starts this early, the generation essentially never got going —
# showing the user a couple of garbled words plus a notice is a poor outcome,
# and it was observed live to be highly reproducible (a bare retry with the
# same settings reproduces the exact same collapse), not a one-off sampling
# fluke. So instead of surfacing the failure immediately, retry once with a
# hotter temperature (nudging the sampler out of the same lock-in) before
# giving up. A response that gets substantially further before degenerating
# is shown as-is (truncated + notice) rather than retried, since it already
# delivered most of its value.
_EARLY_FAILURE_CHAR_THRESHOLD = 150
_RETRY_TEMPERATURE_BOOST = 0.4
_MAX_RETRY_TEMPERATURE = 1.0


def _min_repeats_for_period(period: int) -> int:
    if period <= _SHORT_PERIOD_MAX:
        return _SHORT_PERIOD_MIN_REPEATS
    if period <= _WORD_PERIOD_MAX:
        return _WORD_PERIOD_MIN_REPEATS
    return _SENTENCE_PERIOD_MIN_REPEATS


def _is_runaway_repetition(text: str) -> bool:
    """True if the text ends with the same short unit repeated back-to-back."""
    for period in range(1, _SENTENCE_PERIOD_MAX + 1):
        min_repeats = _min_repeats_for_period(period)
        segment_len = period * min_repeats
        if len(text) < segment_len:
            continue
        segment = text[-segment_len:]
        unit = segment[:period]
        if unit * min_repeats == segment:
            return True
    return False


class FoundryClient:
    def __init__(self):
        self.state = "idle"
        self.message = "Not started"
        self.model_id: Optional[str] = None
        self._chat_client = None
        # Set by stream_chat() when it cuts a response short. Exposed as a
        # flag rather than inline text so callers can surface it as a UI-only
        # notice *without* it polluting the conversation history that gets
        # fed back to the model on the next turn (the model was observed
        # imitating its own past "stopped early" notices otherwise).
        self.last_response_truncated = False

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def init(self) -> None:
        """Start the Foundry Local service and load the configured chat model.

        Safe to call more than once (e.g. across Streamlit re-runs) — it is a
        no-op once ``ready`` is already True.
        """
        if self.ready:
            return
        try:
            from foundry_local_sdk import FoundryLocalManager
            from foundry_local_sdk.configuration import Configuration
        except ImportError:
            self.state = "unavailable"
            self.message = (
                "foundry-local-sdk is not installed. Run `pip install -r requirements.txt` "
                "to enable answer generation (retrieval still works without it)."
            )
            return

        try:
            self.state = "starting"
            if FoundryLocalManager.instance is None:
                FoundryLocalManager.initialize(Configuration(app_name=config.APP_NAME))
            manager = FoundryLocalManager.instance

            model = manager.catalog.get_model(config.MODEL_ALIAS)
            if model is None:
                raise RuntimeError(f"Model '{config.MODEL_ALIAS}' was not found in the local catalogue.")
            if not model.is_cached:
                model.download()
            if not model.is_loaded:
                model.load()

            from foundry_local_sdk.openai import ChatClientSettings

            is_thinking_model = config.MODEL_ALIAS.lower().startswith("qwen3")
            max_tokens = config.THINKING_MODEL_MAX_TOKENS if is_thinking_model else config.MAX_TOKENS

            self._chat_client = model.get_chat_client()
            self._chat_client.settings = ChatClientSettings(
                temperature=config.TEMPERATURE,
                max_tokens=max_tokens,
                frequency_penalty=config.FREQUENCY_PENALTY,
                presence_penalty=config.PRESENCE_PENALTY,
            )
            self.model_id = model.id
            self.state = "ready"
            self.message = f"Offline Ready · {self.model_id}"
        except Exception as err:  # noqa: BLE001 - never let a startup failure crash the app
            self.state = "unavailable"
            self.message = self._friendly_error(err)

    def stream_chat(self, messages: List[dict], _allow_retry: bool = True) -> Iterator[str]:
        """Yield response text incrementally. Raises if the model is not ready.

        Stops early, with no further text yielded, if the model falls into a
        repetition-collapse loop (see ``_is_runaway_repetition``). Check
        ``last_response_truncated`` afterwards to know whether that happened —
        it is *not* embedded in the yielded text, so it never ends up as part
        of stored conversation history.

        Early text is buffered (not yielded live) until it either clears the
        ``_EARLY_FAILURE_CHAR_THRESHOLD`` or loops — if it loops that early,
        the buffered (garbled) text is discarded entirely and one retry is
        made at a higher temperature, so the caller never sees the failed
        first attempt.
        """
        if not self.ready:
            raise RuntimeError(self.message)

        self.last_response_truncated = False
        buffer = ""
        pending = ""
        committed = False

        for chunk in self._chat_client.complete_streaming_chat(messages):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
            if not text:
                continue

            buffer += text

            if committed:
                yield text
                if _is_runaway_repetition(buffer):
                    self.last_response_truncated = True
                    return
                continue

            pending += text
            if _is_runaway_repetition(buffer):
                if _allow_retry and len(buffer) < _EARLY_FAILURE_CHAR_THRESHOLD:
                    yield from self._retry_hotter(messages)
                    return
                yield pending
                self.last_response_truncated = True
                return

            if len(pending) >= _EARLY_FAILURE_CHAR_THRESHOLD:
                committed = True
                yield pending
                pending = ""

        if not committed and pending:
            yield pending

    def _retry_hotter(self, messages: List[dict]) -> Iterator[str]:
        """Re-run stream_chat once with a boosted temperature, then restore it."""
        original_temperature = self._chat_client.settings.temperature
        self._chat_client.settings.temperature = min(
            _MAX_RETRY_TEMPERATURE, (original_temperature or 0.0) + _RETRY_TEMPERATURE_BOOST
        )
        try:
            yield from self.stream_chat(messages, _allow_retry=False)
        finally:
            self._chat_client.settings.temperature = original_temperature

    @staticmethod
    def _friendly_error(err: Exception) -> str:
        msg = str(err)
        low = msg.lower()
        if any(key in low for key in ("connect", "refused", "endpoint", "not installed", "path", "service")):
            return (
                "Foundry Local service is not reachable. Install/start it with "
                "`winget install Microsoft.FoundryLocal`, then retry."
            )
        return f"Could not start the local model: {msg}"
