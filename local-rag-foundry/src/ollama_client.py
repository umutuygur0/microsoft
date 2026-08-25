"""Ollama chat client: a fully local, alternative LLM backend to Foundry
Local (see src/foundry_client.py) selected via ``config.LLM_PROVIDER``.

Same public interface as ``FoundryClient`` (``ready``, ``message``, ``init()``,
``stream_chat()``, ``last_response_truncated``) so ``src/chat_engine.py``
works unchanged regardless of which provider is active — it only ever calls
through that shared interface, never imports a provider module directly.

Reuses ``src/foundry_client.py``'s repetition-collapse guard (buffer/retry
logic, threshold constants) rather than duplicating it — the failure mode
(a small/quantized local model getting stuck repeating itself) is a property
of small local models in general, not of Foundry Local specifically.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

import config
from src.foundry_client import (
    _EARLY_FAILURE_CHAR_THRESHOLD,
    _MAX_RETRY_TEMPERATURE,
    _RETRY_TEMPERATURE_BOOST,
    _is_runaway_repetition,
)


class OllamaClient:
    def __init__(self):
        self.state = "idle"
        self.message = "Not started"
        self.model_id: Optional[str] = None
        self.last_response_truncated = False

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def init(self) -> None:
        """Verify the Ollama service is reachable and the configured model
        is present locally, pulling it if not. Safe to call more than once."""
        if self.ready:
            return
        try:
            import ollama
        except ImportError:
            self.state = "unavailable"
            self.message = "ollama Python package is not installed."
            return

        try:
            self.state = "starting"
            client = ollama.Client()
            local_models = {m.model for m in client.list().models}
            if config.OLLAMA_MODEL not in local_models:
                client.pull(config.OLLAMA_MODEL)
            self.model_id = config.OLLAMA_MODEL
            self.state = "ready"
            self.message = f"Offline Ready (Ollama) · {self.model_id}"
        except Exception as err:  # noqa: BLE001 - never let a startup failure crash the app
            self.state = "unavailable"
            self.message = self._friendly_error(err)

    def stream_chat(self, messages: List[dict], _allow_retry: bool = True,
                     _temperature: Optional[float] = None) -> Iterator[str]:
        """Yield response text incrementally. Mirrors FoundryClient.stream_chat's
        buffering/early-retry/repetition-guard behaviour exactly (see that
        module's docstring) — only the underlying transport differs."""
        if not self.ready:
            raise RuntimeError(self.message)

        import ollama

        self.last_response_truncated = False
        temperature = _temperature if _temperature is not None else config.TEMPERATURE
        options = {
            "temperature": temperature,
            "num_predict": config.MAX_TOKENS,
            # Ollama has one combined repeat_penalty rather than separate
            # frequency/presence penalties; anchored at 1.0 (no penalty).
            "repeat_penalty": 1.0 + config.FREQUENCY_PENALTY,
        }

        buffer = ""
        pending = ""
        committed = False

        stream = ollama.chat(
            model=self.model_id,
            messages=messages,
            stream=True,
            think=False,  # native Ollama toggle -- see TEST_REPORT.md for why
            options=options,
            keep_alive=config.OLLAMA_KEEP_ALIVE,
        )
        for chunk in stream:
            text = chunk["message"]["content"]
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
                    hotter = min(_MAX_RETRY_TEMPERATURE, temperature + _RETRY_TEMPERATURE_BOOST)
                    yield from self.stream_chat(messages, _allow_retry=False, _temperature=hotter)
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

    @staticmethod
    def _friendly_error(err: Exception) -> str:
        msg = str(err)
        low = msg.lower()
        if any(key in low for key in ("connect", "refused", "endpoint")):
            return "Ollama service is not reachable. Start it (`ollama serve`), then retry."
        return f"Could not start the Ollama model: {msg}"
