"""Foundry Local integration: loads a local chat model and streams completions.

Degrades gracefully: if Foundry Local or its SDK is unavailable, ``ready``
stays False and the rest of the app keeps serving retrieval-only results
instead of crashing.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

import config


class FoundryClient:
    def __init__(self):
        self.state = "idle"
        self.message = "Not started"
        self.model_id: Optional[str] = None
        self._chat_client = None

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

            self._chat_client = model.get_chat_client()
            self._chat_client.settings = ChatClientSettings(
                temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS
            )
            self.model_id = model.id
            self.state = "ready"
            self.message = f"Offline Ready · {self.model_id}"
        except Exception as err:  # noqa: BLE001 - never let a startup failure crash the app
            self.state = "unavailable"
            self.message = self._friendly_error(err)

    def stream_chat(self, messages: List[dict]) -> Iterator[str]:
        """Yield response text incrementally. Raises if the model is not ready."""
        if not self.ready:
            raise RuntimeError(self.message)

        for chunk in self._chat_client.complete_streaming_chat(messages):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
            if text:
                yield text

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
