"""Ollama embedding client: alternative to src/embedder.py's LocalEmbedder,
selected via ``config.LLM_PROVIDER``.

Same public interface (``ready``, ``embed()``) so src/vector_store.py and
src/chat_engine.py work unchanged regardless of provider. Meta does not
publish an official "Llama embedding" model, so this is paired with a
separate, purpose-built multilingual embedding model (``config.OLLAMA_EMBEDDING_MODEL``,
default bge-m3 -- chosen for strong multilingual/Turkish retrieval quality)
rather than anything from the Llama family itself.
"""
from __future__ import annotations

from typing import List, Optional

import config


class OllamaEmbedder:
    def __init__(self):
        self.state = "idle"
        self.message = "Not started"
        self.model_id: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def init(self) -> None:
        """Verify the Ollama service is reachable and the configured
        embedding model is present locally, pulling it if not."""
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
            if config.OLLAMA_EMBEDDING_MODEL not in local_models:
                client.pull(config.OLLAMA_EMBEDDING_MODEL)
            self.model_id = config.OLLAMA_EMBEDDING_MODEL
            self.state = "ready"
            self.message = f"Hybrid retrieval active (Ollama) · {self.model_id}"
        except Exception as err:  # noqa: BLE001 - never let this block the rest of the app
            self.state = "unavailable"
            self.message = f"Semantic retrieval unavailable ({err}); using TF-IDF only."

    def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Return one embedding vector per input text, or None if unavailable."""
        if not self.ready or not texts:
            return None
        try:
            import ollama
            response = ollama.embed(model=self.model_id, input=texts)
            return list(response.embeddings)
        except Exception:  # noqa: BLE001 - a transient failure should not break search
            return None
