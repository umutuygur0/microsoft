"""Optional embedding client for hybrid (TF-IDF + semantic) retrieval.

Graceful degradation: if Foundry Local or the embedding model is unavailable,
``ready`` stays False and callers should fall back to pure TF-IDF ranking —
the vector store already does this automatically when no query embedding is
supplied.
"""
from __future__ import annotations

from typing import List, Optional

import config


class LocalEmbedder:
    def __init__(self):
        self.state = "idle"
        self.message = "Not started"
        self.model_id: Optional[str] = None
        self._client = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def init(self) -> None:
        """Load the configured embedding model. Safe to call more than once."""
        if self.ready:
            return
        try:
            from foundry_local_sdk import FoundryLocalManager
            from foundry_local_sdk.configuration import Configuration
        except ImportError:
            self.state = "unavailable"
            self.message = "foundry-local-sdk is not installed."
            return

        try:
            self.state = "starting"
            if FoundryLocalManager.instance is None:
                FoundryLocalManager.initialize(Configuration(app_name=config.APP_NAME))
            manager = FoundryLocalManager.instance

            model = manager.catalog.get_model(config.EMBEDDING_MODEL_ALIAS)
            if model is None:
                raise RuntimeError(f"Embedding model '{config.EMBEDDING_MODEL_ALIAS}' not found.")
            if not model.is_cached:
                model.download()
            if not model.is_loaded:
                model.load()

            self._client = model.get_embedding_client()
            self.model_id = model.id
            self.state = "ready"
            self.message = f"Hybrid retrieval active · {self.model_id}"
        except Exception as err:  # noqa: BLE001 - never let this block the rest of the app
            self.state = "unavailable"
            self.message = f"Semantic retrieval unavailable ({err}); using TF-IDF only."

    def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Return one embedding vector per input text, or None if unavailable."""
        if not self.ready or not texts:
            return None
        try:
            response = self._client.generate_embeddings(texts)
            return [item.embedding for item in response.data]
        except Exception:  # noqa: BLE001 - a transient failure should not break search
            return None
