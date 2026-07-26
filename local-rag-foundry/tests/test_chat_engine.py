from src.chat_engine import ChatEngine
from src.chunker import Chunk
from src.vector_store import VectorStore


def sample_chunks():
    return [
        Chunk("leak", "Gas Leak Detection", "Safety", 0,
              "Use a calibrated combustible gas detector to find a leak near flanges and joints."),
        Chunk("valve", "Valve Maintenance", "Maintenance", 0,
              "Inject valve grease through the sealant fitting and cycle the valve open and close."),
    ]


class _FakeReadyModel:
    ready = True
    message = "ready"

    def stream_chat(self, messages):
        yield "Detect a gas leak "
        yield "with a calibrated detector."


class _FakeUnavailableModel:
    ready = False
    message = "not installed"

    def stream_chat(self, messages):  # pragma: no cover - must never be called
        raise AssertionError("stream_chat should not be called when the model is not ready")


def make_store():
    store = VectorStore(":memory:")
    store.add_chunks(sample_chunks())
    return store


def test_ask_empty_question_yields_error():
    engine = ChatEngine(make_store(), _FakeReadyModel())
    events = list(engine.ask("   "))
    assert events == [{"type": "error", "message": "Empty question."}]


def test_ask_streams_sources_then_tokens_then_done():
    engine = ChatEngine(make_store(), _FakeReadyModel())
    events = list(engine.ask("How do I detect a gas leak?"))
    types = [e["type"] for e in events]
    assert types[0] == "sources"
    assert types[-1] == "done"
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "calibrated detector" in tokens


def test_ask_falls_back_to_passage_when_model_not_ready():
    engine = ChatEngine(make_store(), _FakeUnavailableModel())
    events = list(engine.ask("How do I detect a gas leak?"))
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "not ready" in tokens
    assert "Gas Leak Detection" in tokens


def test_ask_no_matching_context_returns_fallback_message():
    engine = ChatEngine(make_store(), _FakeReadyModel())
    events = list(engine.ask("photosynthesis chlorophyll biology"))
    sources_event = next(e for e in events if e["type"] == "sources")
    assert sources_event["sources"] == []
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "not available in the local knowledge base" in tokens
