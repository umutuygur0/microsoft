import config
from src.chat_engine import ChatEngine, _strip_echoed_instruction
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


class _RecordingModel:
    """Records every call it receives, for asserting a multi-pass conversation."""
    ready = True
    message = "ready"

    def __init__(self, replies=None):
        self.calls: list[list[dict]] = []
        self.last_response_truncated = False
        self._replies = replies  # optional: one canned reply string per call

    def stream_chat(self, messages):
        self.calls.append(messages)
        if self._replies is not None:
            yield self._replies[len(self.calls) - 1]
        else:
            yield "ok"

    @property
    def last_messages(self):
        return self.calls[-1] if self.calls else None


class _TruncatingModel:
    """Simulates a model that collapsed almost immediately (short, unusable draft)."""
    ready = True
    message = "ready"
    last_response_truncated = True

    def stream_chat(self, messages):
        yield "Some partial answer before it looped "
        yield "0000000000"


class _LateTruncationThenTranslateModel:
    """Simulates a grounding pass that produced a full, substantial answer
    before the guard caught some harmless trailing noise — and a translation
    pass that then completes normally.
    """
    ready = True
    message = "ready"

    def __init__(self, long_draft: str, translation: str):
        self._long_draft = long_draft
        self._translation = translation
        self.calls: list[list[dict]] = []
        self.last_response_truncated = False

    def stream_chat(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            self.last_response_truncated = True
            yield self._long_draft
        else:
            self.last_response_truncated = False
            yield self._translation


class _TwoPassModel:
    """Returns the first reply for the grounding pass, the second for translation."""
    ready = True
    message = "ready"

    def __init__(self, draft: str, translation: str, translation_truncated: bool = False):
        self._draft = draft
        self._translation = translation
        self._translation_truncated = translation_truncated
        self.calls: list[list[dict]] = []
        self.last_response_truncated = False

    def stream_chat(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            self.last_response_truncated = False
            yield self._draft
        else:
            self.last_response_truncated = self._translation_truncated
            yield self._translation


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


class _FakeReadyReranker:
    """Deliberately reorders to something hybrid search would never produce
    on its own, so a passing test proves the reranker's order was actually
    used rather than the hybrid order slipping through unchanged."""
    ready = True

    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append((query, [c["doc_id"] for c in candidates], top_k))
        reordered = sorted(candidates, key=lambda c: c["doc_id"])  # arbitrary but deterministic
        for c in reordered:
            c["rerank_score"] = 1.0
        return reordered[:top_k]


def test_ask_uses_reranker_output_order_when_reranker_is_ready():
    reranker = _FakeReadyReranker()
    engine = ChatEngine(make_store(), _FakeReadyModel(), reranker=reranker)
    events = list(engine.ask("gas leak valve ppe"))

    assert len(reranker.calls) == 1
    _, candidate_doc_ids, top_k = reranker.calls[0]
    assert top_k == config.TOP_K
    # hybrid search was asked for the wider reranking candidate pool, not TOP_K
    assert len(candidate_doc_ids) <= config.RERANK_CANDIDATE_K

    sources_event = next(e for e in events if e["type"] == "sources")
    result_doc_ids = [s["doc_id"] for s in sources_event["sources"]]
    assert result_doc_ids == sorted(result_doc_ids)  # matches the reranker's (fake) ordering
    assert all(s["rerank_score"] == 1.0 for s in sources_event["sources"])


def test_ask_ignores_reranker_when_not_ready():
    class _NotReadyReranker:
        ready = False

        def rerank(self, *args, **kwargs):  # pragma: no cover - must never be called
            raise AssertionError("rerank should not be called when not ready")

    engine = ChatEngine(make_store(), _FakeReadyModel(), reranker=_NotReadyReranker())
    events = list(engine.ask("How do I detect a gas leak?"))
    sources_event = next(e for e in events if e["type"] == "sources")
    assert "rerank_score" not in sources_event["sources"][0]


class _ThinkingModel:
    """Simulates a Qwen3-style model that emits a <think> reasoning trace
    before its real answer."""
    ready = True
    message = "ready"
    last_response_truncated = False

    def stream_chat(self, messages):
        yield "<think>Let me work through the retrieved context step by step...</think>"
        yield "Detect a gas leak with a calibrated detector."


def test_ask_strips_a_thinking_block_from_the_grounding_draft():
    engine = ChatEngine(make_store(), _ThinkingModel())
    events = list(engine.ask("How do I detect a gas leak?"))
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == "Detect a gas leak with a calibrated detector.\n\nReference: Gas Leak Detection"
    assert "<think>" not in tokens


def test_ask_falls_back_to_passage_when_model_not_ready():
    engine = ChatEngine(make_store(), _FakeUnavailableModel())
    events = list(engine.ask("How do I detect a gas leak?"))
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "not ready" in tokens
    assert "Gas Leak Detection" in tokens


def test_ask_appends_a_deterministic_reference_footer_from_retrieval_metadata():
    # The model is never asked to cite its own source (see SYSTEM_PROMPT) --
    # repeated live testing found it could not be trusted to do so
    # correctly even with explicit instructions. The footer is built purely
    # from the actual retrieved chunk's title, which the fake model below
    # never even sees.
    engine = ChatEngine(make_store(), _FakeReadyModel())
    events = list(engine.ask("How do I detect a gas leak?"))
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens.endswith("\n\nReference: Gas Leak Detection")


def test_ask_does_not_append_a_reference_footer_to_a_clean_refusal():
    # A model that refuses even though sources *were* retrieved and passed
    # to it (it just judged none of them relevant) must not have a
    # "Reference:" footer attached -- that would claim a source was used
    # when the answer says otherwise.
    from src.prompts import NO_CONTEXT_REPLY

    class _RefusingModel:
        ready = True
        message = "ready"
        last_response_truncated = False

        def stream_chat(self, messages):
            yield NO_CONTEXT_REPLY

    engine = ChatEngine(make_store(), _RefusingModel())
    events = list(engine.ask("How do I detect a gas leak?"))
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == NO_CONTEXT_REPLY
    assert "Reference:" not in tokens


def test_ask_no_matching_context_returns_fallback_message():
    engine = ChatEngine(make_store(), _FakeReadyModel())
    events = list(engine.ask("photosynthesis chlorophyll biology"))
    sources_event = next(e for e in events if e["type"] == "sources")
    assert sources_event["sources"] == []
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "not available in the local knowledge base" in tokens


def test_ask_with_forced_language_runs_a_separate_translation_pass():
    # Regression test: asking a small model to ground an answer in retrieved
    # context *and* translate it in a single pass was found (live testing) to
    # collapse into repetition far more often than grounding alone. So a
    # forced language must be a second, dedicated call — never baked into the
    # first (grounding) prompt.
    model = _RecordingModel(replies=["Detect a gas leak with a calibrated detector.", "Kalibreli bir detektörle..."])
    engine = ChatEngine(make_store(), model)
    events = list(engine.ask("How do I detect a gas leak?", response_language="Turkish"))

    assert len(model.calls) == 2
    grounding_system, grounding_user = model.calls[0][0]["content"], model.calls[0][-1]["content"]
    translation_system, translation_user = model.calls[1][0]["content"], model.calls[1][-1]["content"]

    # The grounding pass must not mention the target language at all.
    assert "Turkish" not in grounding_system
    assert "Turkish" not in grounding_user

    # The translation pass is a standalone request built from the grounded answer.
    assert "translat" in translation_system.lower()
    assert "Turkish" in translation_user
    assert "Detect a gas leak with a calibrated detector." in translation_user

    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == "Kalibreli bir detektörle...\n\nReference: Gas Leak Detection"


def test_ask_without_response_language_only_runs_the_grounding_pass():
    model = _RecordingModel()
    engine = ChatEngine(make_store(), model)
    list(engine.ask("How do I detect a gas leak?"))
    assert len(model.calls) == 1


def test_ask_skips_translation_pass_when_response_language_is_auto():
    model = _RecordingModel()
    engine = ChatEngine(make_store(), model)
    list(engine.ask("How do I detect a gas leak?", response_language="Auto"))
    assert len(model.calls) == 1


def test_ask_skips_translation_when_grounding_draft_is_short_and_truncated():
    # Collapsed almost immediately -> nothing trustworthy to translate; show
    # the (truncated, notice-flagged) draft instead of feeding garbled text
    # into a translation pass.
    engine = ChatEngine(make_store(), _TruncatingModel())
    events = list(engine.ask("How do I detect a gas leak?", response_language="Turkish"))
    notice = next(e for e in events if e["type"] == "notice")
    assert "repeating itself" in notice["message"]
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "0000000000" in tokens  # showed the (short) draft, did not translate it


def test_ask_still_translates_a_long_draft_that_was_truncated_late():
    # Regression test: the repetition guard was observed tripping on a
    # handful of trailing blank lines *after* an otherwise complete, correct
    # answer. That must not throw away a good, substantial draft — it should
    # be cleaned up and translated normally, not shown untranslated.
    long_draft = (
        "Summary: To detect a gas leak, use a calibrated combustible gas "
        "detector near flanges and joints, and check readings carefully "
        "before continuing work. Safety Warnings: Do not use naked flames."
    )
    assert len(long_draft) >= 150
    model = _LateTruncationThenTranslateModel(long_draft, "Özet: Gaz kaçağını tespit etmek için...")
    engine = ChatEngine(make_store(), model)

    events = list(engine.ask("How do I detect a gas leak?", response_language="Turkish"))

    assert len(model.calls) == 2  # translation pass DID run
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == "Özet: Gaz kaçağını tespit etmek için...\n\nReference: Gas Leak Detection"
    assert not any(e["type"] == "notice" for e in events)  # final result was clean


def test_strip_thinking_block_removes_a_closed_think_block():
    from src.chat_engine import _strip_thinking_block

    text = "<think>Let me reason about this step by step...</think>The oxygen limit is 19.5% to 23.5%."
    assert _strip_thinking_block(text) == "The oxygen limit is 19.5% to 23.5%."


def test_strip_thinking_block_removes_an_unclosed_think_block():
    # Regression test: max_tokens cutting the model off mid-"thought" (before
    # the closing tag, and before any real answer) was observed live to
    # produce a completely empty final answer -- there is no way to tell
    # where thinking would have ended, so the whole unclosed block is dropped.
    from src.chat_engine import _strip_thinking_block

    text = "<think>Let me consider the context carefully, first I should look at..."
    assert _strip_thinking_block(text) == ""


def test_strip_thinking_block_leaves_normal_text_unchanged():
    from src.chat_engine import _strip_thinking_block

    text = "Summary: Use PPE at all times."
    assert _strip_thinking_block(text) == text


def test_strip_echoed_instruction_removes_the_literal_prompt():
    # Regression test: a real live failure. When the source text was already
    # close to the target language, the model sometimes echoed the
    # instruction itself instead of just returning the (unchanged) text.
    echoed = "Translate the following text to English:\n\nSummary: Use PPE at all times."
    assert _strip_echoed_instruction(echoed, "English") == "Summary: Use PPE at all times."


def test_strip_echoed_instruction_leaves_normal_output_unchanged():
    text = "Özet: Her zaman KKD kullanın."
    assert _strip_echoed_instruction(text, "Turkish") == text


def test_ask_strips_echoed_instruction_from_a_usable_translation():
    draft = (
        "Summary: To detect a gas leak, use a calibrated combustible gas "
        "detector near flanges and joints, and check readings carefully "
        "before continuing work. Safety Warnings: Do not use naked flames."
    )
    echoed_translation = (
        "Translate the following text to Turkish:\n\n"
        "Özet: Gaz kaçağını tespit etmek için kalibreli bir dedektör kullanın "
        "ve flanşların yakınında dikkatlice kontrol edin, açık alevden kaçının."
    )
    model = _TwoPassModel(draft, echoed_translation)
    engine = ChatEngine(make_store(), model)

    events = list(engine.ask("How do I detect a gas leak?", response_language="Turkish"))

    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "Translate the following text" not in tokens
    assert tokens.startswith("Özet: Gaz kaçağını")


def test_ask_falls_back_to_original_when_translation_pass_collapses():
    draft = (
        "Summary: To detect a gas leak, use a calibrated combustible gas "
        "detector near flanges and joints, and check readings carefully "
        "before continuing work. Safety Warnings: Do not use naked flames."
    )
    model = _TwoPassModel(draft, "0000000000", translation_truncated=True)
    engine = ChatEngine(make_store(), model)

    events = list(engine.ask("How do I detect a gas leak?", response_language="Turkish"))

    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == draft + "\n\nReference: Gas Leak Detection"  # fell back to the original, reliable answer
    notice = next(e for e in events if e["type"] == "notice")
    assert "translate" in notice["message"].lower()
    assert "Turkish" in notice["message"]


def test_ask_emits_a_separate_notice_event_when_response_is_truncated():
    # Regression test: the "stopped early" notice must be its own event, not
    # text mixed into the "token" stream — otherwise it gets saved as part of
    # the assistant's message content and re-fed to the model as conversation
    # history on the next turn, which was observed causing the model to
    # imitate the notice and fabricate fake "stopped early" text unprompted.
    engine = ChatEngine(make_store(), _TruncatingModel())
    events = list(engine.ask("How do I detect a gas leak?"))
    types = [e["type"] for e in events]
    assert "notice" in types

    notice = next(e for e in events if e["type"] == "notice")
    assert "repeating itself" in notice["message"]

    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "repeating itself" not in tokens
    assert "stopped early" not in tokens
