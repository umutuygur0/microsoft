from src.foundry_client import FoundryClient, _is_runaway_repetition


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeSettings:
    def __init__(self, temperature=0.2):
        self.temperature = temperature


class _FakeChatClient:
    """Mimics the pieces of foundry_local_sdk's ChatClient that stream_chat() uses."""

    def __init__(self, pieces, temperature=0.2):
        self._pieces = pieces
        self.calls_consumed = 0
        self.settings = _FakeSettings(temperature)

    def complete_streaming_chat(self, messages):
        for piece in self._pieces:
            self.calls_consumed += 1
            yield _FakeChunk(piece)


class _RetryAwareChatClient:
    """Returns ``bad_pieces`` on the first invocation and ``good_pieces`` on
    the second — simulates a model whose retry-with-higher-temperature call
    behaves differently from its first attempt. Records the temperature used
    on each call so tests can confirm the retry actually ran hotter.
    """

    def __init__(self, bad_pieces, good_pieces, temperature=0.2):
        self._bad_pieces = bad_pieces
        self._good_pieces = good_pieces
        self.invocation_count = 0
        self.temperatures_seen: list[float] = []
        self.settings = _FakeSettings(temperature)

    def complete_streaming_chat(self, messages):
        self.invocation_count += 1
        self.temperatures_seen.append(self.settings.temperature)
        pieces = self._bad_pieces if self.invocation_count == 1 else self._good_pieces
        for piece in pieces:
            yield _FakeChunk(piece)


def _ready_client_with_pieces(pieces) -> tuple[FoundryClient, _FakeChatClient]:
    client = FoundryClient()
    client.state = "ready"
    fake_chat_client = _FakeChatClient(pieces)
    client._chat_client = fake_chat_client  # noqa: SLF001 - test-only wiring, no public constructor for this
    return client, fake_chat_client


def _ready_client(fake_chat_client) -> FoundryClient:
    client = FoundryClient()
    client.state = "ready"
    client._chat_client = fake_chat_client  # noqa: SLF001
    return client


def test_detects_single_character_runaway():
    assert _is_runaway_repetition("0" * 60) is True


def test_detects_two_character_cycle_runaway():
    assert _is_runaway_repetition("ab" * 30) is True


def test_detects_three_character_cycle_runaway():
    assert _is_runaway_repetition("xyz" * 20) is True


def test_detects_repeated_word_runaway():
    # Regression test: a real live failure. "0000..." is a period-1 loop,
    # but small models also loop on a whole *word* (here, comma-and-space
    # included) — a much longer period that the original 1/2/3-only check
    # completely missed, letting the response run all the way to max_tokens
    # instead of being caught by the guard.
    assert _is_runaway_repetition("gidin, " * 10) is True


def test_detects_repeated_english_phrase_runaway():
    assert _is_runaway_repetition("please wait. " * 6) is True


def test_detects_repeated_sentence_runaway():
    # Regression test: a real live failure far worse than the word-level
    # case — the model repeated a whole boilerplate sentence (~65 chars)
    # dozens of times, running to thousands of characters completely
    # undetected because periods were only checked up to 20 characters.
    sentence = "This information is not available in the local knowledge base.\n\n"
    assert _is_runaway_repetition(sentence * 4) is True


def test_detects_repeated_long_paragraph_runaway():
    paragraph = (
        "Summary: Use a calibrated portable gas detector to detect and "
        "confirm a natural gas leak on a pipeline or wellhead installation.\n\n"
    )
    assert len(paragraph) < 220  # still within the sentence-period cap
    assert _is_runaway_repetition(paragraph * 3) is True


def test_normal_prose_is_not_flagged():
    text = (
        "Summary: A flare system is used for the safe burning of relieved "
        "and vented gas, with specific monitoring and operational procedures."
    )
    assert _is_runaway_repetition(text) is False


def test_varied_long_prose_is_not_flagged():
    # Guards against false positives from widening the detector up to
    # 220-character periods: several genuinely different sentences in a row
    # must not be mistaken for one sentence repeating.
    text = (
        "This is a long, detailed, and perfectly normal answer. "
        "It covers several distinct points about the procedure. "
        "Each sentence here says something genuinely different. "
        "Safety equipment must always be worn during the process."
    )
    assert _is_runaway_repetition(text) is False


def test_short_digit_run_is_not_flagged():
    # A handful of repeated digits can legitimately occur (e.g. a large
    # number like "1000000"); only a longer run should be treated as a loop.
    assert _is_runaway_repetition("Reading: 100000 Pa") is False


def test_repetition_only_at_the_start_is_not_flagged():
    # The guard only cares about the *trailing* text — a loop that
    # self-corrected earlier in the text should not trip it.
    text = ("0" * 60) + " This is a perfectly normal sentence that follows."
    assert _is_runaway_repetition(text) is False


def test_stream_chat_gives_up_after_one_retry_if_collapse_recurs():
    # An early collapse (well before _EARLY_FAILURE_CHAR_THRESHOLD) triggers
    # a hotter retry. If the retry *also* collapses immediately, stream_chat
    # must give up after exactly one retry — never loop indefinitely.
    bad_pieces = ["Summary: "] + ["0"] * 20
    fake_chat_client = _RetryAwareChatClient(bad_pieces, bad_pieces)
    client = _ready_client(fake_chat_client)

    collected = "".join(client.stream_chat([]))

    assert fake_chat_client.invocation_count == 2  # first attempt + exactly one retry
    assert client.last_response_truncated is True
    assert collected  # still returns *something* rather than nothing


def test_stream_chat_recovers_via_hotter_retry():
    # This is the actual live bug: a query collapsed into repetition almost
    # immediately, deterministically, on every attempt at the same settings.
    # A retry at a higher temperature should escape the lock-in and return a
    # clean answer, with the failed first attempt never visible to the caller.
    bad_pieces = ["Summary: "] + ["0"] * 20
    good_pieces = ["Summary: ", "Corrosion ", "inspection ", "keeps ", "pipework ", "safe."]
    fake_chat_client = _RetryAwareChatClient(bad_pieces, good_pieces)
    client = _ready_client(fake_chat_client)

    collected = "".join(client.stream_chat([]))

    assert fake_chat_client.invocation_count == 2
    assert client.last_response_truncated is False
    assert collected == "Summary: Corrosion inspection keeps pipework safe."
    assert "0" not in collected  # the garbled first attempt must not leak through

    # Retry actually used a higher temperature, and it was restored afterwards.
    assert fake_chat_client.temperatures_seen[1] > fake_chat_client.temperatures_seen[0]
    assert fake_chat_client.settings.temperature == 0.2


def test_stream_chat_does_not_retry_a_collapse_that_happens_late():
    # A response that got substantially far (past _EARLY_FAILURE_CHAR_THRESHOLD)
    # before degenerating already delivered most of its value — show it
    # truncated rather than discarding it and retrying from scratch. Uses
    # varied (non-repeating) sentences so the "clean" prefix itself does not
    # trip the sentence-level repetition check.
    long_prefix = [
        "This is a long, detailed, and perfectly normal answer. ",
        "It covers several distinct points about the procedure. ",
        "Each sentence here says something genuinely different. ",
        "Safety equipment must always be worn during the process. ",
        "Records should be kept for every inspection performed. ",
        "Finally, report any anomalies to the shift supervisor. ",
    ]  # ~330 chars, no repeated unit
    pieces = long_prefix + ["0"] * 20
    fake_chat_client = _RetryAwareChatClient(pieces, ["should never be used"])
    client = _ready_client(fake_chat_client)

    collected = "".join(client.stream_chat([]))

    assert fake_chat_client.invocation_count == 1  # no retry attempted
    assert client.last_response_truncated is True
    assert collected.startswith("This is a long, detailed")


def test_stream_chat_stops_early_on_repetition_and_sets_truncated_flag():
    # 5 normal words, then a long "0" runaway that would otherwise continue
    # for many more pieces if not for the guard. (Early + short -> this also
    # exercises the retry path, which reproduces the same failure since the
    # fake client always returns the same fixed pieces.)
    pieces = ["Summary ", "of ", "a ", "flare ", "system. "] + ["0"] * 200
    client, fake_chat_client = _ready_client_with_pieces(pieces)

    collected = "".join(client.stream_chat([]))

    assert client.last_response_truncated is True
    assert collected.startswith("Summary of a flare system. ")


def test_stream_chat_stops_early_on_repeated_word_and_sets_truncated_flag():
    # End-to-end version of the live bug: streamed word-by-word, "gidin, "
    # repeating must be caught the same way a character-level loop is.
    pieces = ["Summary: ", "Corrosion ", "inspection, "] + ["gidin, "] * 50
    client, fake_chat_client = _ready_client_with_pieces(pieces)

    collected = "".join(client.stream_chat([]))

    assert client.last_response_truncated is True
    assert collected.startswith("Summary: Corrosion inspection, ")


def test_stream_chat_leaves_flag_false_for_a_clean_response():
    pieces = ["This ", "is ", "a ", "perfectly ", "normal ", "answer."]
    client, _ = _ready_client_with_pieces(pieces)

    collected = "".join(client.stream_chat([]))

    assert client.last_response_truncated is False
    assert collected == "This is a perfectly normal answer."


def test_stream_chat_resets_truncated_flag_on_a_later_clean_call():
    # A model that looped on a previous question must not leave a stale
    # "truncated" flag bleeding into an unrelated later question's result.
    client, _ = _ready_client_with_pieces(["0"] * 200)
    list(client.stream_chat([]))
    assert client.last_response_truncated is True

    client._chat_client = _FakeChatClient(["A ", "clean ", "answer."])  # noqa: SLF001
    collected = "".join(client.stream_chat([]))

    assert client.last_response_truncated is False
    assert collected == "A clean answer."
