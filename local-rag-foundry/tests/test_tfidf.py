from src.tfidf import term_frequency, tokenize


def test_tokenize_strips_punctuation_and_stopwords():
    tf = term_frequency("The Gas Leak! Detection procedure, and the response.")
    assert "the" not in tf
    assert "and" not in tf
    assert tf["gas"] == 1
    assert tf["leak"] == 1


def test_tokenize_preserves_turkish_characters():
    # Regression test: the tokenizer used to strip ç/ğ/ı/ö/ş/ü as "non-word"
    # characters, corrupting words instead of just lowercasing them.
    tokens = tokenize("Bir boru hattında gaz kaçağını nasıl tespit ederim?")
    assert "kaçağını" in tokens
    assert "hattında" in tokens
    assert "tespit" in tokens


def test_tokenize_filters_turkish_stopwords():
    tf = term_frequency("Bu gaz kaçağı için ne yapmalıyım?")
    assert "bu" not in tf
    assert "için" not in tf
    assert "ne" not in tf
    assert "gaz" in tf
    assert "kaçağı" in tf


def test_tokenize_handles_mixed_language_query():
    # Code-switched Turkish + English query must tokenize both halves intact.
    tokens = tokenize("Gas leak durumunda pipeline yakınında ne yapmalıyım?")
    assert "gas" in tokens
    assert "leak" in tokens
    assert "pipeline" in tokens
    assert "yakınında" in tokens


