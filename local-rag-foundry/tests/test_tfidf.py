import pytest

from src.tfidf import compute_idf, cosine_similarity, term_frequency, tfidf_vector, tokenize


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


def test_idf_rare_terms_score_higher():
    docs = [term_frequency("gas leak detection"), term_frequency("gas valve maintenance")]
    idf = compute_idf(docs)
    assert idf["leak"] > idf["gas"]  # "gas" appears in both docs, "leak" only in one


def test_cosine_similarity_identical_vectors_is_one():
    v = tfidf_vector(term_frequency("gas leak detection"), {})
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_disjoint_vectors_is_zero():
    a = tfidf_vector(term_frequency("gas leak"), {})
    b = tfidf_vector(term_frequency("valve maintenance"), {})
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_empty_vector_is_zero():
    assert cosine_similarity({}, {"gas": 1.0}) == 0.0
