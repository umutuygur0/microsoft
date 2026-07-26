import pytest

from src.tfidf import compute_idf, cosine_similarity, term_frequency, tfidf_vector


def test_tokenize_strips_punctuation_and_stopwords():
    tf = term_frequency("The Gas Leak! Detection procedure, and the response.")
    assert "the" not in tf
    assert "and" not in tf
    assert tf["gas"] == 1
    assert tf["leak"] == 1


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
