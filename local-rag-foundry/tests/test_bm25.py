from collections import Counter

from src.bm25 import bm25_score, compute_bm25_idf


def test_compute_bm25_idf_is_higher_for_rarer_terms():
    tf_maps = [
        Counter({"gas": 2, "leak": 1}),
        Counter({"gas": 1, "valve": 3}),
        Counter({"valve": 1, "rare": 5}),
    ]
    idf = compute_bm25_idf(tf_maps)
    # "gas" appears in 2/3 docs, "rare" in 1/3 -> rarer term gets higher idf.
    assert idf["rare"] > idf["gas"]


def test_compute_bm25_idf_never_negative():
    # A term appearing in every document must still yield a non-negative
    # idf (the "+1 inside the log" BM25 variant, unlike the classic
    # Robertson-Sparck-Jones formula which can go negative here).
    tf_maps = [Counter({"common": 1}) for _ in range(10)]
    idf = compute_bm25_idf(tf_maps)
    assert idf["common"] >= 0


def test_bm25_score_zero_for_no_term_overlap():
    idf = {"gas": 1.0, "leak": 1.0}
    doc_tf = Counter({"valve": 3})
    assert bm25_score(["gas", "leak"], doc_tf, doc_len=3, avg_doc_len=3, idf=idf) == 0.0


def test_bm25_score_higher_for_more_term_matches():
    idf = {"gas": 1.0, "leak": 1.0, "valve": 1.0}
    one_match = bm25_score(["gas", "leak"], Counter({"gas": 1, "other": 5}), doc_len=6, avg_doc_len=6, idf=idf)
    two_matches = bm25_score(["gas", "leak"], Counter({"gas": 1, "leak": 1}), doc_len=2, avg_doc_len=6, idf=idf)
    assert two_matches > one_match


def test_bm25_score_saturates_term_frequency():
    # Going from 1 to 2 occurrences should help more than going from 20 to
    # 21 -- BM25's defining difference from linear TF-IDF scaling.
    idf = {"gas": 1.0}
    doc_len = avg_doc_len = 50
    score_1 = bm25_score(["gas"], Counter({"gas": 1}), doc_len, avg_doc_len, idf)
    score_2 = bm25_score(["gas"], Counter({"gas": 2}), doc_len, avg_doc_len, idf)
    score_20 = bm25_score(["gas"], Counter({"gas": 20}), doc_len, avg_doc_len, idf)
    score_21 = bm25_score(["gas"], Counter({"gas": 21}), doc_len, avg_doc_len, idf)
    assert (score_2 - score_1) > (score_21 - score_20)


def test_bm25_score_penalizes_longer_documents_for_equal_term_frequency():
    # Same raw term frequency, but one document is much longer than the
    # corpus average -> BM25's length normalization should score it lower.
    idf = {"gas": 1.0}
    avg_doc_len = 50
    short_doc_score = bm25_score(["gas"], Counter({"gas": 1}), doc_len=20, avg_doc_len=avg_doc_len, idf=idf)
    long_doc_score = bm25_score(["gas"], Counter({"gas": 1}), doc_len=200, avg_doc_len=avg_doc_len, idf=idf)
    assert short_doc_score > long_doc_score


def test_bm25_score_zero_for_empty_document():
    idf = {"gas": 1.0}
    assert bm25_score(["gas"], Counter(), doc_len=0, avg_doc_len=10, idf=idf) == 0.0
