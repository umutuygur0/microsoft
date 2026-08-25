"""Okapi BM25 scoring: replaces plain TF-IDF + cosine similarity as the
sparse (keyword) half of hybrid retrieval.

Why BM25 over TF-IDF: TF-IDF's term-frequency contribution scales linearly
(a term appearing 20 times counts 20x as much as once), and it has no notion
of document length (a keyword mentioned once in a 20-word chunk and once in
a 200-word chunk score identically). BM25 fixes both — term-frequency
contribution saturates (diminishing returns past a few occurrences) and
scores are normalized against how long the chunk is relative to the corpus
average. Both are well-established improvements for ad-hoc passage
retrieval and require no extra dependency (pure Python, same tokenizer and
stored term-frequency maps already used everywhere else in this project).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable

# Standard Okapi BM25 free parameters. k1 controls term-frequency
# saturation (higher = less saturation); b controls document-length
# normalization strength (0 = none, 1 = full). These are the commonly used
# defaults across BM25 implementations (e.g. Elasticsearch, Lucene) and were
# not tuned further for this corpus.
K1 = 1.5
B = 0.75


def compute_bm25_idf(tf_maps: Iterable[Counter]) -> Dict[str, float]:
    """Okapi BM25's own IDF variant: ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5)).

    The "+1 inside the log" variant (rather than the classic Robertson-
    Sparck-Jones formula) is used deliberately: it stays non-negative even
    for a term that appears in more than half the corpus, which the classic
    formula does not guarantee.
    """
    tf_maps = list(tf_maps)
    doc_freq: Counter = Counter()
    for tf in tf_maps:
        doc_freq.update(tf.keys())
    n_docs = len(tf_maps) or 1
    return {
        term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }


def bm25_score(
    query_terms: Iterable[str],
    doc_tf: Counter,
    doc_len: int,
    avg_doc_len: float,
    idf: Dict[str, float],
    k1: float = K1,
    b: float = B,
) -> float:
    """Raw (unnormalized) BM25 score of one document against query terms.

    Unlike cosine similarity, this is unbounded (can exceed 1.0) — callers
    that blend it with a [0, 1]-ranged signal (e.g. semantic cosine
    similarity) must normalize across the candidate set first, see
    ``src/vector_store.py``.
    """
    if doc_len <= 0 or avg_doc_len <= 0:
        return 0.0
    length_norm = 1 - b + b * (doc_len / avg_doc_len)
    score = 0.0
    for term in query_terms:
        freq = doc_tf.get(term, 0)
        if freq == 0:
            continue
        score += idf.get(term, 0.0) * (freq * (k1 + 1)) / (freq + k1 * length_norm)
    return score
