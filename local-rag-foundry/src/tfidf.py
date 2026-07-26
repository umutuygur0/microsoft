"""Pure-Python TF-IDF utilities: tokenizing, term frequency, IDF, cosine similarity.

No embedding model, no third-party dependency. Good enough for a small,
domain-specific corpus and fully transparent (every weight can be inspected).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "at", "for", "with", "by", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "into", "about", "you", "your", "we", "our", "i", "he", "she",
    "they", "them", "his", "her", "do", "does", "did", "so", "not", "no",
}

_NON_WORD_RE = re.compile(r"[^a-z0-9\s]")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens with punctuation and stopwords stripped."""
    if not text:
        return []
    cleaned = _NON_WORD_RE.sub(" ", text.lower())
    return [tok for tok in cleaned.split() if len(tok) > 1 and tok not in STOPWORDS]


def term_frequency(text: str) -> Counter:
    """Raw term-frequency counter for a piece of text."""
    return Counter(tokenize(text))


def compute_idf(tf_maps: Iterable[Counter]) -> Dict[str, float]:
    """idf(t) = ln(1 + N / df(t)) across a corpus of term-frequency counters."""
    tf_maps = list(tf_maps)
    doc_freq: Counter = Counter()
    for tf in tf_maps:
        doc_freq.update(tf.keys())
    n_docs = len(tf_maps) or 1
    return {term: math.log(1 + n_docs / freq) for term, freq in doc_freq.items()}


def tfidf_vector(tf: Counter, idf: Dict[str, float]) -> Dict[str, float]:
    """Weight a term-frequency counter by IDF. Unseen terms default to idf=1."""
    return {term: freq * idf.get(term, 1.0) for term, freq in tf.items()}


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors represented as dicts."""
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(weight * large.get(term, 0.0) for term, weight in small.items())
    if dot == 0:
        return 0.0
    mag_a = math.sqrt(sum(w * w for w in a.values()))
    mag_b = math.sqrt(sum(w * w for w in b.values()))
    return dot / (mag_a * mag_b)


def dense_cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two dense float vectors (e.g. embeddings)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
