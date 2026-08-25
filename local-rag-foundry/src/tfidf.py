"""Pure-Python tokenizing/term-frequency utilities, plus dense cosine
similarity for embeddings.

No third-party dependency for tokenizing. Sparse-vector scoring (originally
TF-IDF + cosine similarity here) has moved to src/bm25.py's Okapi BM25 --
see that module's docstring for why. This module keeps only what BM25 and
the embedding half of hybrid retrieval still need: tokenizing, raw term
frequency, and dense (embedding) cosine similarity.

The tokenizer is Unicode-aware (not English/ASCII-only): it keeps letters from
any script — e.g. Turkish "ç ğ ı ö ş ü" — instead of stripping them, and the
combined English+Turkish stopword list means a query can freely mix languages
(the embedding half of hybrid retrieval, see src/embedder.py, is what actually
carries cross-lingual matches; this tokenizer just has to not corrupt the text).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "at", "for", "with", "by", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "into", "about", "you", "your", "we", "our", "i", "he", "she",
    "they", "them", "his", "her", "do", "does", "did", "so", "not", "no",
}

TURKISH_STOPWORDS = {
    "acaba", "ama", "ancak", "bazı", "belki", "biri", "birşey", "biz", "bu",
    "bunu", "bunlar", "çok", "çünkü", "da", "daha", "de", "defa", "değil",
    "diye", "eğer", "en", "gibi", "hangi", "hem", "hep", "hepsi", "her",
    "hiç", "için", "ile", "ise", "kaç", "kez", "ki", "kim", "mı", "mi", "mu",
    "mü", "mıdır", "midir", "nasıl", "ne", "neden", "nedir", "nerde",
    "nerede", "nereye", "niçin", "niye", "o", "onun", "sanki", "şey", "siz",
    "şu", "tüm", "var", "ve", "veya", "ya", "yani", "yok",
}

# One combined set rather than per-language switching: a query or document may
# freely mix languages (code-switching), so there is no single "current
# language" to pick a stopword list for.
STOPWORDS = ENGLISH_STOPWORDS | TURKISH_STOPWORDS

# Unicode-aware: \w keeps letters from any script (é, ç, ğ, ı, ö, ş, ü, ...)
# instead of the old ASCII-only [a-z0-9] pattern, which silently deleted
# non-English letters and corrupted words like "kaçağı" -> "ka a".
_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens with punctuation and stopwords stripped."""
    if not text:
        return []
    cleaned = _NON_WORD_RE.sub(" ", text.lower())
    return [tok for tok in cleaned.split() if len(tok) > 1 and tok not in STOPWORDS]


def term_frequency(text: str) -> Counter:
    """Raw term-frequency counter for a piece of text."""
    return Counter(tokenize(text))


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
