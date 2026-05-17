"""
bm25_store.py

Responsibility: keyword-based search over chunk text using BM25.

BM25 is the gold standard keyword ranking algorithm.
It improves on plain word-counting by:
  1. Penalising very common words (like "the") — they tell you little
  2. Rewarding rare words (like "d_k") — they're more distinctive
  3. Normalising for document length — a long chunk shouldn't win
     just because it repeats a word more often than a short chunk

We rebuild the BM25 index every time chunks are loaded.
This is fine — it's fast (milliseconds for a typical paper).

BM25 operates on tokens (words), not embeddings.
So "attention" and "attends" are DIFFERENT tokens to BM25
but SIMILAR vectors to the embedding model.
That's exactly why we need both — they complement each other.
"""

import re
from rank_bm25 import BM25Okapi


def _tokenise(text):
    """
    Convert text to a list of lowercase tokens.

    Simple tokenisation: lowercase, split on non-alphanumeric characters.
    We keep numbers because "d_k", "512", "0.1" are meaningful in papers.

    Example:
        "Attention(Q, K, V) = softmax(QK^T)" 
        → ["attention", "q", "k", "v", "softmax", "qk", "t"]
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


class BM25Store:
    """
    Wraps rank_bm25.BM25Okapi with our chunk format.

    Usage:
        store = BM25Store()
        store.index(chunks)
        results = store.search("attention formula", n_results=5)
    """

    def __init__(self):
        self._bm25       = None   # the BM25 index (built on demand)
        self._chunks     = []     # the original chunks, kept for result lookup
        self._tokenised  = []     # tokenised version of each chunk's text

    def index(self, chunks):
        """
        Build a BM25 index from a list of chunks.

        Call this after loading chunks into the vector store.
        Both stores should always index the same set of chunks.

        Args:
            chunks: list of chunk dicts from pdf_processor
        """
        if not chunks:
            self._bm25      = None
            self._chunks    = []
            self._tokenised = []
            return

        self._chunks    = chunks
        self._tokenised = [_tokenise(chunk["text"]) for chunk in chunks]
        self._bm25      = BM25Okapi(self._tokenised)

    def search(self, query, n_results=10):
        """
        Find chunks whose text matches the query keywords.

        Returns a list of dicts with chunk data + a 'bm25_score' field.
        Results are sorted by score descending (best first).

        Note: we return n_results=10 by default here (more than vector search)
        because BM25 is fast and we want to give RRF enough candidates to blend.
        """
        if self._bm25 is None or not self._chunks:
            return []

        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        # get_scores returns one score per chunk in the same order as self._chunks
        scores = self._bm25.get_scores(query_tokens)

        # Pair each chunk with its score, sort descending
        scored = sorted(
            zip(self._chunks, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        # Filter out zero-score chunks (no keyword overlap at all)
        results = []
        for chunk, score in scored[:n_results]:
            if score > 0:
                result = dict(chunk)          # copy so we don't mutate original
                result["bm25_score"] = score
                results.append(result)

        return results

    def is_ready(self):
        """Return True if the index has been built."""
        return self._bm25 is not None

    def chunk_count(self):
        """How many chunks are indexed."""
        return len(self._chunks)