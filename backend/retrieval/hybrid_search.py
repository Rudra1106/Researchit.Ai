"""
hybrid_search.py

Responsibility: blend BM25 + vector search results using
Reciprocal Rank Fusion (RRF).

Why RRF works:
  BM25 scores and vector similarity scores live on completely different
  numerical scales. You cannot average them directly.

  RRF converts both to RANKS first (1st place, 2nd place, ...), then
  combines the ranks with a formula that rewards appearing near the top
  in EITHER list.

  RRF formula for chunk c:
    score(c) = 1/(k + rank_vector(c)) + 1/(k + rank_bm25(c))

  where k=60 is a standard smoothing constant. If a chunk didn't appear
  in one of the searches, we treat its rank as infinity (score contribution = 0).

  A chunk that is #1 in vector AND #1 in BM25 gets the highest possible score.
  A chunk that is #1 in only one search still scores well.
  A chunk that doesn't appear in either search scores 0.
"""

from backend.vector_store          import VectorStore
from backend.retrieval.bm25_store  import BM25Store

# Standard RRF smoothing constant.
# Higher k = less reward for top ranks, more smoothing.
# 60 is the value from the original RRF paper and works well in practice.
RRF_K = 60


def _reciprocal_rank_fusion(vector_results, bm25_results, k=RRF_K):
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Args:
        vector_results: list of chunk dicts, ordered best→worst by vector score
        bm25_results:   list of chunk dicts, ordered best→worst by BM25 score
        k:              smoothing constant

    Returns:
        list of chunk dicts sorted by RRF score descending,
        each with a 'rrf_score' field added
    """
    # Map chunk_id → RRF score accumulator
    rrf_scores = {}
    # Map chunk_id → the chunk dict (so we can reconstruct results)
    chunk_by_id = {}

    # Process vector results — rank is 1-indexed
    for rank, chunk in enumerate(vector_results, start=1):
        cid = str(chunk["chunk_id"])
        rrf_scores[cid]  = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_by_id[cid] = chunk

    # Process BM25 results — same formula
    for rank, chunk in enumerate(bm25_results, start=1):
        cid = str(chunk["chunk_id"])
        rrf_scores[cid]  = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_by_id[cid] = chunk

    # Sort by RRF score, best first
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    results = []
    for cid in sorted_ids:
        chunk = dict(chunk_by_id[cid])
        chunk["rrf_score"] = round(rrf_scores[cid], 6)
        results.append(chunk)

    return results


def hybrid_search(query, vector_store, bm25_store, n_results=5, where=None):
    """
    Run BM25 + vector search and blend results with RRF.

    Args:
        query:        the search string
        vector_store: a VectorStore instance (already loaded with chunks)
        bm25_store:   a BM25Store instance (already indexed with same chunks)
        n_results:    how many final results to return
        where:        optional metadata filter passed to vector search
                      e.g. {"section": "Methods"} or {"has_math": True}

    Returns:
        list of chunk dicts, sorted by RRF score, with 'rrf_score' field added
    """
    # Fetch more candidates than we need — RRF will rerank and we trim at the end.
    # Fetching 2× gives RRF enough overlap to work with.
    candidate_count = n_results * 2

    vector_results = vector_store.search(
        query,
        n_results=candidate_count,
        where=where,
    )

    bm25_results = bm25_store.search(
        query,
        n_results=candidate_count,
    )

    if not vector_results and not bm25_results:
        return []

    merged = _reciprocal_rank_fusion(vector_results, bm25_results)
    return merged[:n_results]