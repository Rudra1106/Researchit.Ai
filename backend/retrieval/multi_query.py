"""
multi_query.py

Responsibility: improve retrieval by searching multiple phrasings
of the user's question.

The problem: embedding search is sensitive to wording.
  "how does position work" vs "positional encoding" are semantically
  close but their vectors aren't identical. One phrasing might miss
  the best chunk.

The fix: ask the LLM to rephrase the question 3 ways, run all 4
searches (original + 3 rephrasings), and merge the results.

Deduplication strategy:
  If the same chunk appears in multiple search results, keep the
  version with the highest rrf_score. Don't count it multiple times.
"""

import json
import re

from backend.llm_client              import llm_complete
from backend.retrieval.hybrid_search import hybrid_search

# How many rephrasings to generate.
NUM_REPHRASINGS = 3


def _generate_rephrasings(question):
    """
    Ask the LLM to rephrase the question NUM_REPHRASINGS ways.

    We use a strict JSON prompt so the output is easy to parse.
    Returns a list of rephrasing strings.
    Falls back to [] on any error — the caller always has the original question.
    """
    prompt = f"""You are helping improve search over a research paper.

Rephrase the following question {NUM_REPHRASINGS} different ways to increase the
chance of finding the relevant section. Use different vocabulary and sentence structure.
Focus on the core technical concept being asked about.

Original question: {question}

Respond with ONLY a JSON array of strings, no explanation, no markdown, no backticks.
Example format: ["rephrasing one", "rephrasing two", "rephrasing three"]"""

    try:
        messages = [{"role": "user", "content": prompt}]
        # Short timeout — rephrasings are best-effort; if slow, we skip them
        raw = llm_complete(messages, temperature=0.7, timeout=25)
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        rephrasings = json.loads(raw)
        if not isinstance(rephrasings, list):
            return []
        return [r for r in rephrasings if isinstance(r, str) and r.strip()]
    except Exception:
        return []


def _merge_results(all_results, n_results):
    """
    Merge multiple lists of search results, deduplicating by chunk_id.

    When the same chunk appears in multiple searches, keep the one with the
    highest rrf_score (it ranked well in at least one search).

    Returns the top n_results chunks sorted by rrf_score.
    """
    best_by_id = {}

    for results in all_results:
        for chunk in results:
            cid = str(chunk["chunk_id"])
            existing = best_by_id.get(cid)
            if existing is None or chunk.get("rrf_score", 0) > existing.get("rrf_score", 0):
                best_by_id[cid] = chunk

    sorted_chunks = sorted(
        best_by_id.values(),
        key=lambda c: c.get("rrf_score", 0),
        reverse=True,
    )
    return sorted_chunks[:n_results]


def multi_query_search(question, vector_store, bm25_store, n_results=5):
    """
    Run hybrid search on the original question plus rephrasings, merge results.

    Args:
        question:     the original user question string
        vector_store: VectorStore instance
        bm25_store:   BM25Store instance
        n_results:    how many chunks to return after merging

    Returns:
        list of chunk dicts, deduplicated and sorted by rrf_score
    """
    # Always search the original question first
    original_results = hybrid_search(
        question, vector_store, bm25_store, n_results=n_results
    )

    # Generate rephrasings — this calls the LLM
    rephrasings = _generate_rephrasings(question)

    # Search each rephrasing
    rephrasing_results = []
    for rephrased in rephrasings:
        results = hybrid_search(
            rephrased, vector_store, bm25_store, n_results=n_results
        )
        rephrasing_results.append(results)

    # Merge: original + all rephrasings
    all_results = [original_results] + rephrasing_results
    return _merge_results(all_results, n_results)