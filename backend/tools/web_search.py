"""
web_search.py

Responsibility: fetch relevant external information for a query.

Two sources:
  1. Wikipedia  — structured encyclopedia articles, free, no key
  2. DuckDuckGo — general web search fallback, free, no key

Both are best-effort: if they fail (network error, no result),
we return None and the caller continues without them.
The paper chunks are always the primary source — web results enrich, not replace.

Wikipedia strategy:
  We don't search Wikipedia for the raw question ("what is softmax in attention?").
  We first extract the core concept ("softmax"), then fetch that Wikipedia article.
  This gives us a clean, focused article rather than a search results page.

DuckDuckGo strategy:
  DuckDuckGo has an "Instant Answer" API (ddg.gg/?q=...&format=json)
  that returns a short structured answer for well-known concepts.
  No API key, no rate limit (reasonable use), completely free.
"""

import re
import requests

WIKIPEDIA_API   = "https://en.wikipedia.org/api/rest_v1/page/summary"
DDG_API         = "https://api.duckduckgo.com/"
REQUEST_TIMEOUT = 10   # seconds — web calls must be fast or we skip them


# ── Wikipedia ──────────────────────────────────────────────────────────────────

def _extract_core_concept(question):
    """
    Pull the most likely Wikipedia-searchable concept from a question.

    We strip question words and common filler to get the noun phrase.
    Examples:
      "what is softmax?"               → "softmax"
      "how does self-attention work?"  → "self-attention"
      "explain backpropagation"        → "backpropagation"

    This is a simple heuristic — good enough for technical concepts.
    In Phase 4 the knowledge graph will do this more precisely.
    """
    question = question.lower().strip().rstrip("?")

    # Remove common question prefixes
    prefixes = [
        "what is", "what are", "how does", "how do", "explain",
        "describe", "tell me about", "can you explain", "what does",
        "why is", "why does", "what makes",
    ]
    for prefix in prefixes:
        if question.startswith(prefix):
            question = question[len(prefix):].strip()
            break

    # Take the first meaningful noun phrase (up to 4 words)
    words = question.split()[:4]
    concept = " ".join(words)

    return concept


def search_wikipedia(query, extract_concept=True):
    """
    Fetch a Wikipedia summary for the core concept in the query.

    Args:
        query:           the user's question or a concept string
        extract_concept: if True, extract the core concept from the query first
                         if False, use the query directly as the search term

    Returns:
        dict with keys: source, title, text, url, concept
        or None if nothing found
    """
    concept = _extract_core_concept(query) if extract_concept else query
    if not concept:
        return None

    # Wikipedia page titles use underscores and title case
    search_term = concept.replace(" ", "_")

    try:
        response = requests.get(
            f"{WIKIPEDIA_API}/{search_term}",
            params={"redirect": "true"},   # follow redirects (e.g. "relu" → "ReLU")
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "ResearchTutorBot/1.0 (educational project)"},
        )
    except requests.exceptions.RequestException:
        return None

    if response.status_code != 200:
        return None

    data = response.json()

    # Disambiguation pages aren't useful — they just list other pages
    if data.get("type") == "disambiguation":
        return None

    summary = data.get("extract", "").strip()
    if not summary:
        return None

    # Wikipedia summaries can be very long. Take first 3 paragraphs.
    # This keeps our LLM context window clean.
    paragraphs = [p.strip() for p in summary.split("\n") if p.strip()]
    trimmed    = "\n\n".join(paragraphs[:3])

    return {
        "source":  "wikipedia",
        "concept": concept,
        "title":   data.get("title", concept),
        "text":    trimmed,
        "url":     data.get("content_urls", {}).get("desktop", {}).get("page", ""),
    }


# ── DuckDuckGo Instant Answer ──────────────────────────────────────────────────

def search_duckduckgo(query):
    """
    Query DuckDuckGo's Instant Answer API.

    This returns a short structured answer for well-known topics.
    It's not a full web search — it's more like a knowledge panel.
    Works well for: named algorithms, models, papers, authors.
    Returns None if DDG has no instant answer (which is common).

    Args:
        query: the search string

    Returns:
        dict with keys: source, title, text, url
        or None if no instant answer found
    """
    try:
        response = requests.get(
            DDG_API,
            params={
                "q":       query,
                "format":  "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "ResearchTutorBot/1.0 (educational project)"},
        )
    except requests.exceptions.RequestException:
        return None

    if response.status_code != 200:
        return None

    data = response.json()

    # DDG returns an Abstract for well-known topics
    abstract = data.get("Abstract", "").strip()
    if not abstract:
        return None

    return {
        "source": "duckduckgo",
        "title":  data.get("Heading", query),
        "text":   abstract,
        "url":    data.get("AbstractURL", ""),
    }


# ── Combined web search ────────────────────────────────────────────────────────

def web_search(query):
    """
    Try Wikipedia first, fall back to DuckDuckGo.

    Returns a single best result or None.
    We prefer Wikipedia because it's more structured and reliable.

    Args:
        query: the user's question

    Returns:
        dict with source info, or None
    """
    wiki = search_wikipedia(query)
    if wiki:
        return wiki

    ddg = search_duckduckgo(query)
    return ddg   # may also be None — that's fine