"""
enrichment.py

Responsibility: coordinate all external information sources
and produce a unified enriched context for the LLM.

This is the module that main.py calls. It handles:
  1. Running Wikipedia + Wolfram in parallel (using threads)
  2. Formatting all sources into a clean context block
  3. Extracting concept triples for the knowledge graph (Phase 4 interface)
  4. Graceful degradation — if any source fails, the rest continue

Parallelism note:
  Web calls can each take 1-3 seconds. Running them sequentially adds 3-6s
  of latency per question. Running them in parallel with ThreadPoolExecutor
  means we wait only as long as the slowest source.
  For I/O-bound work (network calls), threads work well in Python.
"""

import re
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

from backend.tools.web_search import web_search
from backend.tools.wolfram    import query_wolfram_if_math
from backend.llm_client       import llm_complete

# Maximum seconds to wait for ALL web sources combined.
ENRICHMENT_TIMEOUT = 8


# ── Source formatting ──────────────────────────────────────────────────────────

def _format_paper_chunks(chunks):
    """Format paper chunks into a labeled context block."""
    if not chunks:
        return None

    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Paper — {chunk['section']}, Page {chunk['page']}]"
        parts.append(f"{header}\n{chunk['text']}")

    return {
        "source_type": "paper",
        "label":       "From the research paper",
        "text":        "\n\n".join(parts),
    }


def _format_web_result(result):
    """Format a Wikipedia or DuckDuckGo result into a labeled block."""
    if not result:
        return None

    source_labels = {
        "wikipedia":   "Wikipedia",
        "duckduckgo":  "DuckDuckGo",
    }
    label = source_labels.get(result["source"], result["source"])
    title = result.get("title", "")
    url   = result.get("url", "")

    header = f"[{label}: {title}]"
    if url:
        header += f" ({url})"

    return {
        "source_type": result["source"],
        "label":       f"{label}: {title}",
        "text":        f"{header}\n{result['text']}",
    }


def _format_wolfram_result(result):
    """Format a Wolfram Alpha result."""
    if not result:
        return None

    return {
        "source_type": "wolfram_alpha",
        "label":       "Wolfram Alpha computation",
        "text":        f"[Wolfram Alpha — query: '{result['query']}']\n{result['result']}",
    }


def _build_enriched_context(formatted_sources):
    """
    Combine all formatted sources into the final context string
    that gets injected into the LLM prompt.

    Structure:
        === From the research paper ===
        <paper chunks>

        === Wikipedia: Softmax function ===
        <wikipedia text>

        === Wolfram Alpha computation ===
        <wolfram result>
    """
    sections = []
    for src in formatted_sources:
        if src:
            sections.append(f"=== {src['label']} ===\n{src['text']}")

    return "\n\n".join(sections)


# ── Triple extraction (Phase 4 interface) ──────────────────────────────────────

def extract_triples(text, source_label):
    """
    Extract concept-relationship-concept triples from text using the LLM.

    This defines the interface for Phase 4's knowledge graph.
    Right now it just returns the triples — in Phase 4 we'll store them.

    Returns:
        list of dicts: [{subject, relation, object, source}, ...]
        Returns [] on any failure (triple extraction is best-effort).

    Example output:
        [
          {"subject": "ReLU", "relation": "replaces", "object": "sigmoid",    "source": "wikipedia"},
          {"subject": "ReLU", "relation": "solves",   "object": "vanishing gradient", "source": "wikipedia"},
        ]
    """
    if not text or len(text.strip()) < 50:
        return []

    prompt = f"""Extract concept relationships from this text about machine learning.

Return ONLY a JSON array of triples. Each triple has:
  "subject"  : the first concept (a noun or noun phrase)
  "relation" : the relationship verb (uses, replaces, solves, causes, improves, etc.)
  "object"   : the second concept

Rules:
- Only extract clear, factual relationships
- Keep subject and object short (1-4 words)
- Maximum 8 triples
- If no clear relationships exist, return []

Text:
{text[:800]}

Response (JSON array only, no explanation):"""

    try:
        messages = [{"role": "user", "content": prompt}]
        raw = llm_complete(messages, temperature=0.1)
        raw = raw.strip()

        # Strip markdown fences the model sometimes adds
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

        triples = json.loads(raw)

        if not isinstance(triples, list):
            return []

        # Validate and stamp with source
        result = []
        for t in triples:
            if isinstance(t, dict) and all(k in t for k in ("subject", "relation", "object")):
                result.append({
                    "subject":  str(t["subject"]).strip(),
                    "relation": str(t["relation"]).strip(),
                    "object":   str(t["object"]).strip(),
                    "source":   source_label,
                })
        return result

    except Exception:
        return []


# ── Main enrichment function ───────────────────────────────────────────────────

def enrich(question, paper_chunks, skip_triple_extraction=False):
    """
    Gather all information sources for a question and return enriched context.

    Runs Wikipedia and Wolfram in parallel for speed.
    Falls back gracefully if either source fails.

    Args:
        question:               the user's question string
        paper_chunks:           list of chunk dicts from retrieval (may be empty)
        skip_triple_extraction: if True, skip the LLM call for extracting triples
                                from web results (saves ~20–60s on slow local models).
                                Set True in the chat endpoint; False at upload time.

    Returns:
        {
          "context":       str  — formatted text block for the LLM prompt
          "sources_used":  list — which sources contributed (for logging/UI)
          "triples":       list — concept triples extracted (empty when skipped)
        }
    """
    formatted_sources = []
    sources_used      = []
    all_triples       = []

    # Paper chunks are always first — they are ground truth
    paper_formatted = _format_paper_chunks(paper_chunks)
    if paper_formatted:
        formatted_sources.append(paper_formatted)
        sources_used.append("paper")

    # Run web sources in parallel
    web_result     = None
    wolfram_result = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(web_search,             question): "web",
            executor.submit(query_wolfram_if_math,  question): "wolfram",
        }

        for future in as_completed(futures, timeout=ENRICHMENT_TIMEOUT):
            source_name = futures[future]
            try:
                result = future.result()
                if source_name == "web":
                    web_result = result
                elif source_name == "wolfram":
                    wolfram_result = result
            except Exception:
                pass   # this source failed — continue without it

    # Format web result
    web_formatted = _format_web_result(web_result)
    if web_formatted:
        formatted_sources.append(web_formatted)
        sources_used.append(web_result["source"])

        # Extract triples from web content — skip during chat for speed
        if not skip_triple_extraction:
            triples = extract_triples(web_result["text"], web_result["source"])
            all_triples.extend(triples)

    # Format Wolfram result
    wolfram_formatted = _format_wolfram_result(wolfram_result)
    if wolfram_formatted:
        formatted_sources.append(wolfram_formatted)
        sources_used.append("wolfram_alpha")

    context = _build_enriched_context(formatted_sources)

    return {
        "context":      context,
        "sources_used": sources_used,
        "triples":      all_triples,   # stored in knowledge graph in Phase 4
    }