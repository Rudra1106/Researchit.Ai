"""
graph_retrieval.py

Responsibility: use the knowledge graph to improve retrieval and explanation.

Two main functions:

1. augment_query(query, graph)
   Extracts key terms from the query, finds their graph neighbors,
   and appends related concepts to the query.
   This makes hybrid_search find chunks it would otherwise miss.

2. explain_path(concept_a, concept_b, graph)
   Returns a human-readable explanation of how two concepts connect.
   Used when the tutor answers "how is X related to Y?"
"""

import re

# Common English words that aren't meaningful concepts.
# We skip these when extracting key terms from a query.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "what", "why", "how", "when", "where", "which", "who", "whom",
    "this", "that", "these", "those", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "about", "as", "into", "through",
    "and", "or", "but", "not", "so", "yet", "both", "either",
    "than", "then", "just", "more", "also", "paper", "model",
    "explain", "describe", "show", "tell", "give", "make",
}

# How many related concepts to append to the query
MAX_CONCEPTS_TO_APPEND = 5


def _extract_key_terms(query):
    """
    Pull meaningful terms from a query by removing stopwords
    and short tokens.

    "how does self-attention work in transformers?"
    → ["self-attention", "transformers"]
    """
    # Lowercase and split on whitespace / punctuation
    tokens = re.findall(r"[a-zA-Z0-9_\-]+", query.lower())

    key_terms = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) < 3:
            continue
        key_terms.append(token)

    # Also try bigrams — two-word phrases often match better
    # e.g. "self" + "attention" → "self-attention" might be in the graph
    words = query.lower().split()
    for i in range(len(words) - 1):
        bigram = f"{words[i]}-{words[i+1]}"
        if words[i] not in STOPWORDS and words[i+1] not in STOPWORDS:
            key_terms.append(bigram)
        # Also try space-joined
        key_terms.append(f"{words[i]} {words[i+1]}")

    return list(dict.fromkeys(key_terms))  # deduplicate preserving order


def augment_query(query, graph):
    """
    Expand a search query with related concepts from the knowledge graph.

    Example:
        query = "why is ReLU better than sigmoid?"
        graph has: ReLU → solves → vanishing gradient
                   sigmoid → causes → vanishing gradient

        augmented = "why is ReLU better than sigmoid? vanishing gradient slow training"

    The augmented query then goes into hybrid_search, which finds chunks
    mentioning "vanishing gradient" even though the user didn't say it.

    Args:
        query: the original user question string
        graph: a KnowledgeGraph instance

    Returns:
        augmented query string (original + related concepts appended)
        If graph has nothing relevant, returns the original query unchanged.
    """
    if graph.node_count() == 0:
        return query

    key_terms = _extract_key_terms(query)

    # Find related concepts for each key term
    all_related = []
    seen        = set()

    for term in key_terms:
        related = graph.find_related_concepts(term, depth=2)
        for concept in related:
            norm = concept.lower()
            if norm not in seen and norm not in query.lower():
                all_related.append(concept)
                seen.add(norm)

        if len(all_related) >= MAX_CONCEPTS_TO_APPEND:
            break

    if not all_related:
        return query

    appended = " ".join(all_related[:MAX_CONCEPTS_TO_APPEND])
    return f"{query} {appended}"


def explain_path(concept_a, concept_b, graph):
    """
    Find the conceptual path between two concepts and return a
    human-readable explanation of the connection.

    Example:
        concept_a = "ReLU"
        concept_b = "faster training"

        Path found: ReLU → solves → vanishing gradient
                    vanishing gradient → causes → slow training
                    slow training ↔ faster training (inverse)

        Returns: "ReLU solves vanishing gradient, which causes slow training"

    Args:
        concept_a: first concept string
        concept_b: second concept string
        graph:     KnowledgeGraph instance

    Returns:
        str — human-readable path explanation, or empty string if no path
    """
    path = graph.get_path(concept_a, concept_b)

    if not path:
        return ""

    # Format the path as a natural-language chain
    parts = []
    for i, (subj, rel, obj) in enumerate(path):
        if i == 0:
            parts.append(f"{subj} {rel} {obj}")
        else:
            parts.append(f"which {rel} {obj}")

    return ", ".join(parts)


def get_context_for_concept(concept, graph):
    """
    Return a formatted string describing a concept's graph neighborhood.
    Used to inject graph-derived context into the LLM prompt.

    Example output:
        "Based on the knowledge graph:
         • ReLU solves vanishing gradient (from: paper)
         • ReLU replaces sigmoid (from: paper, wikipedia)
         • vanishing gradient causes slow training (from: wikipedia)"

    Returns empty string if concept not in graph.
    """
    edges = graph.get_edges_for_concept(concept)
    if not edges:
        return ""

    lines = ["Based on the knowledge graph:"]
    for edge in edges[:8]:   # limit to 8 edges to keep context clean
        sources_str = ", ".join(edge["sources"])
        lines.append(
            f"  • {edge['subject']} {edge['relation']} {edge['object']}"
            f" (from: {sources_str})"
        )

    return "\n".join(lines)