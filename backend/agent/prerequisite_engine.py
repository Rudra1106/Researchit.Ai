"""
prerequisite_engine.py

Detects which prerequisite concepts the student is missing before they can
understand the concept they asked about.

Uses:
  1. dl_prerequisites.json — curated static DL dependency map
  2. KnowledgeGraph REQUIRES edges — paper-specific concept dependencies

Returns an ordered queue (most fundamental first) of missing prerequisites.
Capped at MAX_DEPTH=2 levels to avoid turning responses into lectures.
"""

import json
import os
from collections import deque

_DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
_PREREQ_PATH = os.path.join(_DATA_DIR, "dl_prerequisites.json")

MAX_DEPTH = 2   # How many prerequisite levels to recurse


def _load():
    try:
        with open(_PREREQ_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

DL_PREREQUISITES = _load()


def _norm(s: str) -> str:
    return s.lower().strip()


def _find_prereqs_in_map(concept: str) -> list:
    """Look up direct prerequisites from the static map (fuzzy key match)."""
    norm = _norm(concept)
    # Exact match first
    if norm in DL_PREREQUISITES:
        return DL_PREREQUISITES[norm]
    # Partial match — concept string contains a key
    for key, prereqs in DL_PREREQUISITES.items():
        if key in norm or norm in key:
            return prereqs
    return []


def get_prerequisite_queue(
    target_concept: str,
    learner_profile: dict,
    knowledge_graph=None,
) -> list:
    """
    Return an ordered list of concepts to teach BEFORE target_concept.

    Order: most fundamental → most advanced (so we teach dot product before
    attention before multi-head attention).

    Args:
        target_concept:  concept the student asked about
        learner_profile: has known_concepts + taught_this_session
        knowledge_graph: optional, for paper-specific REQUIRES edges

    Returns:
        list of concept strings the student is missing, most-fundamental-first
    """
    known = set(
        _norm(c) for c in
        learner_profile.get("known_concepts", []) +
        learner_profile.get("taught_this_session", [])
    )
    target = _norm(target_concept)

    # BFS
    queue   = deque([(target, 0)])
    visited = {target}
    found   = []   # (concept, depth)

    while queue:
        concept, depth = queue.popleft()
        if depth >= MAX_DEPTH:
            continue

        direct = _find_prereqs_in_map(concept)

        # Supplement from knowledge graph
        if knowledge_graph is not None:
            try:
                graph_prereqs = knowledge_graph.get_prerequisites(concept)
                seen = set(direct)
                for gp in graph_prereqs:
                    if gp not in seen:
                        direct.append(gp)
                        seen.add(gp)
            except Exception:
                pass

        for prereq in direct:
            norm = _norm(prereq)
            if norm not in visited and norm not in known:
                visited.add(norm)
                found.append((norm, depth + 1))
                queue.append((norm, depth + 1))

    if not found:
        return []

    # Sort deepest first (most fundamental → teach first)
    found.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in found]
