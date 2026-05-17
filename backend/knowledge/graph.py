"""
graph.py

Responsibility: store and query a concept knowledge graph.

The graph is a NetworkX DiGraph where:
  - Nodes are concept strings ("ReLU", "vanishing gradient", "softmax")
  - Edges are directed relationships ("ReLU" --solves--> "vanishing gradient")
  - Both nodes and edges carry metadata (source, page, timestamp)

Design decisions:
  1. Case-insensitive storage — "ReLU" and "relu" are the same node.
     We normalise to lowercase internally, display in original case.

  2. Duplicate edges allowed with different sources.
     If the paper AND Wikipedia both say "ReLU solves vanishing gradient",
     that's stronger evidence — we track both and weight accordingly.

  3. Persistence to JSON — the graph survives server restarts.
     NetworkX doesn't have built-in persistence so we serialise manually.

  4. Fuzzy node matching — "self attention" matches "self-attention".
     Users don't type concept names exactly. We handle the gap.
"""

import json
import os
import re
from datetime import datetime
from difflib import get_close_matches

import networkx as nx


# Path where the graph is saved between server restarts
GRAPH_SAVE_PATH = "./knowledge_graph.json"

# How similar a query term must be to match a node (0-1, 1=exact)
FUZZY_MATCH_CUTOFF = 0.75

# Max nodes to return from a neighborhood traversal
MAX_RELATED_CONCEPTS = 15


class KnowledgeGraph:
    """
    Wraps NetworkX DiGraph with our domain-specific operations.

    Usage:
        graph = KnowledgeGraph()

        # Add triples from pdf_processor or enrichment
        graph.add_triples([
            {"subject": "ReLU", "relation": "solves",
             "object": "vanishing gradient", "source": "paper"},
        ])

        # Find concepts related to a term
        related = graph.find_related_concepts("ReLU", depth=2)
        # → ["vanishing gradient", "sigmoid", "slow training", ...]

        # Augment a search query with related concepts
        augmented = graph.augment_query("why is ReLU better?")
        # → "why is ReLU better? vanishing gradient sigmoid training speed"
    """

    def __init__(self, save_path=GRAPH_SAVE_PATH):
        self._graph     = nx.DiGraph()
        self._save_path = save_path
        self._load_if_exists()

    # ── Normalisation ──────────────────────────────────────────────────────────

    def _normalise(self, concept):
        """
        Normalise a concept string for consistent storage.
        "Self-Attention" → "self-attention"
        "  ReLU  "       → "relu"

        We store in lowercase and strip edges.
        The original casing is stored as a node attribute.
        """
        return concept.lower().strip()

    def _canonical(self, concept):
        """
        Return the stored canonical form of a concept.
        If not in graph, returns the normalised input.
        """
        norm = self._normalise(concept)
        if self._graph.has_node(norm):
            return norm
        return norm

    # ── Write operations ───────────────────────────────────────────────────────

    def add_triples(self, triples):
        """
        Add a list of concept triples to the graph.

        Each triple must have: subject, relation, object, source.
        Optional: page (int), confidence (float 0-1).

        Silently skips invalid triples (missing fields, empty strings).
        """
        added = 0
        for triple in triples:
            # Validate
            if not isinstance(triple, dict):
                continue
            subj = triple.get("subject",  "").strip()
            rel  = triple.get("relation", "").strip()
            obj  = triple.get("object",   "").strip()
            src  = triple.get("source",   "unknown")

            if not subj or not rel or not obj:
                continue
            if len(subj) > 100 or len(obj) > 100:
                continue   # suspiciously long — likely an extraction error

            subj_norm = self._normalise(subj)
            obj_norm  = self._normalise(obj)

            # Add nodes with original-case display name
            if not self._graph.has_node(subj_norm):
                self._graph.add_node(subj_norm, display=subj, sources=set())
            self._graph.nodes[subj_norm]["sources"].add(src)

            if not self._graph.has_node(obj_norm):
                self._graph.add_node(obj_norm, display=obj, sources=set())
            self._graph.nodes[obj_norm]["sources"].add(src)

            # Add edge — NetworkX allows multiple edges in a MultiDiGraph,
            # but we use DiGraph so we overwrite if the same pair already exists.
            # We track source as a list on the edge so we can accumulate.
            if self._graph.has_edge(subj_norm, obj_norm):
                existing = self._graph[subj_norm][obj_norm]
                if src not in existing.get("sources", []):
                    existing["sources"].append(src)
                existing["weight"] = existing.get("weight", 1) + 1
            else:
                self._graph.add_edge(
                    subj_norm,
                    obj_norm,
                    relation=rel,
                    sources=[src],
                    weight=1,
                    page=triple.get("page"),
                    added_at=datetime.utcnow().isoformat(),
                )
            added += 1

        return added   # number of triples successfully added

    def clear(self):
        """Remove all nodes and edges."""
        self._graph.clear()

    # ── Read operations ────────────────────────────────────────────────────────

    def node_count(self):
        return self._graph.number_of_nodes()

    def edge_count(self):
        return self._graph.number_of_edges()

    def has_concept(self, concept):
        """Return True if a concept exists in the graph (case-insensitive)."""
        return self._graph.has_node(self._normalise(concept))

    def find_node(self, concept):
        """
        Find a node by exact or fuzzy match.
        Returns the normalised node key or None.

        Tries in order:
          1. Exact (normalised) match
          2. Fuzzy match against all node names
        """
        norm = self._normalise(concept)

        # Exact match
        if self._graph.has_node(norm):
            return norm

        # Fuzzy match
        all_nodes = list(self._graph.nodes())
        matches = get_close_matches(norm, all_nodes, n=1, cutoff=FUZZY_MATCH_CUTOFF)
        return matches[0] if matches else None

    def find_related_concepts(self, concept, depth=2):
        """
        Find concepts related to a given concept via graph traversal.

        Uses BFS (breadth-first search) up to `depth` hops away.
        Both incoming and outgoing edges are followed — the graph is
        traversed as undirected for discovery purposes.

        Args:
            concept: the starting concept string
            depth:   how many hops to traverse (1=direct, 2=neighbors of neighbors)

        Returns:
            list of concept strings (display names), excluding the input concept
        """
        node = self.find_node(concept)
        if node is None:
            return []

        visited   = {node}
        related   = []
        current   = {node}

        for _ in range(depth):
            next_level = set()
            for n in current:
                # Outgoing edges (n → neighbor)
                for neighbor in self._graph.successors(n):
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)

                # Incoming edges (neighbor → n)
                for neighbor in self._graph.predecessors(n):
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)

            related.extend(next_level)
            current = next_level

            if len(related) >= MAX_RELATED_CONCEPTS:
                break

        # Return display names (original case), limited to max
        result = []
        for node_key in related[:MAX_RELATED_CONCEPTS]:
            node_data = self._graph.nodes[node_key]
            result.append(node_data.get("display", node_key))

        return result

    def get_edges_for_concept(self, concept):
        """
        Return all edges (triples) involving a concept.
        Used to explain relationships to the user.

        Returns list of:
          {"subject", "relation", "object", "sources", "weight"}
        """
        node = self.find_node(concept)
        if node is None:
            return []

        edges = []

        # Outgoing: concept → object
        for _, obj, data in self._graph.out_edges(node, data=True):
            obj_display = self._graph.nodes[obj].get("display", obj)
            node_display = self._graph.nodes[node].get("display", node)
            edges.append({
                "subject":  node_display,
                "relation": data.get("relation", "relates to"),
                "object":   obj_display,
                "sources":  data.get("sources", []),
                "weight":   data.get("weight", 1),
            })

        # Incoming: subject → concept
        for subj, _, data in self._graph.in_edges(node, data=True):
            subj_display = self._graph.nodes[subj].get("display", subj)
            node_display = self._graph.nodes[node].get("display", node)
            edges.append({
                "subject":  subj_display,
                "relation": data.get("relation", "relates to"),
                "object":   node_display,
                "sources":  data.get("sources", []),
                "weight":   data.get("weight", 1),
            })

        return edges

    def get_path(self, concept_a, concept_b):
        """
        Find the shortest path between two concepts.

        Used when the user asks "how is X related to Y?" —
        the path through the graph IS the explanation.

        Returns:
            list of (subject, relation, object) tuples representing each hop,
            or [] if no path exists.
        """
        node_a = self.find_node(concept_a)
        node_b = self.find_node(concept_b)

        if node_a is None or node_b is None:
            return []

        try:
            # Find shortest path in the undirected version
            # (so we can traverse in either direction)
            undirected = self._graph.to_undirected()
            path_nodes = nx.shortest_path(undirected, node_a, node_b)
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []

        # Convert node path into (subject, relation, object) triples
        hops = []
        for i in range(len(path_nodes) - 1):
            a = path_nodes[i]
            b = path_nodes[i + 1]

            # Check both directions for the edge
            if self._graph.has_edge(a, b):
                data = self._graph[a][b]
                a_display = self._graph.nodes[a].get("display", a)
                b_display = self._graph.nodes[b].get("display", b)
                hops.append((a_display, data.get("relation", "relates to"), b_display))
            elif self._graph.has_edge(b, a):
                data = self._graph[b][a]
                a_display = self._graph.nodes[a].get("display", a)
                b_display = self._graph.nodes[b].get("display", b)
                hops.append((b_display, data.get("relation", "relates to"), a_display))

        return hops

    def get_all_concepts(self):
        """Return all concept display names in the graph."""
        return [
            self._graph.nodes[n].get("display", n)
            for n in self._graph.nodes()
        ]

    def get_stats(self):
        """Summary stats — useful for debugging and the /health endpoint."""
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "concepts": self.get_all_concepts()[:20],  # first 20 only
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path=None):
        """
        Serialise the graph to JSON.

        Sets are not JSON-serialisable, so we convert them to lists.
        """
        path = path or self._save_path

        nodes = []
        for node_id, data in self._graph.nodes(data=True):
            nodes.append({
                "id":      node_id,
                "display": data.get("display", node_id),
                "sources": list(data.get("sources", [])),
            })

        edges = []
        for src, dst, data in self._graph.edges(data=True):
            edges.append({
                "from":     src,
                "to":       dst,
                "relation": data.get("relation", ""),
                "sources":  data.get("sources", []),
                "weight":   data.get("weight", 1),
                "page":     data.get("page"),
                "added_at": data.get("added_at", ""),
            })

        with open(path, "w") as f:
            json.dump({"nodes": nodes, "edges": edges}, f, indent=2)

    def _load_if_exists(self):
        """Load graph from disk on startup if a save file exists."""
        if not os.path.exists(self._save_path):
            return

        try:
            with open(self._save_path) as f:
                data = json.load(f)

            for node in data.get("nodes", []):
                self._graph.add_node(
                    node["id"],
                    display=node.get("display", node["id"]),
                    sources=set(node.get("sources", [])),
                )

            for edge in data.get("edges", []):
                self._graph.add_edge(
                    edge["from"],
                    edge["to"],
                    relation=edge.get("relation", ""),
                    sources=edge.get("sources", []),
                    weight=edge.get("weight", 1),
                    page=edge.get("page"),
                    added_at=edge.get("added_at", ""),
                )
        except Exception:
            # Corrupt save file — start fresh
            self._graph.clear()