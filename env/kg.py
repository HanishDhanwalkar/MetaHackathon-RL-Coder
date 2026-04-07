"""
Lightweight code knowledge graph backed by NetworkX.

Nodes represent code elements (functions, variables, classes);
each carries a *kind* and a free-text *context* blob.

The ``query`` method performs **name-similarity** lookup using
``difflib.SequenceMatcher`` and returns the top-k most similar nodes
along with their metadata — used to populate ``kg_context`` in
``Observation``.
"""

from difflib import SequenceMatcher
from typing import Any, Dict, List

import networkx as nx


class CodeKnowledgeGraph:
    """In-memory knowledge graph over code elements."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_node(self, name: str, kind: str, context: str = "") -> None:
        """Add a code element node.

        Args:
            name:    Identifier name (e.g. ``my_func``).
            kind:    One of ``"function"``, ``"variable"``, ``"class"``.
            context: Free-text description of the element.
        """
        self.graph.add_node(name, kind=kind, context=context)

    def add_edge(self, src: str, dst: str, relation: str = "uses") -> None:
        """Add a directed relationship between two nodes."""
        self.graph.add_edge(src, dst, relation=relation)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the *top_k* nodes most similar to *query_name*.

        Similarity is measured via ``SequenceMatcher.ratio()`` on
        lower-cased identifier names.
        """
        if not self.graph.nodes:
            return []

        scored: List[Dict[str, Any]] = []
        q_lower = query_name.lower()

        for node_name in self.graph.nodes:
            sim = SequenceMatcher(None, q_lower, node_name.lower()).ratio()
            data = self.graph.nodes[node_name]
            scored.append(
                {
                    "name": node_name,
                    "kind": data.get("kind", "unknown"),
                    "context": data.get("context", ""),
                    "similarity": round(sim, 4),
                }
            )

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()
