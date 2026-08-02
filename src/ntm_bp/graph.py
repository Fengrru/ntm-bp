"""Probability dependency graph for NTM-BP (stdlib-only implementation).

A directed graph whose nodes are fact IDs and whose edges carry
Conditional Probability Tables (CPTs):

    CPT[parent_value][child_value] = P(child_value | parent_value)

Each edge also carries an optional ``relation_type`` tag
(``"supports"`` / ``"contradicts"`` / ``"neutral"`` / ``"default"``)
consumed by the experimental CO engine; the core NTM-BP engine ignores
it. The original research implementation wrapped ``networkx.DiGraph``;
this lightweight port uses plain dict adjacency lists so the package
has zero third-party dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


class DependencyGraph:
    """Directed fact-dependency graph with per-edge CPTs.

    Each edge e = (parent, child) means "child depends on parent":
    a change in the parent's value should decay the child's belief.
    """

    def __init__(self) -> None:
        self._out: Dict[str, List[str]] = {}
        self._in: Dict[str, List[str]] = {}
        self._cpt: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
        self._relation_types: Dict[Tuple[str, str], str] = {}
        self._nodes: Set[str] = set()

    # ------------------------------------------------------------------
    # Node / edge management
    # ------------------------------------------------------------------

    def add_node(self, fact_id: str) -> None:
        """Add a fact node to the graph."""
        if fact_id not in self._nodes:
            self._nodes.add(fact_id)
            self._out.setdefault(fact_id, [])
            self._in.setdefault(fact_id, [])

    def add_edge(
        self,
        parent_id: str,
        child_id: str,
        cpt: Optional[Dict[str, Dict[str, float]]] = None,
        relation_type: str = "default",
    ) -> None:
        """Add a directed dependency edge: parent -> child.

        Args:
            parent_id: The premise fact.
            child_id: The conclusion fact that depends on the premise.
            cpt: Conditional probability table. If None, an empty CPT is
                used (lookups then fall back to the uniform default 0.5).
            relation_type: Semantic tag for the edge ("supports",
                "contradicts", "neutral", or "default"). Only consumed by
                the experimental CO engine; NTM-BP ignores it.
        """
        self.add_node(parent_id)
        self.add_node(child_id)
        if child_id not in self._out[parent_id]:
            self._out[parent_id].append(child_id)
        if parent_id not in self._in[child_id]:
            self._in[child_id].append(parent_id)
        self._cpt[(parent_id, child_id)] = cpt if cpt is not None else {}
        self._relation_types[(parent_id, child_id)] = relation_type

    def remove_edge(self, parent_id: str, child_id: str) -> None:
        """Remove a dependency edge."""
        if child_id in self._out.get(parent_id, []):
            self._out[parent_id].remove(child_id)
        if parent_id in self._in.get(child_id, []):
            self._in[child_id].remove(parent_id)
        self._cpt.pop((parent_id, child_id), None)
        self._relation_types.pop((parent_id, child_id), None)

    def remove_node(self, fact_id: str) -> None:
        """Remove a fact node and all its incident edges."""
        for child in list(self._out.get(fact_id, [])):
            self._cpt.pop((fact_id, child), None)
            self._relation_types.pop((fact_id, child), None)
            self._in[child].remove(fact_id)
        for parent in list(self._in.get(fact_id, [])):
            self._cpt.pop((parent, fact_id), None)
            self._relation_types.pop((parent, fact_id), None)
            self._out[parent].remove(fact_id)
        self._out.pop(fact_id, None)
        self._in.pop(fact_id, None)
        self._nodes.discard(fact_id)

    def has_node(self, fact_id: str) -> bool:
        """Check if a fact exists as a node."""
        return fact_id in self._nodes

    def has_edge(self, parent_id: str, child_id: str) -> bool:
        """Check if a dependency edge exists."""
        return (parent_id, child_id) in self._cpt

    # ------------------------------------------------------------------
    # CPT management
    # ------------------------------------------------------------------

    def set_cpt(
        self,
        parent_id: str,
        child_id: str,
        cpt: Dict[str, Dict[str, float]],
    ) -> None:
        """Set the conditional probability table for an edge.

        Args:
            parent_id: Premise fact ID.
            child_id: Conclusion fact ID.
            cpt: CPT dict where cpt[parent_value][child_value] = probability.

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self.has_edge(parent_id, child_id):
            raise KeyError(f"Edge ({parent_id} -> {child_id}) does not exist.")
        self._cpt[(parent_id, child_id)] = cpt

    def get_cpt(
        self, parent_id: str, child_id: str
    ) -> Dict[str, Dict[str, float]]:
        """Get the CPT for an edge.

        Returns:
            CPT dict, or empty dict if no CPT is set.

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self.has_edge(parent_id, child_id):
            raise KeyError(f"Edge ({parent_id} -> {child_id}) does not exist.")
        return self._cpt[(parent_id, child_id)]

    def get_conditional_probability(
        self, parent_id: str, child_id: str, parent_value: str, child_value: str
    ) -> float:
        """Get P(child_value | parent_value) from the CPT.

        Returns:
            The conditional probability, or 0.5 (uniform) if unknown.
        """
        cpt = self._cpt.get((parent_id, child_id), {})
        if parent_value in cpt and child_value in cpt[parent_value]:
            return cpt[parent_value][child_value]
        return 0.5  # Uniform default

    def get_relation_type(self, parent_id: str, child_id: str) -> str:
        """Get the relation type tag of an edge.

        Returns:
            "supports", "contradicts", "neutral", or "default" (missing
            edges / untagged edges fall back to "default").
        """
        return self._relation_types.get((parent_id, child_id), "default")

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def children(self, fact_id: str) -> List[str]:
        """Return the direct children (dependents) of a fact."""
        return list(self._out.get(fact_id, []))

    def parents(self, fact_id: str) -> List[str]:
        """Return the direct parents (premises) of a fact."""
        return list(self._in.get(fact_id, []))

    def descendants(self, fact_id: str) -> Set[str]:
        """Return all reachable descendant nodes via iterative DFS."""
        seen: Set[str] = set()
        stack = list(self._out.get(fact_id, []))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self._out.get(node, []))
        return seen

    def ancestors(self, fact_id: str) -> Set[str]:
        """Return all ancestor nodes (premises this fact depends on)."""
        seen: Set[str] = set()
        stack = list(self._in.get(fact_id, []))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self._in.get(node, []))
        return seen

    def reachable_from(self, fact_id: str) -> Set[str]:
        """Return all nodes reachable from `fact_id` (inclusive).

        This is the set of nodes a cascade could affect.
        """
        result = self.descendants(fact_id)
        result.add(fact_id)
        return result

    def bfs_order(self, source: str) -> List[str]:
        """Return nodes in BFS order from the source (exclusive of source)."""
        order: List[str] = []
        frontier = list(self._out.get(source, []))
        seen = set(frontier)
        while frontier:
            node = frontier.pop(0)
            order.append(node)
            for child in self._out.get(node, []):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return order

    # ------------------------------------------------------------------
    # Graph properties
    # ------------------------------------------------------------------

    def has_cycle(self) -> bool:
        """Check whether the graph contains any cycle (iterative DFS)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in self._nodes}
        for start in self._nodes:
            if color[start] != WHITE:
                continue
            stack = [(start, iter(self._out.get(start, [])))]
            color[start] = GRAY
            while stack:
                node, it = stack[-1]
                advanced = False
                for child in it:
                    if color[child] == GRAY:
                        return True
                    if color[child] == WHITE:
                        color[child] = GRAY
                        stack.append((child, iter(self._out.get(child, []))))
                        advanced = True
                        break
                if not advanced:
                    color[node] = BLACK
                    stack.pop()
        return False

    def is_tree(self) -> bool:
        """Check whether the graph is acyclic."""
        return not self.has_cycle()

    @property
    def node_count(self) -> int:
        """Number of fact nodes in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of dependency edges."""
        return len(self._cpt)

    @property
    def nodes(self) -> List[str]:
        """List all fact IDs in the graph."""
        return list(self._nodes)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary.

        The output matches the ``dependency_graph`` object of the JSON
        Schema (see ``schemas/dependency_graph.schema.json``): edges are
        ``{"parent", "child", "cpt", "relation_type"}`` objects.
        """
        return {
            "nodes": sorted(self._nodes),
            "edges": [
                {
                    "parent": u,
                    "child": v,
                    "cpt": dict(cpt),
                    "relation_type": self._relation_types.get((u, v), "default"),
                }
                for (u, v), cpt in sorted(self._cpt.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DependencyGraph":
        """Deserialize a graph from a dictionary (JSON Schema format)."""
        g = cls()
        for node in data.get("nodes", []):
            g.add_node(node)
        for edge in data.get("edges", []):
            g.add_edge(
                parent_id=edge["parent"],
                child_id=edge["child"],
                cpt=edge.get("cpt", {}),
                relation_type=edge.get("relation_type", "default"),
            )
        return g

    def __repr__(self) -> str:
        return (
            f"DependencyGraph(nodes={self.node_count}, edges={self.edge_count}, "
            f"tree={self.is_tree()})"
        )
