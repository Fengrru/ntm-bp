"""BFS cascade baseline: the binary STALE algorithm from classical TMS.

When a fact is updated, a BFS traversal marks every reachable downstream
fact as STALE (belief set to 0.0).  This is the baseline system compared
against NTM-BP in the paper (Section 7, "Exp 1: three-way comparison").
"""

from __future__ import annotations

from collections import deque
from typing import Set

from ntm_bp.memory import MemoryState


class BfsCascadeEngine:
    """CoW+BFS baseline cascade engine.

    Attributes:
        max_depth: Maximum BFS traversal depth (0 = unlimited, default).
    """

    def __init__(self, max_depth: int = 0) -> None:
        self.max_depth = max_depth

    def cascade(self, memory: MemoryState, source_fact_id: str) -> MemoryState:
        """Execute a BFS cascade from the updated fact.

        Args:
            memory: The current memory state (will be mutated).
            source_fact_id: The fact that was just updated.

        Returns:
            The updated memory state with downstream facts marked STALE.

        Raises:
            KeyError: If source_fact_id is not in the dependency graph.
            RuntimeError: If no dependency graph is attached to memory.
        """
        graph = memory.graph
        if graph is None:
            raise RuntimeError(
                "No dependency graph available. "
                "Set memory.graph before calling cascade()."
            )

        if source_fact_id not in graph.nodes:
            raise KeyError(
                f"Source fact '{source_fact_id}' not found in dependency graph. "
                "Facts must be added to the graph before cascading."
            )

        # BFS traversal with deque (O(1) popleft) and depth tracking
        visited: Set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        queue.append((source_fact_id, 0))
        visited.add(source_fact_id)

        while queue:
            current, depth = queue.popleft()

            if current != source_fact_id:
                # Mark as STALE (belief set to 0.0)
                memory.mark_stale(current)

            # Add children within depth limit (max_depth=0 means unlimited)
            if self.max_depth == 0 or depth + 1 <= self.max_depth:
                for child in graph.children(current):
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, depth + 1))

        return memory

    def cascade_with_report(
        self, memory: MemoryState, source_fact_id: str
    ) -> tuple[MemoryState, dict]:
        """Like cascade(), but returns a report of what was affected.

        Returns:
            Tuple of (updated_memory, report_dict) where report includes:
            - affected_count: number of facts marked STALE
            - affected_facts: list of affected fact IDs
        """
        before = set(memory.get_stale_facts())
        memory = self.cascade(memory, source_fact_id)
        after = set(memory.get_stale_facts())
        affected = list(after - before)

        return memory, {
            "source": source_fact_id,
            "affected_count": len(affected),
            "affected_facts": affected,
        }
