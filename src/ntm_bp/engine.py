"""NTM-BP: Neural Truth Maintenance with Belief Propagation.

Implements Algorithm 1 of the NTM whitepaper: replaces binary STALE
marking with continuous belief decay using a priority-queue-based
belief propagation over the probabilistic dependency graph.
"""

from __future__ import annotations

from typing import Optional, Set, Tuple

from ntm_bp.graph import DependencyGraph
from ntm_bp.memory import FactStatus, MemoryState
from ntm_bp.propagation import DecayCalculator, PropagationMessage, PropagationQueue


class NtmBpEngine:
    """Probabilistic belief propagation engine for NTM.

    When a fact is updated, propagates confidence decay along
    dependency edges using conditional probability tables, marking
    facts as REEVAL_REQUIRED only when their confidence drops below
    the threshold tau.

    Attributes:
        gamma: Propagation decay coefficient (default 0.7).
        tau: Re-evaluation threshold (default 0.7).
        cutoff: Minimum delta for propagation (default 0.01).
        max_iterations: Safety limit for cyclic graphs (default 1000).
        lazy: If True, only mark REEVAL_REQUIRED without triggering
            immediate re-inference (default False).
    """

    def __init__(
        self,
        gamma: float = 0.7,
        tau: float = 0.7,
        cutoff: float = 0.01,
        max_iterations: int = 1000,
        lazy: bool = False,
    ) -> None:
        self._decay = DecayCalculator(gamma=gamma, cutoff=cutoff)
        self.tau = tau
        self.max_iterations = max_iterations
        self.lazy = lazy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(
        self,
        memory: MemoryState,
        source_fact_id: str,
        source_old_belief: Optional[float] = None,
    ) -> MemoryState:
        """Execute NTM-BP belief propagation from an updated fact.

        Args:
            memory: Memory state (will be mutated).
            source_fact_id: The fact that was just updated.
            source_old_belief: The belief of the source fact BEFORE the
                update. Used to scale the initial delta proportionally to
                the actual change. If None, defaults to 1.0 (full change
                assumed).

        Returns:
            Updated memory with propagated beliefs and REEVAL_REQUIRED marks.

        Raises:
            RuntimeError: If no dependency graph is attached to memory.
            KeyError: If source_fact_id is not in the graph.
        """
        graph = memory.graph
        if graph is None:
            raise RuntimeError(
                "No dependency graph available. "
                "Set memory.graph before calling propagate()."
            )
        if source_fact_id not in graph.nodes:
            raise KeyError(
                f"Source fact '{source_fact_id}' not found in dependency graph."
            )

        source_new_belief = memory.get_belief(source_fact_id)

        # Compute source belief change ratio. This scales the initial
        # delta: a small change in the source causes a proportionally
        # small cascade. The sigmoid raw/(raw+alpha) ensures small
        # changes still trigger a noticeable cascade.
        SIGMOID_ALPHA = 0.15
        source_change_ratio = 1.0
        if source_old_belief is not None and source_old_belief > 0:
            raw_change = max(
                0.0, abs(source_old_belief - source_new_belief) / source_old_belief
            )
            source_change_ratio = raw_change / (raw_change + SIGMOID_ALPHA)

        # Initialize propagation queue with direct children of source
        queue = PropagationQueue()
        visited: Set[str] = {source_fact_id}

        for child_id in graph.children(source_fact_id):
            cond_prob = self._get_conditional_prob(
                graph, source_fact_id, child_id, memory
            )
            delta = self._compute_initial_delta(cond_prob, source_change_ratio)
            if delta <= self._decay.cutoff:
                continue
            queue.push(
                PropagationMessage(
                    priority=-delta,
                    fact_id=child_id,
                    delta=delta,
                    metadata={"parent_id": source_fact_id},
                )
            )
            visited.add(child_id)

        # Iterative propagation
        iterations = 0

        while not queue.is_empty and iterations < self.max_iterations:
            msg = queue.pop()
            if msg is None:
                break
            iterations += 1

            fact_id = msg.fact_id
            delta = msg.delta

            # Update belief: B_new = B_old * (1 - delta)
            old_belief = memory.get_belief(fact_id) or 0.0
            new_belief = self._decay.compute_new_belief(old_belief, delta)
            memory.set_belief(fact_id, new_belief)

            # Mark if below threshold
            if new_belief < self.tau:
                memory.mark_reeval(fact_id)

            # Propagate to children (subsequent hops use gamma decay)
            for child_id in graph.children(fact_id):
                if child_id in visited:
                    continue
                visited.add(child_id)

                delta_next = self._compute_next_delta(
                    graph, fact_id, child_id, delta, memory
                )
                if delta_next > self._decay.cutoff:
                    queue.push(
                        PropagationMessage(
                            priority=-delta_next,
                            fact_id=child_id,
                            delta=delta_next,
                            metadata={"parent_id": fact_id},
                        )
                    )

        return memory

    def propagate_with_report(
        self,
        memory: MemoryState,
        source_fact_id: str,
        source_old_belief: Optional[float] = None,
    ) -> Tuple[MemoryState, dict]:
        """Like propagate(), but returns a detailed report.

        Returns:
            Tuple of (updated_memory, report_dict) with keys:
            - source: source fact ID
            - reeval_count: number of facts marked REEVAL_REQUIRED
            - reeval_facts: list of affected fact IDs
            - belief_changes: dict of fact_id -> (old_belief, new_belief)
            - iterations: number of propagation steps
        """
        before = {
            fid: memory.get_belief(fid)
            for fid in memory.facts
        }

        memory = self.propagate(memory, source_fact_id, source_old_belief)

        reeval_facts = memory.get_reeval_facts()
        belief_changes = {}
        for fid in memory.facts:
            old = before.get(fid)
            new = memory.get_belief(fid)
            if old != new:
                belief_changes[fid] = (old, new)

        affected = [
            fid for fid in memory.facts
            if memory.facts[fid].status == FactStatus.REEVAL_REQUIRED
        ]

        return memory, {
            "source": source_fact_id,
            "reeval_count": len(affected),
            "reeval_facts": affected,
            "belief_changes": belief_changes,
            "iterations": len(belief_changes),
        }

    # ------------------------------------------------------------------
    # Delta computation (can be overridden by subclasses)
    # ------------------------------------------------------------------

    def _compute_initial_delta(
        self, cond_prob: float, source_change_ratio: float = 1.0
    ) -> float:
        """Compute initial delta for direct children of the updated fact.

        delta_init = source_change_ratio * P(child | parent)

        The decay transmitted along an edge is proportional to the
        edge's dependency strength: a conclusion fully determined by
        the premise (P -> 1) inherits the full source change, while a
        negligible dependency (P -> 0) transmits no decay.

        Args:
            cond_prob: P(child | parent) from CPT.
            source_change_ratio: How much the source fact's belief
                changed relative to its original value [0, 1].

        Returns:
            Initial delta, or 0.0 if below cutoff.
        """
        return source_change_ratio * cond_prob

    def _compute_next_delta(
        self,
        graph: DependencyGraph,
        parent_id: str,
        child_id: str,
        current_delta: float,
        memory: MemoryState,
    ) -> float:
        """Compute delta for subsequent hops (with gamma decay).

        delta_next = current_delta * gamma * P(child | parent)

        Args:
            graph: Dependency graph.
            parent_id: Current fact being propagated from.
            child_id: Target fact.
            current_delta: Accumulated delta from parent.
            memory: Memory state (for fact value lookup).

        Returns:
            Next-hop delta, or 0.0 if below cutoff.
        """
        cond_prob = self._get_conditional_prob(
            graph, parent_id, child_id, memory
        )
        return self._decay.compute_next_delta(current_delta, cond_prob)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_conditional_prob(
        graph: DependencyGraph,
        parent_id: str,
        child_id: str,
        memory: MemoryState,
    ) -> float:
        """Look up P(child_value | parent_value) from the CPT.

        Uses the current value of each fact to index the CPT. Falls
        back to 0.5 (uniform) if the CPT entry is missing.
        """
        parent_fact = memory.get_fact(parent_id)
        child_fact = memory.get_fact(child_id)
        if parent_fact is None or child_fact is None:
            return 0.5

        return graph.get_conditional_probability(
            parent_id,
            child_id,
            parent_fact.value,
            child_fact.value,
        )
