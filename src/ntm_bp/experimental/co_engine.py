"""CO (Cognitive Override): relationship-aware belief propagation.

EXPERIMENTAL -- not part of the stable core API.  Ported from the NTM
paper project; kept under :mod:`ntm_bp.experimental` because the paper
reports that CO degrades on ``neutral`` relation chains (F1 drops to
0.709 on PC-500).  Use the standard :class:`ntm_bp.NtmBpEngine` unless
you know your relation annotations are reliable.

Extends NtmBpEngine with relationship-type-dependent gamma modulation.
Different dependency relationships (supports, contradicts, neutral) have
different propagation strengths.

Algorithm:
    NTM-BP base with relation-aware gamma modulation for subsequent hops.
    Initial delta follows NTM-BP formula (no gamma in initial).
    Subsequent hops use:
        gamma_effective = gamma * RELATION_MULTIPLIER[relation_type]
    Where:
        "supports":    1.2 (enhanced propagation - support is strong)
        "contradicts": 1.3 (strong propagation - contradictions spread fast)
        "neutral":     0.5 (weak propagation - neutral facts are loosely coupled)
        "default":     1.0 (standard NTM-BP propagation)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ntm_bp.engine import NtmBpEngine
from ntm_bp.graph import DependencyGraph
from ntm_bp.memory import FactStatus, MemoryState
from ntm_bp.propagation import DecayCalculator


# Gamma multipliers for each relationship type
RELATION_MULTIPLIER: Dict[str, float] = {
    "supports": 1.2,
    "contradicts": 1.3,
    "neutral": 0.5,
    "default": 1.0,
}


class CoEngine(NtmBpEngine):
    """Cognitive Override engine with relationship-aware propagation.

    Inherits all NTM-BP functionality and overrides the delta computation
    to incorporate edge-level relation_type information for subsequent hops.
    Initial delta follows the same formula as NTM-BP (no gamma),
    making CO == NTM-BP when all relations are "default".
    """

    def __init__(
        self,
        gamma: float = 0.7,
        tau: float = 0.7,
        cutoff: float = 0.01,
        max_iterations: int = 1000,
        lazy: bool = False,
    ) -> None:
        super().__init__(gamma=gamma, tau=tau, cutoff=cutoff, max_iterations=max_iterations, lazy=lazy)

    # ------------------------------------------------------------------
    # Override delta computation to add relation awareness
    # ------------------------------------------------------------------

    def _compute_initial_delta(
        self, cond_prob: float, source_change_ratio: float = 1.0
    ) -> float:
        """Compute initial delta with relation-aware modulation.

        Uses same base formula as NTM-BP: source_change_ratio * cond_prob
        No gamma in initial delta (consistent with NTM-BP).

        Args:
            cond_prob: P(child | parent) from CPT.
            source_change_ratio: How much the source fact's belief changed.

        Returns:
            Initial delta.
        """
        # Same formula as NTM-BP for initial delta
        return source_change_ratio * cond_prob

    def _compute_next_delta(
        self, graph: DependencyGraph, parent_id: str, child_id: str,
        current_delta: float, memory: MemoryState
    ) -> float:
        """Compute next-hop delta with relation-aware gamma modulation.

        delta_next = current_delta * gamma_effective * cond_prob
        where gamma_effective = min(gamma * multiplier, 1.0)

        Args:
            graph: Dependency graph.
            parent_id: Current fact being propagated from.
            child_id: Target fact.
            current_delta: Accumulated delta from parent.
            memory: Memory state.

        Returns:
            Next-hop delta, or 0.0 if below cutoff.
        """
        cond_prob = self._get_conditional_prob(
            graph, parent_id, child_id, memory
        )
        relation_type = graph.get_relation_type(parent_id, child_id)
        multiplier = RELATION_MULTIPLIER.get(relation_type, 1.0)
        effective_gamma = min(self._decay.gamma * multiplier, 1.0)
        raw = current_delta * effective_gamma * cond_prob
        return raw if raw > self._decay.cutoff else 0.0

    # ------------------------------------------------------------------
    # propagate_with_report with relation-type stats
    # ------------------------------------------------------------------

    def propagate_with_report(
        self, memory: MemoryState, source_fact_id: str,
        source_old_belief: Optional[float] = None,
    ) -> Tuple[MemoryState, dict]:
        """Like propagate(), but returns a detailed report with relation-type stats."""
        before = {fid: memory.get_belief(fid) for fid in memory.facts}
        memory = self.propagate(memory, source_fact_id, source_old_belief)

        reeval_facts = memory.get_reeval_facts()
        belief_changes = {}
        for fid in memory.facts:
            old = before.get(fid)
            new = memory.get_belief(fid)
            if old != new:
                belief_changes[fid] = (old, new)

        # Count by relation type
        graph = memory.graph
        type_counts: Dict[str, int] = {}
        if graph is not None:
            for fid in reeval_facts:
                for parent_id in graph.parents(fid):
                    rt = graph.get_relation_type(parent_id, fid)
                    type_counts[rt] = type_counts.get(rt, 0) + 1

        affected = [fid for fid in memory.facts if memory.facts[fid].status == FactStatus.REEVAL_REQUIRED]

        return memory, {
            "source": source_fact_id,
            "reeval_count": len(affected),
            "reeval_facts": affected,
            "belief_changes": belief_changes,
            "relation_type_counts": type_counts,
        }
