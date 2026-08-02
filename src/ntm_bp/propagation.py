"""Propagation utilities for the NTM-BP engine.

Priority-queue message passing and the decay arithmetic used by
Algorithm 1 of the NTM whitepaper:

    delta_init   = source_change_ratio * P(child | parent)
    delta_next   = delta * gamma * P(child | parent)
    B_new        = B_old * (1 - delta)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(order=True)
class PropagationMessage:
    """A message in the propagation queue.

    Attributes:
        priority: Sort key (lower = processed first). For NTM-BP this is
            the negative delta, so larger decay is processed first.
        fact_id: The target fact ID.
        delta: The accumulated decay factor [0, 1].
        metadata: Additional data (e.g., parent_id).
    """

    priority: float
    fact_id: str = field(compare=False)
    delta: float = field(compare=False, default=0.0)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)


class PropagationQueue:
    """Priority queue for belief-propagation messages."""

    def __init__(self) -> None:
        self._heap: List[PropagationMessage] = []

    def push(self, message: PropagationMessage) -> None:
        """Push a message onto the queue."""
        heapq.heappush(self._heap, message)

    def pop(self) -> Optional[PropagationMessage]:
        """Pop the highest-priority message, or None if empty."""
        if self._heap:
            return heapq.heappop(self._heap)
        return None

    @property
    def is_empty(self) -> bool:
        """Check whether the queue is empty."""
        return len(self._heap) == 0

    @property
    def size(self) -> int:
        """Number of messages in the queue."""
        return len(self._heap)

    def clear(self) -> None:
        """Remove all messages."""
        self._heap.clear()


class DecayCalculator:
    """Calculates belief decay factors for the NTM-BP algorithm."""

    def __init__(self, gamma: float = 0.7, cutoff: float = 0.01) -> None:
        """Initialize the decay arithmetic.

        Args:
            gamma: Propagation decay coefficient (default 0.7).
            cutoff: Minimum delta threshold for propagation (default 0.01).

        Raises:
            ValueError: If gamma or cutoff is outside (0, 1].
        """
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if not 0.0 < cutoff <= 1.0:
            raise ValueError(f"cutoff must be in (0, 1], got {cutoff}")

        self.gamma = gamma
        self.cutoff = cutoff

    def compute_next_delta(self, current_delta: float, conditional_prob: float) -> float:
        """Compute the delta to propagate to the next hop.

        delta_next = current_delta * gamma * P(child | parent)

        Args:
            current_delta: Accumulated decay factor from parent.
            conditional_prob: P(child | parent) from CPT.

        Returns:
            Next-hop delta, or 0.0 if below cutoff.
        """
        raw = current_delta * self.gamma * conditional_prob
        return raw if raw > self.cutoff else 0.0

    def compute_new_belief(self, old_belief: float, delta: float) -> float:
        """Compute the updated belief after applying decay.

        B_new = B_old * (1 - delta)

        Args:
            old_belief: Current belief value [0, 1].
            delta: Decay factor [0, 1].

        Returns:
            Updated belief clamped to [0, 1].
        """
        return max(0.0, min(1.0, old_belief * (1.0 - delta)))
