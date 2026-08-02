"""Tests for the decay arithmetic and propagation queue."""

import pytest

from ntm_bp.propagation import DecayCalculator, PropagationMessage, PropagationQueue


class TestDecayCalculator:
    def test_next_delta_formula(self):
        """delta_next = current_delta * gamma * P(child | parent)."""
        calc = DecayCalculator(gamma=0.7, cutoff=0.01)
        # 0.5 * 0.7 * 0.5 = 0.175
        assert calc.compute_next_delta(0.5, 0.5) == pytest.approx(0.175)

    def test_next_delta_below_cutoff_returns_zero(self):
        """Deltas at or below the cutoff are truncated to 0."""
        calc = DecayCalculator(gamma=0.7, cutoff=0.01)
        assert calc.compute_next_delta(0.01, 0.5) == pytest.approx(0.0035 - 0.0035)  # 0
        assert calc.compute_next_delta(0.001, 0.5) == 0.0

    def test_new_belief_formula(self):
        """B_new = B_old * (1 - delta)."""
        calc = DecayCalculator()
        assert calc.compute_new_belief(0.8, 0.25) == pytest.approx(0.6)

    def test_new_belief_clamped(self):
        """Belief never leaves [0, 1]."""
        calc = DecayCalculator()
        assert calc.compute_new_belief(0.5, 2.0) == 0.0
        assert calc.compute_new_belief(0.5, -1.0) == 1.0

    def test_validation(self):
        """Invalid gamma/cutoff are rejected."""
        with pytest.raises(ValueError):
            DecayCalculator(gamma=0.0)
        with pytest.raises(ValueError):
            DecayCalculator(gamma=1.5)
        with pytest.raises(ValueError):
            DecayCalculator(cutoff=0.0)
        with pytest.raises(ValueError):
            DecayCalculator(cutoff=2.0)

    def test_gamma_zero_disables_multi_hop(self):
        """gamma -> 0 means no propagation beyond the first hop."""
        calc = DecayCalculator(gamma=0.001, cutoff=0.00001)
        assert calc.compute_next_delta(0.5, 0.5) > 0  # still nonzero
        assert calc.compute_next_delta(0.5, 0.5) == pytest.approx(0.5 * 0.001 * 0.5)


class TestPropagationQueue:
    def test_priority_order(self):
        """Higher delta (lower priority value) is popped first."""
        q = PropagationQueue()
        q.push(PropagationMessage(priority=-0.3, fact_id="f2", delta=0.3))
        q.push(PropagationMessage(priority=-0.7, fact_id="f1", delta=0.7))
        assert q.pop().fact_id == "f1"
        assert q.pop().fact_id == "f2"

    def test_empty(self):
        q = PropagationQueue()
        assert q.is_empty
        assert q.pop() is None

    def test_size_and_clear(self):
        q = PropagationQueue()
        q.push(PropagationMessage(priority=-0.1, fact_id="f1", delta=0.1))
        assert q.size == 1
        q.clear()
        assert q.size == 0
        assert q.is_empty

    def test_metadata_preserved(self):
        """Metadata travels with the message."""
        q = PropagationQueue()
        q.push(
            PropagationMessage(
                priority=-0.2, fact_id="f2", delta=0.2,
                metadata={"parent_id": "f1"},
            )
        )
        msg = q.pop()
        assert msg.metadata == {"parent_id": "f1"}
