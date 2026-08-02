"""Tests for NtmBpEngine (NTM-BP algorithm).

Ported from the research codebase's tests/test_ntm_bp.py; the BFS
comparison test is rewritten without the (out-of-scope) baseline engine.
"""

import pytest

from ntm_bp.engine import NtmBpEngine
from ntm_bp.graph import DependencyGraph
from ntm_bp.memory import FactStatus, MemoryState


# ------------------------------------------------------------------
# Fixtures for common graph topologies
# ------------------------------------------------------------------

@pytest.fixture
def chain_memory():
    """f1 -> f2 -> f3 chain with uniform CPT (default 0.5)."""
    mem = MemoryState(tau=0.7)
    mem.add_fact("f1", "Acme Corp 信誉良好", 0.9)
    mem.add_fact("f2", "Contract A 低风险", 0.85)
    mem.add_fact("f3", "推荐投资", 0.8)

    graph = DependencyGraph()
    graph.add_edge("f1", "f2")
    graph.add_edge("f2", "f3")
    mem.graph = graph
    return mem, graph


@pytest.fixture
def branch_memory():
    """f1 -> f2, f1 -> f3 (branching from source)."""
    mem = MemoryState(tau=0.7)
    mem.add_fact("f1", "root", 0.9)
    mem.add_fact("f2", "child_a", 0.85)
    mem.add_fact("f3", "child_b", 0.85)

    graph = DependencyGraph()
    graph.add_edge("f1", "f2")
    graph.add_edge("f1", "f3")
    mem.graph = graph
    return mem, graph


@pytest.fixture
def deterministic_memory():
    """Chain with deterministic CPT (P=1.0) - no decay expected."""
    mem = MemoryState(tau=0.7)
    mem.add_fact("f1", "true", 0.9)
    mem.add_fact("f2", "true", 0.85)
    mem.add_fact("f3", "true", 0.8)

    graph = DependencyGraph()
    graph.add_edge("f1", "f2", cpt={"true": {"true": 1.0}})
    graph.add_edge("f2", "f3", cpt={"true": {"true": 1.0}})
    mem.graph = graph
    return mem, graph


@pytest.fixture
def weak_dependency_memory():
    """Edge with very weak dependency (P=0.2) - strong decay expected."""
    mem = MemoryState(tau=0.7)
    mem.add_fact("f1", "premise", 0.9)
    mem.add_fact("f2", "conclusion", 0.85)

    graph = DependencyGraph()
    graph.add_edge("f1", "f2", cpt={"premise": {"conclusion": 0.2}})
    mem.graph = graph
    return mem, graph


@pytest.fixture
def cyclic_memory():
    """f1 -> f2 -> f3 -> f1 (cycle)."""
    mem = MemoryState(tau=0.7)
    mem.add_fact("f1", "a", 0.9)
    mem.add_fact("f2", "b", 0.85)
    mem.add_fact("f3", "c", 0.8)

    graph = DependencyGraph()
    graph.add_edge("f1", "f2")
    graph.add_edge("f2", "f3")
    graph.add_edge("f3", "f1")
    mem.graph = graph
    return mem, graph


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestNtmBpEngine:
    """Core NTM-BP algorithm tests."""

    def test_simple_chain(self, chain_memory):
        """Belief decays along a chain with source change scaling."""
        mem, _ = chain_memory
        old_belief = mem.get_belief("f1")  # 0.9
        mem.update_fact("f1", "new value", 0.4)

        ALPHA = 0.15
        # source_change = (0.9-0.4)/0.9 = 0.556, sigmoid = 0.556/0.706 = 0.787
        engine = NtmBpEngine(gamma=0.7, tau=0.7)
        result = engine.propagate(mem, "f1", source_old_belief=old_belief)

        # f1 keeps its new confidence
        assert result.get_belief("f1") == 0.4
        assert result.facts["f1"].status == FactStatus.ACTIVE

        source_change = (0.9 - 0.4) / 0.9
        adjusted = source_change / (source_change + ALPHA)
        # f2: delta_init = adjusted * 0.5
        expected_f2 = 0.85 * (1 - adjusted * 0.5)
        assert result.get_belief("f2") == pytest.approx(expected_f2)
        assert result.facts["f2"].status == FactStatus.REEVAL_REQUIRED

        # f3: delta_next = (adjusted * 0.5) * 0.7 * 0.5
        expected_f3 = 0.8 * (1 - adjusted * 0.5 * 0.7 * 0.5)
        assert result.get_belief("f3") == pytest.approx(expected_f3)
        assert result.facts["f3"].status == FactStatus.REEVAL_REQUIRED

    def test_branch_propagation(self, branch_memory):
        """Both branches decay independently with source change scaling."""
        mem, _ = branch_memory
        old_belief = mem.get_belief("f1")  # 0.9
        mem.update_fact("f1", "new value", 0.4)

        ALPHA = 0.15
        engine = NtmBpEngine(gamma=0.7, tau=0.7)
        result = engine.propagate(mem, "f1", source_old_belief=old_belief)

        src = (0.9 - 0.4) / 0.9
        adjusted = src / (src + ALPHA)
        expected = 0.85 * (1 - adjusted * 0.5)
        assert result.get_belief("f2") == pytest.approx(expected)
        assert result.get_belief("f3") == pytest.approx(expected)
        assert result.facts["f2"].status == FactStatus.REEVAL_REQUIRED
        assert result.facts["f3"].status == FactStatus.REEVAL_REQUIRED

    def test_deterministic_dependency(self, deterministic_memory):
        """P=1.0 (fully determined) -> full source change is inherited.

        Paper: a conclusion whose value is fully determined by the
        premise (P -> 1) inherits the full source change.
        """
        mem, _ = deterministic_memory
        mem.update_fact("f1", "true", 0.4)  # no old belief -> full change

        engine = NtmBpEngine(gamma=0.7, tau=0.7)
        result = engine.propagate(mem, "f1")

        # delta_init = 1.0 * 1.0 = 1.0 -> B(f2) = 0.85 * 0.0 = 0.0
        assert result.get_belief("f2") == 0.0
        assert result.facts["f2"].status == FactStatus.REEVAL_REQUIRED
        # delta_next = 1.0 * 0.7 * 1.0 = 0.7 -> B(f3) = 0.8 * 0.3 = 0.24
        assert result.get_belief("f3") == pytest.approx(0.24)
        assert result.facts["f3"].status == FactStatus.REEVAL_REQUIRED

    def test_weak_dependency(self, weak_dependency_memory):
        """P=0.2 (negligible dependency) -> little decay.

        Paper: an edge with negligible dependency (P -> 0) transmits
        no decay.
        """
        mem, _ = weak_dependency_memory
        old_belief = mem.get_belief("f1")  # 0.9
        mem.update_fact("f1", "premise", 0.4)

        engine = NtmBpEngine(gamma=0.7, tau=0.7)
        result = engine.propagate(mem, "f1", source_old_belief=old_belief)

        # source_change = 0.556, sigmoid = 0.787, cond_prob = 0.2
        # delta_init = 0.787 * 0.2 = 0.157, B_new = 0.85 * 0.843 = 0.716
        ALPHA = 0.15
        src = (0.9 - 0.4) / 0.9
        adjusted = src / (src + ALPHA)
        expected = 0.85 * (1 - adjusted * 0.2)
        assert result.get_belief("f2") == pytest.approx(expected)
        assert result.facts["f2"].status == FactStatus.ACTIVE  # 0.716 > 0.7

    def test_threshold_above_tau(self):
        """Small source change + strong edge -> belief stays above tau."""
        mem = MemoryState(tau=0.5)
        mem.add_fact("f1", "premise", 0.9)
        mem.add_fact("f2", "conclusion", 0.95)
        graph = DependencyGraph()
        graph.add_edge("f1", "f2", cpt={"premise": {"conclusion": 0.9}})
        mem.graph = graph

        old = mem.get_belief("f1")
        mem.update_fact("f1", "premise", 0.8)

        engine = NtmBpEngine(gamma=0.7, tau=0.5)
        result = engine.propagate(mem, "f1", source_old_belief=old)

        # raw = 0.1/0.9 = 0.111, sigmoid = 0.426, delta = 0.426*0.9 = 0.383
        # B(f2) = 0.95 * 0.617 = 0.586 > 0.5 -> ACTIVE
        raw = 0.1 / 0.9
        expected = 0.95 * (1 - raw / (raw + 0.15) * 0.9)
        assert result.get_belief("f2") == pytest.approx(expected)
        assert result.facts["f2"].status == FactStatus.ACTIVE

    def test_threshold_below_tau(self):
        """Belief drops below tau -> REEVAL_REQUIRED."""
        mem = MemoryState(tau=0.7)
        mem.add_fact("f1", "premise", 0.9)
        mem.add_fact("f2", "conclusion", 0.5)
        graph = DependencyGraph()
        graph.add_edge("f1", "f2", cpt={"premise": {"conclusion": 0.5}})
        mem.graph = graph

        mem.update_fact("f1", "premise", 0.3)

        engine = NtmBpEngine(gamma=0.7, tau=0.7)
        result = engine.propagate(mem, "f1")

        # delta = 0.5, B_new = 0.5 * 0.5 = 0.25 < 0.7
        assert result.facts["f2"].status == FactStatus.REEVAL_REQUIRED

    def test_truncation(self):
        """Very deep chain -> propagation stops at cutoff."""
        mem = MemoryState(tau=0.0)  # Never trigger REEVAL
        for i in range(10):
            mem.add_fact(f"f{i}", f"v{i}", 0.9)
        graph = DependencyGraph()
        for i in range(9):
            graph.add_edge(f"f{i}", f"f{i+1}")
        mem.graph = graph

        mem.update_fact("f0", "v0_new", 0.5)

        engine = NtmBpEngine(gamma=0.5, tau=0.0, cutoff=0.01)
        result = engine.propagate(mem, "f0")

        # Delta decays exponentially; far nodes stay unchanged.
        assert result.get_belief("f9") == 0.9

    def test_cyclic_graph(self, cyclic_memory):
        """Cycle should not cause an infinite loop."""
        mem, _ = cyclic_memory
        mem.update_fact("f1", "new value", 0.4)

        engine = NtmBpEngine(gamma=0.7, tau=0.7, max_iterations=50)
        result = engine.propagate(mem, "f1")

        assert result.get_belief("f2") < 0.85
        assert result.get_belief("f3") < 0.8
        assert result.get_belief("f1") == 0.4

    def test_no_graph(self):
        """Should raise RuntimeError when no graph is set."""
        mem = MemoryState()
        mem.add_fact("f1", "value", 0.9)
        engine = NtmBpEngine()
        with pytest.raises(RuntimeError, match="No dependency graph"):
            engine.propagate(mem, "f1")

    def test_nonexistent_source(self):
        """Should raise KeyError when source is not in graph."""
        mem = MemoryState()
        mem.add_fact("f1", "value", 0.9)
        graph = DependencyGraph()
        graph.add_node("f2")
        mem.graph = graph
        engine = NtmBpEngine()
        with pytest.raises(KeyError):
            engine.propagate(mem, "f1")

    def test_gamma_variation(self):
        """Higher gamma -> more decay in subsequent hops."""
        ALPHA = 0.15

        def build_and_run(gamma):
            mem = MemoryState(tau=0.0)
            mem.add_fact("f1", "v1", 0.9)
            mem.add_fact("f2", "v2", 0.9)
            mem.add_fact("f3", "v3", 0.9)
            g = DependencyGraph()
            g.add_edge("f1", "f2")
            g.add_edge("f2", "f3")
            mem.graph = g
            old = mem.get_belief("f1")
            mem.update_fact("f1", "v1_new", 0.5)
            return NtmBpEngine(gamma=gamma, tau=0.0).propagate(
                mem, "f1", source_old_belief=old
            )

        result_low = build_and_run(0.3)
        result_high = build_and_run(0.9)

        # f2 identical for both (gamma applies only from the 2nd hop)
        assert result_low.get_belief("f2") == result_high.get_belief("f2")
        assert result_low.get_belief("f3") > result_high.get_belief("f3")

    def test_propagate_with_report(self, chain_memory):
        """Report should contain correct metadata."""
        mem, _ = chain_memory
        mem.update_fact("f1", "new value", 0.4)

        engine = NtmBpEngine(gamma=0.7, tau=0.7)
        result, report = engine.propagate_with_report(mem, "f1")

        assert report["source"] == "f1"
        assert report["reeval_count"] == 2
        assert "f2" in report["reeval_facts"]
        assert "f3" in report["reeval_facts"]
        assert "f2" in report["belief_changes"]
        assert "f3" in report["belief_changes"]

    def test_partial_overthrow(self):
        """Partial overthrow: strong first hop re-evaluates f2, weak
        second hop leaves f3 ACTIVE.

        f1 (premise) -> f2 (conclusion) -> f3 (decision) with a strong
        first edge (P=1.0, fully determined) and a weak second edge
        (P=0.1, nearly independent). After a belief collapse at f1, f2
        is re-evaluated but f3 retains enough confidence to stay
        ACTIVE - the key difference vs. binary BFS, which would
        blanket-invalidate both.
        """
        mem = MemoryState(tau=0.7)
        mem.add_fact("f1", "premise", 0.9)
        mem.add_fact("f2", "conclusion", 0.85)
        mem.add_fact("f3", "decision", 0.8)

        graph = DependencyGraph()
        graph.add_edge("f1", "f2", cpt={"premise": {"conclusion": 1.0}})
        graph.add_edge("f2", "f3", cpt={"conclusion": {"decision": 0.1}})
        mem.graph = graph

        old_belief = mem.get_belief("f1")  # 0.9
        mem.update_fact("f1", "premise", 0.4)

        engine = NtmBpEngine(gamma=0.7, tau=0.7)
        result = engine.propagate(mem, "f1", source_old_belief=old_belief)

        # f2: delta = 0.787 * 1.0 = 0.787 -> B = 0.85*0.213 = 0.181 < 0.7
        assert result.facts["f2"].status == FactStatus.REEVAL_REQUIRED
        # f3: delta = 0.787 * 0.7 * 0.1 = 0.055 -> B = 0.8*0.945 = 0.756 > 0.7
        assert result.facts["f3"].status == FactStatus.ACTIVE
        assert result.get_reeval_facts() == ["f2"]

    def test_lazy_mode(self, chain_memory):
        """Lazy mode still updates beliefs and marks REEVAL_REQUIRED."""
        mem, _ = chain_memory
        old_belief = mem.get_belief("f1")  # 0.9
        mem.update_fact("f1", "new value", 0.4)

        engine = NtmBpEngine(gamma=0.7, tau=0.7, lazy=True)
        result = engine.propagate(mem, "f1", source_old_belief=old_belief)

        ALPHA = 0.15
        src = (0.9 - 0.4) / 0.9
        expected_f2 = 0.85 * (1 - src / (src + ALPHA) * 0.5)
        assert result.get_belief("f2") == pytest.approx(expected_f2)
        assert result.facts["f2"].status == FactStatus.REEVAL_REQUIRED

    def test_source_belief_unchanged(self):
        """The source fact's own belief remains as set by update_fact."""
        mem = MemoryState(tau=0.7)
        mem.add_fact("f1", "original", 0.9)
        mem.add_fact("f2", "dependent", 0.85)
        graph = DependencyGraph()
        graph.add_edge("f1", "f2")
        mem.graph = graph

        mem.update_fact("f1", "updated", 0.3)

        engine = NtmBpEngine()
        result = engine.propagate(mem, "f1")

        assert result.get_belief("f1") == 0.3
        assert result.facts["f1"].value == "updated"

    def test_small_source_change_small_cascade(self):
        """A small source change causes a proportionally small cascade."""
        mem = MemoryState(tau=0.3)
        mem.add_fact("f1", "premise", 0.9)
        mem.add_fact("f2", "conclusion", 0.9)
        graph = DependencyGraph()
        graph.add_edge("f1", "f2")
        mem.graph = graph

        old = mem.get_belief("f1")
        mem.update_fact("f1", "premise", 0.85)  # small change: 5.6%

        engine = NtmBpEngine(gamma=0.7, tau=0.3)
        result = engine.propagate(mem, "f1", source_old_belief=old)

        # raw = 0.0556, sigmoid = 0.0556/0.2056 = 0.270, delta = 0.135
        # B(f2) = 0.9 * 0.865 = 0.778, well above tau -> ACTIVE
        assert result.get_belief("f2") == pytest.approx(0.9 * (1 - (0.0556 / 0.2056) * 0.5), abs=1e-3)
        assert result.facts["f2"].status == FactStatus.ACTIVE
