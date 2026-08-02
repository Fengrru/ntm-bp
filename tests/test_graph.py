"""Tests for the stdlib-only DependencyGraph."""

import pytest

from ntm_bp.graph import DependencyGraph


class TestGraphBasics:
    def test_add_nodes_and_edges(self):
        g = DependencyGraph()
        g.add_edge("f1", "f2")
        g.add_edge("f1", "f3")
        assert g.node_count == 3
        assert g.edge_count == 2
        assert g.has_node("f1")
        assert g.has_edge("f1", "f2")

    def test_children_and_parents(self):
        g = DependencyGraph()
        g.add_edge("f1", "f2")
        g.add_edge("f2", "f3")
        assert g.children("f1") == ["f2"]
        assert g.parents("f3") == ["f2"]
        assert g.children("f3") == []

    def test_remove_edge_and_node(self):
        g = DependencyGraph()
        g.add_edge("f1", "f2")
        g.add_edge("f2", "f3")
        g.remove_edge("f1", "f2")
        assert not g.has_edge("f1", "f2")
        g.remove_node("f2")
        assert not g.has_node("f2")
        assert g.children("f1") == []
        assert g.edge_count == 0

    def test_reachable_and_descendants(self):
        g = DependencyGraph()
        g.add_edge("f1", "f2")
        g.add_edge("f2", "f3")
        g.add_edge("f3", "f4")
        assert g.descendants("f1") == {"f2", "f3", "f4"}
        assert g.ancestors("f4") == {"f1", "f2", "f3"}
        assert g.reachable_from("f2") == {"f2", "f3", "f4"}

    def test_bfs_order(self):
        g = DependencyGraph()
        g.add_edge("f1", "f2")
        g.add_edge("f1", "f3")
        g.add_edge("f2", "f4")
        assert g.bfs_order("f1") == ["f2", "f3", "f4"]


class TestGraphCpt:
    def test_default_uniform_probability(self):
        g = DependencyGraph()
        g.add_edge("f1", "f2")
        assert g.get_conditional_probability("f1", "f2", "a", "b") == 0.5

    def test_cpt_lookup(self):
        g = DependencyGraph()
        g.add_edge(
            "f1", "f2",
            cpt={"premise": {"conclusion": 0.9, "other": 0.1}},
        )
        assert g.get_conditional_probability("f1", "f2", "premise", "conclusion") == 0.9
        assert g.get_conditional_probability("f1", "f2", "unknown", "x") == 0.5

    def test_set_cpt(self):
        g = DependencyGraph()
        g.add_edge("f1", "f2")
        g.set_cpt("f1", "f2", {"a": {"b": 0.7}})
        assert g.get_conditional_probability("f1", "f2", "a", "b") == 0.7
        assert g.get_cpt("f1", "f2") == {"a": {"b": 0.7}}

    def test_set_cpt_missing_edge_raises(self):
        g = DependencyGraph()
        with pytest.raises(KeyError):
            g.set_cpt("f1", "f2", {"a": {"b": 0.7}})


class TestGraphCycles:
    def test_chain_is_acyclic(self):
        g = DependencyGraph()
        g.add_edge("f1", "f2")
        g.add_edge("f2", "f3")
        assert g.is_tree()
        assert not g.has_cycle()

    def test_cycle_detected(self):
        g = DependencyGraph()
        g.add_edge("f1", "f2")
        g.add_edge("f2", "f3")
        g.add_edge("f3", "f1")
        assert g.has_cycle()
        assert not g.is_tree()

    def test_self_loop_detected(self):
        g = DependencyGraph()
        g.add_edge("f1", "f1")
        assert g.has_cycle()


class TestGraphSerialization:
    def test_roundtrip(self):
        g = DependencyGraph()
        g.add_edge("f1", "f2", cpt={"a": {"b": 0.8}})
        g.add_edge("f2", "f3")
        data = g.to_dict()
        g2 = DependencyGraph.from_dict(data)
        assert g2.node_count == 3
        assert g2.edge_count == 2
        assert g2.get_conditional_probability("f1", "f2", "a", "b") == 0.8
        assert g2.has_edge("f2", "f3")
