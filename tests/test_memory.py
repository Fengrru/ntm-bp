"""Tests for Fact, FactVersion and MemoryState."""

import pytest

from ntm_bp.memory import Fact, FactStatus, FactVersion, MemoryState


class TestFactVersion:
    def test_validation(self):
        with pytest.raises(ValueError):
            FactVersion("v", 1.5, 0)
        with pytest.raises(ValueError):
            FactVersion("v", -0.1, 0)
        with pytest.raises(ValueError):
            FactVersion("v", 0.5, -1)

    def test_serialization_roundtrip(self):
        v = FactVersion("value", 0.8, 3)
        assert FactVersion.from_dict(v.to_dict()) == v


class TestFact:
    def test_properties(self):
        f = Fact(fact_id="f1", current_version=FactVersion("v1", 0.9, 0))
        assert f.value == "v1"
        assert f.confidence == 0.9
        assert f.timestamp == 0
        assert f.version == 1

    def test_set_version_archives(self):
        f = Fact(fact_id="f1", current_version=FactVersion("v1", 0.9, 0))
        f.set_version(FactVersion("v2", 0.4, 1))
        assert f.value == "v2"
        assert f.version == 2
        assert len(f.version_history) == 2
        assert f.version_history[0].value == "v1"

    def test_equality_by_id(self):
        a = Fact(fact_id="f1", current_version=FactVersion("v1", 0.9, 0))
        b = Fact(fact_id="f1", current_version=FactVersion("v2", 0.5, 1))
        assert a == b
        assert hash(a) == hash(b)

    def test_serialization_roundtrip(self):
        f = Fact(
            fact_id="f1",
            current_version=FactVersion("v1", 0.9, 0),
            status=FactStatus.REEVAL_REQUIRED,
            metadata={"k": "v"},
        )
        f.set_version(FactVersion("v2", 0.4, 1))
        g = Fact.from_dict(f.to_dict())
        assert g.fact_id == f.fact_id
        assert g.value == "v2"
        assert g.status == FactStatus.REEVAL_REQUIRED
        assert g.metadata == {"k": "v"}


class TestMemoryState:
    def test_add_and_get_fact(self):
        mem = MemoryState()
        fact = mem.add_fact("f1", "value", 0.9)
        assert mem.get_fact("f1") is fact
        assert mem.get_belief("f1") == 0.9
        assert mem.fact_count == 1
        assert mem.active_fact_count == 1

    def test_duplicate_fact_raises(self):
        mem = MemoryState()
        mem.add_fact("f1", "value", 0.9)
        with pytest.raises(ValueError):
            mem.add_fact("f1", "other", 0.8)

    def test_set_belief_validation(self):
        mem = MemoryState()
        mem.add_fact("f1", "value", 0.9)
        with pytest.raises(ValueError):
            mem.set_belief("f1", 1.5)
        with pytest.raises(KeyError):
            mem.set_belief("ghost", 0.5)

    def test_update_fact(self):
        mem = MemoryState()
        mem.add_fact("f1", "v1", 0.9)
        mem.update_fact("f1", "v2", 0.4)
        assert mem.get_belief("f1") == 0.4
        assert mem.get_fact("f1").value == "v2"
        assert mem.get_fact("f1").version == 2
        assert mem.time == 1

    def test_update_missing_fact_raises(self):
        mem = MemoryState()
        with pytest.raises(KeyError):
            mem.update_fact("ghost", "v", 0.5)

    def test_status_marking(self):
        mem = MemoryState()
        mem.add_fact("f1", "value", 0.9)
        mem.add_fact("f2", "value", 0.9)
        mem.mark_reeval("f1")
        assert mem.get_reeval_facts() == ["f1"]
        mem.mark_stale("f2")
        assert mem.get_stale_facts() == ["f2"]
        assert mem.get_belief("f2") == 0.0
        mem.mark_active("f1")
        assert mem.get_reeval_facts() == []

    def test_snapshot_roundtrip(self):
        mem = MemoryState(tau=0.6)
        mem.add_fact("f1", "v1", 0.9)
        mem.add_fact("f2", "v2", 0.8)
        from ntm_bp.graph import DependencyGraph

        g = DependencyGraph()
        g.add_edge("f1", "f2")
        mem.graph = g
        mem.update_fact("f1", "v1b", 0.5)

        restored = MemoryState.from_snapshot(mem.to_snapshot())
        assert restored.get_belief("f1") == 0.5
        assert restored.tau == 0.6
        assert restored.graph is not None
        assert restored.graph.has_edge("f1", "f2")
        assert restored.time == mem.time
