"""Minimal end-to-end usage of ntm-bp.

Scenario: a credit-approval agent holds three facts
    f1: "Acme Corp 信誉良好"      (reputation premise)
    f2: "Contract A 低风险"       (risk assessment, strongly depends on f1)
    f3: "推荐投资"               (investment decision, weakly depends on f2)

A new report downgrades confidence in Acme's creditworthiness; f1's
belief collapses from 0.9 to 0.4 (the fact value itself is unchanged).
NTM-BP decays f2 and f3 proportionally to dependency strength, marking
only the conclusions that really need LLM re-evaluation.
"""

from ntm_bp import FactStatus, MemoryState, NtmBpEngine
from ntm_bp.graph import DependencyGraph


def main() -> None:
    # 1. Build memory + dependency graph with CPTs.
    mem = MemoryState(tau=0.7)
    mem.add_fact("f1", "Acme Corp 信誉良好", 0.9)
    mem.add_fact("f2", "Contract A 低风险", 0.85)
    mem.add_fact("f3", "推荐投资", 0.8)

    graph = DependencyGraph()
    # Strong first edge: f2 is fully determined by f1 (P=1.0).
    graph.add_edge(
        "f1", "f2",
        cpt={"Acme Corp 信誉良好": {"Contract A 低风险": 1.0}},
    )
    # Weak second edge: f3 is nearly independent of f2 (P=0.1).
    graph.add_edge(
        "f2", "f3",
        cpt={"Contract A 低风险": {"推荐投资": 0.1}},
    )
    mem.graph = graph

    # 2. The premise's confidence collapses (downgrade report).
    old_belief = mem.get_belief("f1")
    mem.update_fact("f1", "Acme Corp 信誉良好", 0.4)

    # 3. Run NTM-BP.
    engine = NtmBpEngine(gamma=0.7, tau=0.7)
    mem, report = engine.propagate_with_report(mem, "f1", source_old_belief=old_belief)

    # 4. Inspect the result: f2 is re-evaluated, f3 survives.
    print(f"source: {report['source']}")
    for fid, (old, new) in report["belief_changes"].items():
        status = mem.facts[fid].status.value
        print(f"  {fid}: belief {old:.3f} -> {new:.3f}  [{status}]")
    print(f"\n{report['reeval_count']} fact(s) need LLM re-evaluation: "
          f"{report['reeval_facts']}")
    print(f"f3 status: {mem.facts['f3'].status.value} "
          f"(belief {mem.get_belief('f3'):.3f}) - "
          f"{'ACTIVE' if mem.facts['f3'].status == FactStatus.ACTIVE else 'would trigger an LLM call'}")


if __name__ == "__main__":
    main()
