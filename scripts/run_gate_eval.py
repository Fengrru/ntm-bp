"""Semantic gate evaluation on PC-500 / SynDep (requires an LLM client).

Runs the experimental semantic gate engines (v1/v2/v3 scalar prompts and
the Q1/Q2/Q3 structured gate) over both datasets.  Every LLM call is
recorded in ``call_log`` entries (prompt + raw response + parsed value)
and dumped to ``data/gate_call_log.json`` so results can be audited with
``scripts/audit_cache.py``.

Usage:
    # Pipeline test with the deterministic mock (NOT experimental evidence):
    python scripts/run_gate_eval.py --mock

    # Real experiment (reads DEEPSEEK_API_KEY from the environment):
    python scripts/run_gate_eval.py

Notes on scientific integrity:
* The prompts must NOT receive relation_type/theta annotations -- the
  gate engines in ntm_bp.experimental.semantic_gate guarantee this.
* Only dev cases (data/case_split.json) may be used for prompt
  iteration; the reported numbers come from the sealed test split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ntm_bp.evaluation import PartialContradictionLoader, Scenario, SynDepLoader
from ntm_bp.experimental.semantic_gate import (
    PROMPT_BUILDERS,
    SemanticGateEngine,
    StructuredGateEngine,
)
from ntm_bp.llm import DeepSeekClient, LLMInterface, MockLLMClient
from ntm_bp.memory import MemoryState

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_gate_results.json")
LOG_PATH = os.path.join(DATA_DIR, "gate_call_log.json")

TAU = 0.7
TAU_GATE = 0.5


def make_llm(use_mock: bool) -> LLMInterface:
    """Build the LLM client (mock for plumbing, real otherwise)."""
    if use_mock:
        return MockLLMClient()
    return DeepSeekClient(temperature=0.1)


def make_memory(scenario: Scenario) -> MemoryState:
    """Build memory + graph from a scenario (shared by all engines)."""
    from ntm_bp.graph import DependencyGraph

    memory = MemoryState(tau=TAU)
    graph = DependencyGraph()
    for fid, (value, confidence) in scenario.facts.items():
        memory.add_fact(fid, value, confidence)
        graph.add_node(fid)
    for parent_id, child_id in scenario.dependencies:
        edge_key = f"{parent_id}__{child_id}"
        rel_type = scenario.relation_types.get(edge_key, "default")
        graph.add_edge(parent_id, child_id, relation_type=rel_type)
    memory.graph = graph
    return memory


def run_gate_variant(
    llm: LLMInterface,
    scenarios: list,
    engine_cls,
    label: str,
    **engine_kwargs,
) -> dict:
    """Run one gate variant over all scenarios, collecting metrics + call log."""
    engine = engine_cls(llm, **engine_kwargs)
    scenario_results = []
    all_calls = []

    for s in scenarios:
        memory = make_memory(s)
        old_value = memory.get_fact(s.update_fact_id).value
        memory.update_fact(s.update_fact_id, s.new_value, s.new_confidence)
        engine.propagate(memory, s.update_fact_id, old_value, s.new_value, s.new_confidence)

        predicted = set(memory.get_reeval_facts())
        actual = s.ground_truth_reeval
        tp = len(predicted & actual)
        fp = len(predicted - actual)
        fn = len(actual - predicted)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 1.0 if (not predicted and not actual) else (
            2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        )
        scenario_results.append({
            "scenario_id": s.scenario_id,
            "theta": s.theta,
            "f1": f1,
            "predicted": sorted(predicted),
            "actual": sorted(actual),
        })

    for call in engine.call_log:
        entry = dict(call)
        entry["_variant"] = label
        entry["_scenario_count"] = len(scenarios)
        all_calls.append(entry)

    avg_f1 = sum(r["f1"] for r in scenario_results) / len(scenario_results) if scenario_results else 0.0
    return {
        "label": label,
        "avg_f1": avg_f1,
        "scenario_count": len(scenario_results),
        "scenario_results": scenario_results,
        "calls": all_calls,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock", action="store_true",
        help="use MockLLMClient (plumbing test only, NOT experimental evidence)")
    parser.add_argument(
        "--dataset", choices=["pc500", "syndep"], default="pc500",
        help="dataset to evaluate on (default: pc500)")
    args = parser.parse_args()

    llm = make_llm(args.mock)
    if args.mock:
        print("WARNING: running with MockLLMClient -- results are NOT experimental evidence.\n")

    path = os.path.join(DATA_DIR, f"{args.dataset}.json")
    loader = PartialContradictionLoader() if args.dataset == "pc500" else SynDepLoader()
    scenarios = loader.load(path)
    print(f"Loaded {len(scenarios)} scenarios from {args.dataset}.json")

    results = {}
    all_calls = []

    # Scalar-impact gates (v1/v2/v3 frozen prompts)
    for name, builder in PROMPT_BUILDERS.items():
        print(f"  Running SemanticGateEngine [{name}]...")
        res = run_gate_variant(
            llm, scenarios, SemanticGateEngine,
            label=f"semantic_gate_{name}", prompt_builder=builder,
            tau_gate=TAU_GATE,
        )
        results[name] = res
        all_calls.extend(res["calls"])

    # Structured Q1/Q2/Q3 gate
    print("  Running StructuredGateEngine [Q1/Q2/Q3]...")
    res = run_gate_variant(
        llm, scenarios, StructuredGateEngine,
        label="structured_gate_q123",
    )
    results["q123"] = res
    all_calls.extend(res["calls"])

    # Summary table
    print("\n" + "=" * 60)
    print(f"Gate comparison on {args.dataset.upper()} (LLM calls: {len(all_calls)})")
    print("=" * 60)
    print(f"| Variant | F1 | Scenarios | Calls |")
    print("|---------|-----|-----------|-------|")
    for key in ["v1", "v2", "v3", "q123"]:
        r = results[key]
        print(f"| {key} | {r['avg_f1']:.4f} | {r['scenario_count']} | {len(r['calls'])} |")

    # Save results + full call log
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "calls"}
                   for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump({"metadata": {"mock": args.mock, "dataset": args.dataset},
                   "calls": all_calls}, f, ensure_ascii=False, indent=2)
    print(f"Call log saved to {LOG_PATH} (audit with scripts/audit_cache.py)")


if __name__ == "__main__":
    main()
