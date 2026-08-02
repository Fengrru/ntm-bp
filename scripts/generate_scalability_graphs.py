"""Generate random dependency graphs for scalability experiments.

Produces chain + branch hybrid graphs of various sizes (10 to 1000
nodes).  Deterministic: seed 42, 20 graphs per size.

Usage:
    python scripts/generate_scalability_graphs.py
"""

from __future__ import annotations

import json
import os
import random

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "scalability")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_graph(num_nodes: int, branching_factor: int = 2, rng: random.Random = None) -> dict:
    """Generate a random dependency graph.

    Returns a dict compatible with the Scenario format (without ground truth).
    """
    if rng is None:
        rng = random.Random(42)

    # Create chain backbone
    facts = {}
    dependencies = []
    for i in range(num_nodes):
        fid = f"f{i}"
        facts[fid] = [f"fact_{i}", round(rng.uniform(0.7, 0.95), 2)]

    # Chain: f0 -> f1 -> f2 -> ...
    for i in range(num_nodes - 1):
        dependencies.append([f"f{i}", f"f{i + 1}"])

    # Add random branches
    extra_edges = 0
    for i in range(num_nodes):
        if rng.random() < 0.3 and extra_edges < num_nodes // 2:
            low = i + 2
            high = min(i + 4, num_nodes - 1)
            if low <= high:
                target = rng.randint(low, high)
                dependencies.append([f"f{i}", f"f{target}"])
                extra_edges += 1

    # Update fact is always f0, new confidence depends on graph size
    update_confidence = round(rng.uniform(0.1, 0.6), 2)

    return {
        "scenario_id": f"scal_{num_nodes}_{rng.randint(0, 9999):04d}",
        "facts": facts,
        "dependencies": dependencies,
        "update_fact_id": "f0",
        "new_value": "updated_fact",
        "new_confidence": update_confidence,
        "ground_truth_reeval": [],
        "ground_truth_values": {},
        "theta": round(1.0 - update_confidence, 2),
    }


def generate_dataset(sizes: list[int], graphs_per_size: int = 20) -> dict:
    """Generate a full scalability dataset."""
    rng = random.Random(42)
    all_scenarios = {}
    for size in sizes:
        scenarios = []
        for _ in range(graphs_per_size):
            scenarios.append(generate_graph(size, rng=rng))
        all_scenarios[str(size)] = scenarios
    return all_scenarios


def main():
    sizes = [10, 50, 100, 200, 500, 1000]
    graphs_per_size = 20

    print(f"Generating scalability graphs: sizes={sizes}, {graphs_per_size} each")
    data = generate_dataset(sizes, graphs_per_size)

    output_path = os.path.join(OUTPUT_DIR, "scalability_graphs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in data.values())
    print(f"Generated {total} graphs -> {output_path}")
    for size, scenarios in data.items():
        print(f"  size={size}: {len(scenarios)} graphs")


if __name__ == "__main__":
    main()
