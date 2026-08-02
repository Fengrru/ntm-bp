"""Exp 1: Main three-way comparison experiment (BFS vs NTM-BP vs CO).

Runs all three methods on both PC-500 and SynDep datasets and prints
Markdown comparison tables.  This reproduces Figure 1 of the paper
(CO == NTM-BP here because the datasets carry no relation annotations,
so every edge is "default").

Usage:
    python scripts/run_main_eval.py
"""

from __future__ import annotations

import json
import os
import time

from ntm_bp.evaluation import (
    ALL_MODES,
    MODE_BFS,
    MODE_CO,
    MODE_NTM_BP,
    ExperimentConfig,
    ExperimentRunner,
    PartialContradictionLoader,
    SynDepLoader,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_main_results.json")


def load_dataset(name: str) -> list:
    """Load a dataset by name."""
    if name == "syndep":
        path = os.path.join(DATA_DIR, "syndep.json")
        loader = SynDepLoader()
    else:
        path = os.path.join(DATA_DIR, "pc500.json")
        loader = PartialContradictionLoader()
    return loader.load(path)


def run_method(mode: str, scenarios: list, desc: str) -> dict:
    """Run a single method on all scenarios."""

    class PreloadedLoader:
        def load(self, path):
            return scenarios

    config = ExperimentConfig(
        tau=0.7,
        gamma=0.7,
        propagation_cutoff=0.01,
        mode=mode,
        description=desc,
        loader=PreloadedLoader(),
    )
    runner = ExperimentRunner(config)
    result = runner.run()
    return {
        "summary": result.summary,
        "scenario_results": result.scenario_results,
        "wall_time_seconds": result.wall_time_seconds,
    }


def theta_bucket(theta: float) -> str:
    if theta <= 0.3:
        return "low (<=0.3)"
    elif theta <= 0.7:
        return "mid (0.3-0.7)"
    else:
        return "high (>0.7)"


def print_table(label: str, summaries: dict[str, dict], scenarios: list):
    """Print a Markdown comparison table."""
    print(f"\n### {label}\n")
    print("| Metric | BFS | NTM-BP | CO |")
    print("|--------|-----|--------|-----|")
    for metric in ["avg_recompute_rate", "avg_f1", "avg_fnr", "avg_fpr", "avg_agm_distance"]:
        row = [metric]
        for key in ["bfs", "ntm_bp", "co"]:
            row.append(f"{summaries[key].get(metric, 0.0):.4f}")
        print("| " + " | ".join(row) + " |")
    print(f"| scenario_count | {summaries['bfs'].get('scenario_count', 0)} "
          f"| {summaries['ntm_bp'].get('scenario_count', 0)} "
          f"| {summaries['co'].get('scenario_count', 0)} |")

    # Theta buckets
    buckets = {"low (<=0.3)": [], "mid (0.3-0.7)": [], "high (>0.7)": []}
    for s in scenarios:
        buckets[theta_bucket(s.theta)].append(s)

    print("\n| Bucket | Count | Metric | BFS | NTM-BP | CO |")
    print("|-------|-------|--------|-----|--------|-----|")
    for bname in ["low (<=0.3)", "mid (0.3-0.7)", "high (>0.7)"]:
        bscenarios = buckets[bname]
        if not bscenarios:
            continue
        bresults = {}
        for mode_key, mode in [("bfs", MODE_BFS),
                                ("ntm_bp", MODE_NTM_BP),
                                ("co", MODE_CO)]:
            bresults[mode_key] = run_method(mode, bscenarios, f"{mode_key} {bname}")["summary"]
        for metric in ["avg_recompute_rate", "avg_f1"]:
            row = [bname, str(len(bscenarios)), metric]
            for key in ["bfs", "ntm_bp", "co"]:
                row.append(f"{bresults[key].get(metric, 0.0):.4f}")
            print("| " + " | ".join(row) + " |")


def main():
    datasets = ["pc500", "syndep"]
    all_data = {}

    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"{'=' * 60}")
        scenarios = load_dataset(ds_name)
        print(f"Loaded {len(scenarios)} scenarios\n")

        ds_result = {"scenarios_count": len(scenarios), "methods": {}}

        for mode_key, mode, label in [
            ("bfs", MODE_BFS, "BFS"),
            ("ntm_bp", MODE_NTM_BP, "NTM-BP"),
            ("co", MODE_CO, "CO"),
        ]:
            print(f"  Running {label}...")
            t0 = time.time()
            result = run_method(mode, scenarios, label)
            elapsed = time.time() - t0
            print(f"    done in {elapsed:.2f}s")
            ds_result["methods"][mode_key] = {
                "summary": result["summary"],
                "scenario_results": result["scenario_results"],
                "wall_time_seconds": result["wall_time_seconds"],
            }

        all_data[ds_name] = ds_result

        # Print table
        summaries = {k: v["summary"] for k, v in ds_result["methods"].items()}
        print_table(ds_name.upper(), summaries, scenarios)

        # Key finding
        bfs_rr = summaries["bfs"].get("avg_recompute_rate", 0.0)
        ntm_rr = summaries["ntm_bp"].get("avg_recompute_rate", 0.0)
        co_rr = summaries["co"].get("avg_recompute_rate", 0.0)
        ntm_red = ((bfs_rr - ntm_rr) / bfs_rr * 100) if bfs_rr > 0 else 0.0
        co_red = ((bfs_rr - co_rr) / bfs_rr * 100) if bfs_rr > 0 else 0.0
        print(f"\n  NTM-BP RR reduction: {ntm_red:.1f}%  |  CO RR reduction: {co_red:.1f}%")

    # Save all results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
