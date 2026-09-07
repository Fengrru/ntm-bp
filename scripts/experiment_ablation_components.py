"""Exp 6: Ablation - component contribution analysis.

Compares NTM-BP variants: full, no gamma decay, no CPT, BFS baseline.

Usage:
    python scripts/experiment_ablation_components.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ntm_bp.evaluation import PartialContradictionLoader
from ntm_bp.evaluation import ExperimentConfig, ExperimentRunner

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_ablation_components_results.json")


def load_scenarios() -> list:
    path = os.path.join(DATA_DIR, "pc500.json")
    loader = PartialContradictionLoader()
    return loader.load(path)


def run_ablation(scenarios: list, mode: str, gamma: float,
                 tau: float, label: str) -> dict:
    """Run an ablation variant."""

    class PreloadedLoader:
        def load(self, path): return scenarios

    config = ExperimentConfig(
        tau=tau,
        gamma=gamma,
        propagation_cutoff=0.01,
        mode=mode,
        description=label,
        loader=PreloadedLoader(),
    )
    runner = ExperimentRunner(config)
    result = runner.run()
    return result.summary


def main():
    print("Loading PC-500 scenarios...")
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios\n")

    variants = [
        ("full_ntm_bp", "ntm_bp", 0.7, 0.7, "Full NTM-BP"),
        ("no_gamma_decay", "ntm_bp", 1.0, 0.7, "No gamma decay (gamma=1.0)"),
        ("bfs_baseline", "bfs", 0.7, 0.7, "BFS baseline"),
    ]

    results = {}

    for key, mode, gamma, tau, label in variants:
        print(f"  Running {label}...")
        t0 = time.time()
        summary = run_ablation(scenarios, mode, gamma, tau, label)
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.2f}s")
        results[key] = summary

    # Print table
    print("\n" + "=" * 60)
    print("Component Ablation Comparison")
    print("=" * 60)
    print("\n| Variant | RR | F1 | FNR | FPR | AGM |")
    print("|---------|-----|-----|-----|-----|-----|")
    for key, _, _, _, label in variants:
        s = results[key]
        print(f"| {label} | {s.get('avg_recompute_rate', 0):.4f} | "
              f"{s.get('avg_f1', 0):.4f} | {s.get('avg_fnr', 0):.4f} | "
              f"{s.get('avg_fpr', 0):.4f} | {s.get('avg_agm_distance', 0):.4f} |")

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
