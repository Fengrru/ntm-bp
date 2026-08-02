"""CO relation-type ablation on PC-500 (5 variants).

Runs the experimental CO engine with different relation-type
configurations.  This reproduces the paper's negative result: CO on
all-neutral edges degrades badly (F1 drops), which is why the CO engine
lives under ``ntm_bp.experimental`` rather than in the stable API.

Usage:
    python scripts/run_co_relation_eval.py
"""

from __future__ import annotations

import json
import os
import random
import time

from ntm_bp.evaluation import (
    MODE_BFS,
    MODE_CO,
    MODE_NTM_BP,
    ExperimentConfig,
    ExperimentRunner,
    PartialContradictionLoader,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_co_relation_results.json")


def load_scenarios() -> list:
    path = os.path.join(DATA_DIR, "pc500.json")
    loader = PartialContradictionLoader()
    return loader.load(path)


def run_co_with_relation_type(scenarios: list, rel_type: str) -> dict:
    """Run CO with all edges set to a specific relation_type."""

    class PreloadedLoader:
        def load(self, path):
            # Add relation_types to scenarios
            for s in scenarios:
                rt_dict = {}
                for p, c in s.dependencies:
                    rt_dict[f"{p}__{c}"] = rel_type
                s.relation_types = rt_dict
            return scenarios

    config = ExperimentConfig(
        tau=0.7,
        gamma=0.7,
        propagation_cutoff=0.01,
        mode=MODE_CO,
        description=f"CO ({rel_type})",
        loader=PreloadedLoader(),
    )
    runner = ExperimentRunner(config)
    result = runner.run()
    return {
        "summary": result.summary,
        "wall_time_seconds": result.wall_time_seconds,
    }


def main():
    print("Loading PC-500 scenarios...")
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios\n")

    variants = [
        ("ntm_bp", MODE_NTM_BP, "NTM-BP (default)"),
        ("co_supports", None, "CO (all supports)"),
        ("co_contradicts", None, "CO (all contradicts)"),
        ("co_neutral", None, "CO (all neutral)"),
        ("co_mixed", None, "CO (mixed)"),
    ]

    results = {}

    for key, mode, label in variants:
        print(f"  Running {label}...")
        t0 = time.time()

        if mode == MODE_NTM_BP:
            # Run normal NTM-BP as baseline
            class Loader:
                def load(self, path): return scenarios
            config = ExperimentConfig(
                tau=0.7, gamma=0.7, propagation_cutoff=0.01,
                mode=mode, description=label, loader=Loader(),
            )
            result = ExperimentRunner(config).run()
            summary = result.summary
        else:
            # Determine relation type
            if key == "co_mixed":
                rng = random.Random(42)
                rel_types = ["supports", "contradicts", "neutral"]
                # Mix per-edge
                class MixedLoader:
                    def load(self, path):
                        for s in scenarios:
                            rt_dict = {}
                            for p, c in s.dependencies:
                                rt_dict[f"{p}__{c}"] = rng.choice(rel_types)
                            s.relation_types = rt_dict
                        return scenarios
                config = ExperimentConfig(
                    tau=0.7, gamma=0.7, propagation_cutoff=0.01,
                    mode=MODE_CO, description=label, loader=MixedLoader(),
                )
                result = ExperimentRunner(config).run()
                summary = result.summary
            else:
                rel_map = {"co_supports": "supports", "co_contradicts": "contradicts",
                           "co_neutral": "neutral"}
                res = run_co_with_relation_type(scenarios, rel_map[key])
                summary = res["summary"]

        elapsed = time.time() - t0
        print(f"    done in {elapsed:.2f}s")
        results[key] = summary

    # Print comparison table
    print("\n" + "=" * 60)
    print("CO Relation Type Comparison")
    print("=" * 60)
    print("\n| Variant | RR | F1 | FNR | FPR | AGM |")
    print("|---------|-----|-----|-----|-----|-----|")
    for key, _, label in variants:
        s = results[key]
        print(f"| {label} | {s.get('avg_recompute_rate', 0):.4f} | "
              f"{s.get('avg_f1', 0):.4f} | {s.get('avg_fnr', 0):.4f} | "
              f"{s.get('avg_fpr', 0):.4f} | {s.get('avg_agm_distance', 0):.4f} |")

    # Save
    output = {k: v for k, v in results.items()}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
