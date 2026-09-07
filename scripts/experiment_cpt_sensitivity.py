# -*- coding: utf-8 -*-
"""CPT estimation sensitivity analysis (offline, no LLM).

Addresses review point: "CPT estimation error impact is not discussed".

Design:
  - Reference CPT per edge: P(child_value | parent_value) = 0.8 (strong
    dependency), matching the style of DependencyExtractor.build_default_cpt.
  - For sigma in {0, 0.05, 0.10, 0.15, 0.20}, perturb each edge CPT entry
    with Gaussian noise N(0, sigma^2), clip to [0.05, 0.95], run NTM-BP
    (gamma=0.7, tau=0.7) on PC-500, and record F1 / recompute rate.
  - Also report the uniform-0.5 fallback condition (no CPT set), which is
    what the main paper runs use.
  - N_SEEDS independent noise draws per sigma.

Usage:
    python scripts/experiment_cpt_sensitivity.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ntm_bp.engine import NtmBpEngine  # noqa: E402
from ntm_bp.memory import MemoryState  # noqa: E402
from ntm_bp.evaluation import PartialContradictionLoader  # noqa: E402
from ntm_bp.evaluation import compute_f1  # noqa: E402
from ntm_bp.graph import DependencyGraph  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_PATH = os.path.join(DATA_DIR, "pc500.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_cpt_sensitivity_results.json")

SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20]
N_SEEDS = 10
BASE_CPT_PROB = 0.8
TAU = 0.7
GAMMA = 0.7
RNG_SEED = 20260805


def run_scenario(scenario, cpt_noise: dict | None) -> tuple[float, float]:
    """Run one scenario; return (f1, recompute_rate).

    cpt_noise: None -> no CPT set (uniform 0.5 fallback).
               dict {(parent, child): p} -> set explicit CPT entries.
    """
    memory = MemoryState(tau=TAU)
    graph = DependencyGraph()

    for fid, (value, confidence) in scenario.facts.items():
        memory.add_fact(fid, value, confidence)
        graph.add_node(fid)

    for parent_id, child_id in scenario.dependencies:
        graph.add_edge(parent_id, child_id)
        if cpt_noise is not None:
            p = cpt_noise[(parent_id, child_id)]
            parent_values = [scenario.facts[parent_id][0]]
            if parent_id == scenario.update_fact_id:
                parent_values.append(scenario.new_value)
            child_value = scenario.facts[child_id][0]
            cpt = {pv: {child_value: p} for pv in parent_values}
            graph.set_cpt(parent_id, child_id, cpt)

    memory.graph = graph
    old_belief = memory.get_belief(scenario.update_fact_id)
    memory.update_fact(
        scenario.update_fact_id, scenario.new_value, scenario.new_confidence
    )

    engine = NtmBpEngine(tau=TAU, gamma=GAMMA)
    engine.propagate(memory, scenario.update_fact_id, source_old_belief=old_belief)

    predicted = set(memory.get_reeval_facts())
    f1 = compute_f1(predicted, scenario.ground_truth_reeval)
    denom = max(1, len(scenario.facts) - 1)
    rr = len(predicted) / denom
    return f1, rr


def main() -> None:
    loader = PartialContradictionLoader()
    scenarios = loader.load(INPUT_PATH)
    rng = np.random.default_rng(RNG_SEED)

    results: dict = {"uniform_fallback": {}, "per_sigma": {}, "metadata": {}}

    # Condition 0: uniform 0.5 fallback (matches paper main runs)
    f1s, rrs = [], []
    for s in scenarios:
        f1, rr = run_scenario(s, None)
        f1s.append(f1)
        rrs.append(rr)
    results["uniform_fallback"] = {
        "avg_f1": float(np.mean(f1s)),
        "avg_rr": float(np.mean(rrs)),
    }

    # Perturbed CPT conditions
    for sigma in SIGMAS:
        seed_f1, seed_rr = [], []
        for seed_idx in range(N_SEEDS):
            f1s, rrs = [], []
            for s in scenarios:
                # Draw per-edge noise
                noise = {}
                for parent_id, child_id in s.dependencies:
                    p = BASE_CPT_PROB
                    if sigma > 0:
                        p = float(np.clip(p + rng.normal(0.0, sigma), 0.05, 0.95))
                    noise[(parent_id, child_id)] = p
                f1, rr = run_scenario(s, noise)
                f1s.append(f1)
                rrs.append(rr)
            seed_f1.append(float(np.mean(f1s)))
            seed_rr.append(float(np.mean(rrs)))
        results["per_sigma"][f"{sigma}"] = {
            "avg_f1_mean": float(np.mean(seed_f1)),
            "avg_f1_std": float(np.std(seed_f1)),
            "avg_rr_mean": float(np.mean(seed_rr)),
            "avg_rr_std": float(np.std(seed_rr)),
            "per_seed_f1": seed_f1,
        }

    results["metadata"] = {
        "dataset": "PC-500 (500 instances)",
        "base_cpt_prob": BASE_CPT_PROB,
        "sigmas": SIGMAS,
        "n_seeds": N_SEEDS,
        "tau": TAU,
        "gamma": GAMMA,
        "noise_model": "per-edge Gaussian N(0, sigma^2) added to P(child|parent), clipped to [0.05, 0.95]",
        "note": "uniform_fallback = no CPT set (engine uses 0.5 default), matching paper main runs.",
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print("uniform fallback:", results["uniform_fallback"])
    for sigma, r in results["per_sigma"].items():
        print(
            f"sigma={sigma}: F1={r['avg_f1_mean']:.4f}+-{r['avg_f1_std']:.4f} "
            f"RR={r['avg_rr_mean']:.4f}"
        )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
