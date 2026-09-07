"""Exp: is the gamma decay component helpful, harmful, or noise?

Motivated by the component-ablation finding that gamma=1.0 (no decay)
scored HIGHER F1 than the default gamma=0.7. This script sweeps gamma
over PC-500 + SynDep and quantifies uncertainty with CLUSTER bootstrap
(resampling (template_id, new_value) cases, since jitter instances of a
case are near-duplicates).

For each gamma it reports F1 / RR with 95% CIs, plus the CI of the
PAIRED per-case F1 difference against gamma=1.0 (the sharpest test of
whether decay helps: if the difference CI excludes 0, the effect is
real, not small-sample noise).

Purely numeric -- no LLM required.

Usage:
    python scripts/experiment_gamma_analysis.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ntm_bp.engine import NtmBpEngine
from ntm_bp.memory import MemoryState
from ntm_bp.evaluation import compute_f1
from ntm_bp.graph import DependencyGraph

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_gamma_analysis_results.json")

GAMMAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BASELINE_GAMMA = 1.0  # paired comparison target (no decay)
N_BOOTSTRAP = 5000


def build_memory(s: dict) -> MemoryState:
    mem = MemoryState(tau=0.7)
    graph = DependencyGraph()
    for fid, (val, conf) in s["facts"].items():
        mem.add_fact(fid, val, conf)
        graph.add_node(fid)
    for p, c in s["dependencies"]:
        graph.add_edge(p, c)
    mem.graph = graph
    return mem


def run_dataset(name: str, path: str, rng: random.Random) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)["scenarios"]
    n = len(scenarios)
    print(f"\n=== {name}: {n} instances ===")

    # per gamma: case_key -> list of (f1, rr)
    by_case = {g: defaultdict(list) for g in GAMMAS}
    for s in scenarios:
        case_key = f"{s['template_id']}||{s['new_value']}"
        gt = set(s["ground_truth_reeval"])
        fid = s["update_fact_id"]
        for g in GAMMAS:
            mem = build_memory(s)
            ob = mem.get_belief(fid)
            mem.update_fact(fid, s["new_value"], s["new_confidence"])
            NtmBpEngine(gamma=g, tau=0.7).propagate(
                mem, fid, source_old_belief=ob)
            pred = set(mem.get_reeval_facts())
            rr = len(pred) / mem.fact_count
            by_case[g][case_key].append((compute_f1(pred, gt), rr))

    cases = sorted(by_case[GAMMAS[0]].keys())

    def cluster_ci(values_by_case, idx):
        """Bootstrap CI of the mean of metric idx (0=f1, 1=rr)."""
        means = []
        for _ in range(N_BOOTSTRAP):
            sampled = [cases[rng.randrange(len(cases))] for _ in range(len(cases))]
            vals = [v[idx] for ck in sampled for v in values_by_case[ck]]
            means.append(sum(vals) / len(vals))
        means.sort()
        return means[int(0.025 * N_BOOTSTRAP)], means[int(0.975 * N_BOOTSTRAP) - 1]

    summary = {}
    print(f"  {'gamma':>5} {'F1':>8} {'95% CI':>18} {'RR':>8}")
    for g in GAMMAS:
        all_vals = [v for ck in cases for v in by_case[g][ck]]
        f1 = sum(v[0] for v in all_vals) / len(all_vals)
        rr = sum(v[1] for v in all_vals) / len(all_vals)
        lo, hi = cluster_ci(by_case[g], 0)
        summary[str(g)] = {"avg_f1": f1, "f1_ci95": [lo, hi], "avg_rr": rr}
        print(f"  {g:>5} {f1:>8.4f} [{lo:.4f}, {hi:.4f}] {rr:>8.4f}")

    # Paired per-case F1 difference: gamma vs BASELINE_GAMMA (no decay)
    print(f"\n  Paired F1 diff vs gamma={BASELINE_GAMMA} (positive = decay helps):")
    diffs_out = {}
    base_case_means = {
        ck: sum(v[0] for v in by_case[BASELINE_GAMMA][ck]) / len(by_case[BASELINE_GAMMA][ck])
        for ck in cases
    }
    for g in GAMMAS:
        if g == BASELINE_GAMMA:
            continue
        case_diffs = {
            ck: sum(v[0] for v in by_case[g][ck]) / len(by_case[g][ck])
            - base_case_means[ck]
            for ck in cases
        }
        point = sum(case_diffs.values()) / len(cases)
        means = []
        for _ in range(N_BOOTSTRAP):
            sampled = [cases[rng.randrange(len(cases))] for _ in range(len(cases))]
            means.append(sum(case_diffs[ck] for ck in sampled) / len(sampled))
        means.sort()
        lo, hi = means[int(0.025 * N_BOOTSTRAP)], means[int(0.975 * N_BOOTSTRAP) - 1]
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        diffs_out[str(g)] = {"mean_diff": point, "ci95": [lo, hi],
                             "significant": lo > 0 or hi < 0}
        print(f"    gamma={g}: diff={point:+.4f} [{lo:+.4f}, {hi:+.4f}]  ({sig})")

    return {"n_instances": n, "n_cases": len(cases),
            "per_gamma": summary,
            "paired_f1_diff_vs_no_decay": diffs_out}


def main():
    rng = random.Random(42)
    results = {}
    for name, fname in [("pc500", "pc500.json"), ("syndep", "syndep.json")]:
        results[name] = run_dataset(name, os.path.join(DATA_DIR, fname), rng)

    output = {
        "metadata": {
            "gammas": GAMMAS,
            "baseline_gamma": BASELINE_GAMMA,
            "tau": 0.7,
            "n_bootstrap": N_BOOTSTRAP,
            "method": (
                "Cluster bootstrap over (template_id, new_value) cases; "
                "paired per-case F1 differences against gamma=1.0 (no decay)."
            ),
        },
        "results": results,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
