"""SEALED TEST: one-shot evaluation on the sealed test split.

DISCIPLINE
----------
* Loads ONLY test cases from data/case_split.json (pc500 + syndep).
  co_relation is dev-only by design and has no test cases.
* Refuses to run if the output file already exists -- the test can be
  fired exactly once. All prompt iteration happened on dev; this run
  produces the final reported numbers.
* Methods: ntm_bp (numeric reference), gate_v1 (frozen original prompt),
  gate_structured (frozen structured gate, the selected method).
* Cluster bootstrap by (template_id, new_value) case, seed fixed.
* Prompt iteration is now frozen. No builder changes after this run.

Usage:
    python scripts/experiment_sealed_test.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ntm_bp.engine import NtmBpEngine
from ntm_bp.experimental.semantic_gate import (
    PROMPT_BUILDERS,
    SemanticGateEngine,
    StructuredGateEngine,
)
from ntm_bp.memory import MemoryState
from ntm_bp.evaluation import compute_f1
from ntm_bp.graph import DependencyGraph
from ntm_bp.llm import DeepSeekClient

from ntm_bp.experimental.gate_utils import CachedLLM, build_memory

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SPLIT_PATH = os.path.join(DATA_DIR, "case_split.json")
CACHE_PATH = os.path.join(DATA_DIR, "llm_gate_cache.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_sealed_test_results.json")

TAU_GATE = 0.5
N_BOOTSTRAP = 2000
METHODS = ["ntm_bp", "gate_v1", "gate_structured"]


def load_test_scenarios(split: dict) -> dict:
    out = {}
    for name in ["pc500", "syndep"]:
        test_cases = set(split["datasets"][name]["test_cases"])
        with open(os.path.join(DATA_DIR, f"{name}.json"),
                  "r", encoding="utf-8") as f:
            scenarios = json.load(f)["scenarios"]
        out[name] = [
            s for s in scenarios
            if f"{s['template_id']}||{s['new_value']}" in test_cases
        ]
    return out


def case_key(s: dict) -> str:
    return f"{s['template_id']}||{s['new_value']}"


def run_method(scenarios, method, llm):
    records = []
    for s in scenarios:
        gt = set(s["ground_truth_reeval"])
        fid = s["update_fact_id"]
        mem = build_memory(s)
        ob = mem.get_belief(fid)
        mem.update_fact(fid, s["new_value"], s["new_confidence"])
        if method == "ntm_bp":
            NtmBpEngine(gamma=0.7, tau=0.7).propagate(
                mem, fid, source_old_belief=ob)
        elif method == "gate_structured":
            StructuredGateEngine(llm).propagate(
                mem, fid, s["facts"][fid][0],
                s["new_value"], s["new_confidence"])
        else:  # gate_v1 -- frozen builder
            SemanticGateEngine(
                llm, tau_gate=TAU_GATE,
                prompt_builder=PROMPT_BUILDERS["v1"],
            ).propagate(mem, fid, s["facts"][fid][0],
                        s["new_value"], s["new_confidence"])
        pred = set(mem.get_reeval_facts())
        records.append((case_key(s), compute_f1(pred, gt), pred, gt))
    return records


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY is not set.")
        sys.exit(1)
    if os.path.isfile(OUTPUT_PATH):
        print(f"REFUSING to run: {OUTPUT_PATH} already exists.")
        print("The sealed test is one-shot. Delete it only if you know "
              "exactly why (e.g. crashed mid-run with nothing written).")
        sys.exit(1)

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)
    test = load_test_scenarios(split)
    total_cases = 0
    total_instances = 0
    for name, sc in test.items():
        n_cases = len({case_key(s) for s in sc})
        total_cases += n_cases
        total_instances += len(sc)
        print(f"sealed test {name}: {len(sc)} instances, "
              f"{n_cases} cases")
    print(f"total: {total_instances} instances, {total_cases} cases")
    if total_cases != 27:
        print(f"WARNING: expected 27 sealed cases, got {total_cases}")

    cache = {}
    if os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
    print(f"Loaded {len(cache)} cached LLM responses")
    llm = CachedLLM(DeepSeekClient(), cache)

    results = {}
    for name, scenarios in test.items():
        print(f"\n=== {name} (sealed test) ===")
        t0 = time.time()
        results[name] = {}
        f1_records = {m: [] for m in METHODS}
        for m in METHODS:
            calls_before = llm.api_calls
            recs = run_method(scenarios, m, llm)
            new_calls = llm.api_calls - calls_before
            f1_records[m] = recs
            results[name][m] = {"new_api_calls": new_calls}

        # Cluster bootstrap by case
        rng = random.Random(42)
        by_case = {m: defaultdict(list) for m in METHODS}
        for m in METHODS:
            for ck, f1, _, _ in f1_records[m]:
                by_case[m][ck].append(f1)
        cases = sorted(by_case["ntm_bp"].keys())
        n = len(scenarios)

        for m in METHODS:
            point = sum(r[1] for r in f1_records[m]) / n
            means = []
            for _ in range(N_BOOTSTRAP):
                sampled = [cases[rng.randrange(len(cases))]
                           for _ in range(len(cases))]
                vals = [f1 for ck in sampled for f1 in by_case[m][ck]]
                means.append(sum(vals) / len(vals))
            means.sort()
            ci = [means[int(0.025 * N_BOOTSTRAP)],
                  means[int(0.975 * N_BOOTSTRAP) - 1]]
            results[name][m]["avg_f1"] = point
            results[name][m]["ci95"] = ci
            print(f"  {m:<15} F1={point:.4f}  "
                  f"95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]  "
                  f"new_api_calls={results[name][m]['new_api_calls']}")

        # per-case table for the paper appendix
        per_case = {}
        for m in METHODS:
            per_case[m] = {ck: f1 for ck, f1, _, _ in f1_records[m]}
        results[name]["per_case"] = per_case

        # persist cache incrementally
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)

    output = {
        "metadata": {
            "scope": "SEALED test cases from case_split.json (one-shot)",
            "split_file": "case_split.json",
            "tau_gate": TAU_GATE,
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap": "cluster bootstrap resampling (template_id, new_value) cases",
            "llm_model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            "frozen": {
                "gate_v1": "PROMPT_BUILDERS['v1'] (frozen before dev iteration)",
                "gate_structured": "StructuredGateEngine as of prompt_dev run6 "
                                   "(frozen after final dev iteration)",
            },
            "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSealed test results saved to {OUTPUT_PATH}")
    print(f"Total unique API calls: {llm.api_calls} "
          f"(of {llm.requests} LLM requests; rest were cache hits)")


if __name__ == "__main__":
    main()
