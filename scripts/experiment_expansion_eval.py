"""EXPANSION EVAL: frozen methods on the never-seen expansion-500 test.

DESIGN
------
* expansion500.json contains 54 human-annotated cases (2 annotators,
  kappa=1.0) x 10 jitter instances = 540 instances. These cases were
  generated and annotated AFTER all prompt iteration finished; no engine
  has ever seen them. They are used in full as a held-out test -- an
  ex-ante clean evaluation of the frozen methods:
    - bfs            (binary cascade baseline)
    - ntm_bp         (numeric reference)
    - gate_v1        (frozen original scalar prompt)
    - gate_structured(frozen structured gate, selected method)
* Cluster bootstrap by (template_id, new_value) case, seed fixed.
* Refuses to run if the output file exists (one-shot).
* Reuses llm_gate_cache.json; identical prompts across instances of a
  case are cache hits, so API calls stay at case level.

Usage:
    python scripts/experiment_expansion_eval.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ntm_bp.baselines import BfsCascadeEngine
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
DATA_PATH = os.path.join(DATA_DIR, "expansion500.json")
CACHE_PATH = os.path.join(DATA_DIR, "llm_gate_cache.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_expansion_eval_results.json")

TAU_GATE = 0.5
N_BOOTSTRAP = 2000
METHODS = ["bfs", "ntm_bp", "gate_v1", "gate_structured"]


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
        if method == "bfs":
            BfsCascadeEngine().cascade(mem, fid)
            pred = set(mem.get_stale_facts())
            records.append((case_key(s), compute_f1(pred, gt), pred, gt))
            continue
        elif method == "ntm_bp":
            NtmBpEngine(gamma=0.7, tau=0.7).propagate(
                mem, fid, source_old_belief=ob)
        elif method == "gate_structured":
            StructuredGateEngine(llm).propagate(
                mem, fid, s["facts"][fid][0],
                s["new_value"], s["new_confidence"])
        else:  # gate_v1
            SemanticGateEngine(
                llm, tau_gate=TAU_GATE,
                prompt_builder=PROMPT_BUILDERS["v1"],
            ).propagate(mem, fid, s["facts"][fid][0],
                        s["new_value"], s["new_confidence"])
        pred = set(mem.get_reeval_facts())
        records.append((case_key(s), compute_f1(pred, gt), pred, gt))
    return records


def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY is not set.")
        sys.exit(1)
    if os.path.isfile(OUTPUT_PATH):
        print(f"REFUSING to run: {OUTPUT_PATH} already exists.")
        sys.exit(1)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)["scenarios"]
    n_cases = len({case_key(s) for s in scenarios})
    print(f"expansion500: {len(scenarios)} instances, {n_cases} cases "
          f"(never seen by any engine)")

    cache = {}
    if os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
    print(f"Loaded {len(cache)} cached LLM responses")
    llm = CachedLLM(DeepSeekClient(), cache)

    f1_records = {m: [] for m in METHODS}
    t0 = time.time()
    for m in METHODS:
        calls_before = llm.api_calls
        f1_records[m] = run_method(scenarios, m, llm)
        print(f"  {m:<15} done, new_api_calls={llm.api_calls - calls_before}")
    print(f"  methods done in {time.time() - t0:.1f}s")

    # Cluster bootstrap by case
    rng = random.Random(42)
    by_case = {m: defaultdict(list) for m in METHODS}
    for m in METHODS:
        for ck, f1, _, _ in f1_records[m]:
            by_case[m][ck].append(f1)
    cases = sorted(by_case["bfs"].keys())
    n = len(scenarios)

    summary = {}
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
        summary[m] = {"avg_f1": point, "ci95": ci}
        print(f"  {m:<15} F1={point:.4f}  "
              f"95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

    per_case = {m: {ck: f1 for ck, f1, _, _ in f1_records[m]}
                for m in METHODS}
    output = {
        "metadata": {
            "scope": "EXPANSION-500: 54 never-seen cases x 10 instances",
            "methods": METHODS,
            "tau_gate": TAU_GATE,
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap": "cluster bootstrap resampling (template_id, new_value) cases",
            "frozen": "gate_v1 + gate_structured frozen since sealed test; "
                      "no prompt touched expansion data before this run",
            "llm_model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": {"summary": summary, "per_case": per_case},
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Total unique API calls: {llm.api_calls} "
          f"(of {llm.requests} LLM requests; rest were cache hits)")


if __name__ == "__main__":
    main()
