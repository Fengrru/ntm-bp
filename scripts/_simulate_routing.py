"""ROUTING V0 SIMULATION (post-hoc, NOT ex-ante).

Replays the FROZEN components (ntm_bp + gate_structured) from the LLM
cache against the 540 expansion instances and synthesizes the routing v0
prediction defined in docs/stage3_smart_routing.md:

    cls = Q1(old, new)            # 1 LLM call, from structured call_log
    cls == C -> no reeval
    cls == B -> numeric path: R = ntm_bp propagations
    cls == A -> LLM path: R = structured relevance set

HONESTY NOTE: this is a POST-HOC feasibility analysis on a test set the
routing rule was designed after. It is NOT ex-ante evidence and must NOT
be reported as the routing result in the paper. The official routing
evaluation requires a third data batch (see docs/stage3_smart_routing.md).

Zero API calls: every prompt is replayed from llm_gate_cache.json via
CacheOnlyLLM.
"""

import hashlib
import io
import json
import os
import random
import sys
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ntm_bp.engine import NtmBpEngine
from ntm_bp.experimental.semantic_gate import StructuredGateEngine
from ntm_bp.evaluation import compute_f1

from ntm_bp.experimental.gate_utils import build_memory

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_PATH = os.path.join(DATA_DIR, "expansion500.json")
CACHE_PATH = os.path.join(DATA_DIR, "llm_gate_cache.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "routing_v0_simulation.json")

N_BOOTSTRAP = 2000


class CacheOnlyLLM:
    def __init__(self, cache):
        self.cache = cache
        self.requests = 0

    def complete(self, prompt):
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key not in self.cache:
            raise RuntimeError("cache miss")
        self.requests += 1
        return self.cache[key]["response"]


def case_key(s: dict) -> str:
    return f"{s['template_id']}||{s['new_value']}"


def main() -> None:
    cache = json.load(open(CACHE_PATH, encoding="utf-8"))
    llm = CacheOnlyLLM(cache)
    scenarios = json.load(open(DATA_PATH, encoding="utf-8"))["scenarios"]

    records = []          # (case_key, routing_f1, ntm_f1, struct_f1, cls)
    by_case = defaultdict(lambda: {"routing": [], "ntm_bp": [],
                                   "structured": [], "cls": []})
    for s in scenarios:
        gt = set(s["ground_truth_reeval"])
        fid = s["update_fact_id"]
        ck = case_key(s)

        # numeric path
        mem = build_memory(s)
        ob = mem.get_belief(fid)
        mem.update_fact(fid, s["new_value"], s["new_confidence"])
        NtmBpEngine(gamma=0.7, tau=0.7).propagate(
            mem, fid, source_old_belief=ob)
        pred_numeric = set(mem.get_reeval_facts())

        # LLM path + Q1 classification (cache replay)
        mem = build_memory(s)
        mem.update_fact(fid, s["new_value"], s["new_confidence"])
        eng = StructuredGateEngine(llm)
        eng.propagate(mem, fid, s["facts"][fid][0],
                      s["new_value"], s["new_confidence"])
        pred_struct = set(mem.get_reeval_facts())
        cls = next((c["raw_response"].strip().upper()[:1]
                    for c in eng.call_log
                    if c["kind"] == "change_type"), "?")

        if cls == "C":
            pred = set()
        elif cls == "B":
            pred = pred_numeric
        else:  # "A" or unknown -> LLM path
            pred = pred_struct

        records.append((ck, compute_f1(pred, gt),
                        compute_f1(pred_numeric, gt),
                        compute_f1(pred_struct, gt), cls))
        by_case[ck]["routing"].append(compute_f1(pred, gt))
        by_case[ck]["ntm_bp"].append(compute_f1(pred_numeric, gt))
        by_case[ck]["structured"].append(compute_f1(pred_struct, gt))
        by_case[ck]["cls"].append(cls)

    n = len(records)
    names = ["routing", "ntm_bp", "structured"]
    label = {"routing": "routing_v0", "ntm_bp": "ntm_bp",
             "structured": "structured"}
    idx = {"routing": 1, "ntm_bp": 2, "structured": 3}
    point = {m: sum(r[idx[m]] for r in records) / n for m in names}
    cases = sorted(by_case.keys())

    rng = random.Random(42)
    ci = {}
    for m in names:
        means = []
        for _ in range(N_BOOTSTRAP):
            sampled = [cases[rng.randrange(len(cases))]
                       for _ in range(len(cases))]
            vals = [v for ck in sampled for v in by_case[ck][m]]
            means.append(sum(vals) / len(vals))
        means.sort()
        ci[m] = [means[int(0.025 * N_BOOTSTRAP)],
                 means[int(0.975 * N_BOOTSTRAP) - 1]]

    print("ROUTING V0 SIMULATION (post-hoc, NOT ex-ante)")
    print(f"{len(scenarios)} instances / {len(cases)} cases, "
          f"cache replays={llm.requests}, new_api_calls=0")
    for m in names:
        print(f"  {label[m]:<12} F1={point[m]:.4f}  "
              f"95%CI=[{ci[m][0]:.4f}, {ci[m][1]:.4f}]")

    # win/loss vs structured (case level, majority of instances)
    wins = losses = ties = 0
    cls_counts = defaultdict(int)
    for ck in cases:
        v = sum(by_case[ck]["routing"]) / len(by_case[ck]["routing"])
        w = sum(by_case[ck]["structured"]) / len(by_case[ck]["structured"])
        if v > w:
            wins += 1
        elif v < w:
            losses += 1
        else:
            ties += 1
        cls_counts[by_case[ck]["cls"][0]] += 1
    print(f"vs structured: win {wins} / loss {losses} / tie {ties}")
    print(f"Q1 class distribution: {dict(cls_counts)}")

    output = {
        "metadata": {
            "scope": "POST-HOC feasibility analysis, NOT ex-ante. "
                     "Routing rule frozen in docs/stage3_smart_routing.md "
                     "but designed after these results; official evaluation "
                     "requires third data batch.",
            "components_frozen": True,
            "new_api_calls": 0,
            "cache_replays": llm.requests,
            "bootstrap": "cluster bootstrap by case, seed=42",
            "run_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {m: {"avg_f1": point[m], "ci95": ci[m]}
                    for m in names},
        "per_case": {ck: {"routing": sum(by_case[ck]["routing"]) /
                          len(by_case[ck]["routing"]),
                          "ntm_bp": sum(by_case[ck]["ntm_bp"]) /
                          len(by_case[ck]["ntm_bp"]),
                          "structured": sum(by_case[ck]["structured"]) /
                          len(by_case[ck]["structured"]),
                          "q1_class": by_case[ck]["cls"][0]}
                     for ck in cases},
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
