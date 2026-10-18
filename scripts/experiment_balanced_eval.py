#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Balanced adversarial held-out evaluation (EXPANSION-BALANCED).

Zero API calls: the frozen gate prompts are replayed from
``data/llm_gate_cache_balanced.json`` (a cache miss aborts).  Ground truth
is the committed human annotation in ``data/expansion_balanced_sheet.json``.

Usage:
    python scripts/experiment_balanced_eval.py
Writes data/experiment_balanced_results.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import random
import sys
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ntm_bp.baselines import BfsCascadeEngine
from ntm_bp.engine import NtmBpEngine
from ntm_bp.evaluation import compute_f1
from ntm_bp.experimental.gate_utils import build_memory
from ntm_bp.experimental.semantic_gate import (
    PROMPT_BUILDERS, SemanticGateEngine, StructuredGateEngine,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.path.join(DATA, "llm_gate_cache_balanced.json")
SHEET = os.path.join(DATA, "expansion_balanced_sheet.json")
OUT = os.path.join(DATA, "experiment_balanced_results.json")
N_BOOT, SEED = 2000, 42
METHODS = ["BFS", "NTM-BP", "gate-v1", "Structured", "kw_text_conf"]

_spec = importlib.util.spec_from_file_location(
    "kwmod", os.path.join(os.path.dirname(__file__), "experiment_keyword_baseline.py"))
kwmod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(kwmod)


class CacheOnlyLLM:
    def __init__(self, cache):
        self.cache = cache
        self.requests = 0

    def complete(self, prompt):
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key not in self.cache:
            raise RuntimeError("CACHE MISS: " + prompt[:80])
        self.requests += 1
        return self.cache[key]["response"]


def case_key(c):
    return c["case_id"].split("balanced::", 1)[-1]


def macro(v):
    return sum(v) / len(v)


def paired_bootstrap(a, b, cases, seed=SEED):
    rng = random.Random(seed)
    diffs = []
    for _ in range(N_BOOT):
        samp = [cases[rng.randrange(len(cases))] for _ in range(len(cases))]
        A = [v for c in samp for v in a[c]]
        Bv = [v for c in samp for v in b[c]]
        diffs.append(sum(A) / len(A) - sum(Bv) / len(Bv))
    obs = (sum(v for c in cases for v in a[c]) / sum(len(a[c]) for c in cases)
           - sum(v for c in cases for v in b[c]) / sum(len(b[c]) for c in cases))
    diffs.sort()
    ci = [diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT) - 1]]
    p = (2.0 * sum(1 for d in diffs if d <= 0) / N_BOOT if obs > 0
         else 2.0 * sum(1 for d in diffs if d >= 0) / N_BOOT)
    return obs, ci, min(1.0, p)


def main():
    cases = json.load(open(SHEET, encoding="utf-8"))["cases"]
    assert all(c["reeval"] is not None for c in cases), "labels incomplete"
    cache = json.load(open(CACHE, encoding="utf-8"))
    llm = CacheOnlyLLM(cache)

    per_inst = {m: [] for m in METHODS}
    by_case = {m: defaultdict(list) for m in METHODS}
    for c in cases:
        ck = case_key(c)
        gt = set(c["reeval"])
        fid = c["update_fact_id"]
        ob = c["facts"][fid][1]
        scen = {"facts": c["facts"], "dependencies": c["dependencies"]}
        mem = build_memory(scen); mem.update_fact(fid, c["new_value"], c["new_confidence"])
        m2 = build_memory(scen); m2.update_fact(fid, c["new_value"], c["new_confidence"])
        BfsCascadeEngine().cascade(m2, fid)
        preds = {"BFS": set(m2.get_stale_facts())}
        m3 = build_memory(scen); m3.update_fact(fid, c["new_value"], c["new_confidence"])
        NtmBpEngine(gamma=0.7, tau=0.7).propagate(m3, fid, source_old_belief=ob)
        preds["NTM-BP"] = set(m3.get_reeval_facts())
        m4 = build_memory(scen); m4.update_fact(fid, c["new_value"], c["new_confidence"])
        SemanticGateEngine(llm, tau_gate=0.5, prompt_builder=PROMPT_BUILDERS["v1"]).propagate(
            m4, fid, c["facts"][fid][0], c["new_value"], c["new_confidence"])
        preds["gate-v1"] = set(m4.get_reeval_facts())
        m5 = build_memory(scen); m5.update_fact(fid, c["new_value"], c["new_confidence"])
        StructuredGateEngine(llm).propagate(
            m5, fid, c["facts"][fid][0], c["new_value"], c["new_confidence"])
        preds["Structured"] = set(m5.get_reeval_facts())
        preds["kw_text_conf"] = kwmod.predict(
            {"facts": c["facts"], "dependencies": c["dependencies"],
             "update_fact_id": fid, "new_value": c["new_value"],
             "new_confidence": c["new_confidence"]}, "kw_text_conf")
        for m, pr in preds.items():
            by_case[m][ck].append(compute_f1(pr, gt))
            per_inst[m].append(compute_f1(pr, gt))

    keys = sorted(set(case_key(c) for c in cases))
    nonempty = [c for c in cases if c["reeval"]]
    empty = [c for c in cases if not c["reeval"]]
    ne_idx = [i for i, c in enumerate(cases) if c["reeval"]]
    em_idx = [i for i, c in enumerate(cases) if not c["reeval"]]
    ne = {m: macro([per_inst[m][i] for i in ne_idx]) for m in METHODS}
    em = {m: macro([per_inst[m][i] for i in em_idx]) for m in METHODS}
    macro_f1 = {m: macro(per_inst[m]) for m in METHODS}
    tests = {}
    for comp in ("NTM-BP", "gate-v1", "kw_text_conf"):
        obs, ci, p = paired_bootstrap(by_case["Structured"], by_case[comp], keys)
        tests["structured_minus_" + comp.replace("-", "_")] = {
            "observed_diff": obs, "ci95": ci, "p_value": p, "significant_at_0.05": p < 0.05}

    results = {
        "metadata": {"scope": "EXPANSION-BALANCED, 40 cases (20 non-empty/20 empty)",
                     "labels": "dual-annotator (author + blinded advisor), kappa=1.0",
                     "cache_replays": llm.requests, "new_api_calls": 0},
        "instance_level_macro_f1": macro_f1,
        "nonempty_f1": ne, "empty_f1": em,
        "paired_tests": tests,
    }
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("macro F1:", {k: round(v, 4) for k, v in macro_f1.items()})
    print("non-empty F1:", {k: round(v, 4) for k, v in ne.items()})
    print("empty F1:", {k: round(v, 4) for k, v in em.items()})
    for k, v in tests.items():
        print("  %-34s d=%+.4f p=%.4f%s" % (k, v["observed_diff"], v["p_value"],
                                            " *" if v["significant_at_0.05"] else ""))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
