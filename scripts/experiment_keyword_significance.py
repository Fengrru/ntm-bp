#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keyword-baseline significance audit on Expansion-500 (offline).

The paper reports the zero-cost keyword baseline only descriptively; this
script adds the paired cluster-bootstrap tests (full set and stratified
subsets) that the review flagged.  Zero API calls (frozen cache replay).

Usage:
    python scripts/experiment_keyword_significance.py
Writes data/experiment_keyword_significance_results.json.
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
OUT = os.path.join(DATA, "experiment_keyword_significance_results.json")
N_BOOT, SEED = 2000, 42
MARKED = [
    "恶化", "下滑", "下降", "缩减", "减少", "流失", "罢工", "中断", "缺陷",
    "亏损", "违约", "违规", "处罚", "罚款", "诉讼", "争议", "风险", "威胁",
    "失败", "未达", "不足", "延迟", "延期", "取消", "终止", "退出", "撤回",
    "超标", "异常", "波动", "不确定", "歧义", "矛盾", "冲突", "负面",
    "离职", "辞职", "解雇", "裁员", "关闭", "停产", "破产", "清算",
    "大幅", "严重", "重大", "急剧", "骤降", "暴跌", "崩盘", "危机",
]

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
            raise RuntimeError("CACHE MISS")
        self.requests += 1
        return self.cache[key]["response"]


def case_key(s):
    return s["template_id"] + "||" + s["new_value"]


def is_marked(v):
    return any(k in v for k in MARKED)


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
    cache = json.load(open(os.path.join(DATA, "llm_gate_cache.json"), encoding="utf-8"))
    llm = CacheOnlyLLM(cache)
    exp = json.load(open(os.path.join(DATA, "expansion500.json"), encoding="utf-8"))["scenarios"]

    by_case = defaultdict(lambda: defaultdict(list))
    idx = defaultdict(list)
    for i, s in enumerate(exp):
        ck = case_key(s)
        gt = set(s["ground_truth_reeval"])
        fid = s["update_fact_id"]
        ob = s["facts"][fid][1]
        preds = {}
        m1 = build_memory(s); m1.update_fact(fid, s["new_value"], s["new_confidence"])
        BfsCascadeEngine().cascade(m1, fid); preds["BFS"] = set(m1.get_stale_facts())
        m2 = build_memory(s); m2.update_fact(fid, s["new_value"], s["new_confidence"])
        NtmBpEngine(gamma=0.7, tau=0.7).propagate(m2, fid, source_old_belief=ob)
        preds["NTM-BP"] = set(m2.get_reeval_facts())
        m3 = build_memory(s); m3.update_fact(fid, s["new_value"], s["new_confidence"])
        SemanticGateEngine(llm, tau_gate=0.5, prompt_builder=PROMPT_BUILDERS["v1"]).propagate(
            m3, fid, s["facts"][fid][0], s["new_value"], s["new_confidence"])
        preds["gate-v1"] = set(m3.get_reeval_facts())
        m4 = build_memory(s); m4.update_fact(fid, s["new_value"], s["new_confidence"])
        StructuredGateEngine(llm).propagate(
            m4, fid, s["facts"][fid][0], s["new_value"], s["new_confidence"])
        preds["Structured"] = set(m4.get_reeval_facts())
        preds["kw_text_conf"] = kwmod.predict(s, "kw_text_conf")
        for m, pr in preds.items():
            by_case[m][ck].append(compute_f1(pr, gt))
        idx["all"].append(ck)
        idx["marked" if is_marked(s["new_value"]) else "unmarked"].append(ck)
        idx["nonempty" if s["ground_truth_reeval"] else "empty"].append(ck)

    tests = {}
    for tag in ("all", "marked", "unmarked", "nonempty", "empty"):
        cs = sorted(set(idx[tag]))
        for comp in ("Structured", "NTM-BP", "gate-v1"):
            obs, ci, p = paired_bootstrap(by_case["kw_text_conf"], by_case[comp], cs)
            tests[f"kw_text_conf_minus_{comp}[{tag}]"] = {
                "observed_diff": obs, "ci95": ci, "p_value": p,
                "significant_at_0.05": p < 0.05, "n_cases": len(cs)}

    results = {
        "metadata": {"scope": "Expansion-500", "resamples": N_BOOT, "seed": SEED,
                     "cache_replays": llm.requests, "new_api_calls": 0},
        "paired_tests": tests,
    }
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for k, v in tests.items():
        print("  %-46s d=%+.4f p=%.4f%s" % (k, v["observed_diff"], v["p_value"],
                                            " *" if v["significant_at_0.05"] else ""))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
