#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Significance, subset, and call-budget audit for the Expansion-500 claims.

Protocol (paper Appendix A, ``Bootstrap``):
  * paired cluster bootstrap at the case level (``template_id || new_value``);
  * resample cases with replacement and take *all* jitter instances within a
    drawn case, so instance-level F1 scores are nested within their case;
  * 2,000 resamples, seed 42; paired mean difference (structured minus
    competitor); two-sided p = 2 * min(P(diff* <= 0), P(diff* >= 0)).

Reports, in addition to the full-set paired tests:
  * a Holm--Bonferroni correction over the three structured-vs-* tests;
  * stratified tests on the 32 lexically unmarked cases (post-hoc);
  * empty / non-empty subset instance-level macro F1;
  * the per-case win/loss/tie breakdown.  Each case is represented by its
    *final jitter instance* for this case-level analysis only (the aggregate
    metrics everywhere else in the paper are instance-level macro means);
  * the audited Q1/Q2/Q3 call budget and the closed-form E[calls].

Zero API calls: every LLM prompt is replayed from
``data/llm_gate_cache.json`` and a cache miss aborts.

Usage:
    python scripts/experiment_significance.py
Writes ``data/experiment_significance_results.json``.
"""
from __future__ import annotations

import hashlib
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
    PROMPT_BUILDERS,
    SemanticGateEngine,
    StructuredGateEngine,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_PATH = os.path.join(DATA_DIR, "llm_gate_cache.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_significance_results.json")
N_BOOTSTRAP = 2000
SEED = 42
METHODS = ["BFS", "NTM-BP", "gate-v1", "Structured"]

# Exact 50-keyword lexical-marking list used for the paper's Table 4 split.
MARKED_KEYWORDS = [
    "恶化", "下滑", "下降", "缩减", "减少", "流失", "罢工", "中断", "缺陷",
    "亏损", "违约", "违规", "处罚", "罚款", "诉讼", "争议", "风险", "威胁",
    "失败", "未达", "不足", "延迟", "延期", "取消", "终止", "退出", "撤回",
    "超标", "异常", "波动", "不确定", "歧义", "矛盾", "冲突", "负面",
    "离职", "辞职", "解雇", "裁员", "关闭", "停产", "破产", "清算",
    "大幅", "严重", "重大", "急剧", "骤降", "暴跌", "崩盘", "危机",
]


class CacheOnlyLLM:
    """Deterministic replay from the frozen cache; never calls the network."""

    def __init__(self, cache):
        self.cache = cache
        self.requests = 0

    def complete(self, prompt):
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key not in self.cache:
            raise RuntimeError(
                "CACHE MISS: offline audit cannot continue; the frozen cache "
                "does not contain this prompt.")
        self.requests += 1
        return self.cache[key]["response"]


def case_key(s):
    return s["template_id"] + "||" + s["new_value"]


def is_marked(v):
    return any(k in v for k in MARKED_KEYWORDS)


def load(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def run_method(method, s, llm):
    """Return (f1, call_log) for one instance (call_log empty for numeric)."""
    gt = set(s["ground_truth_reeval"])
    fid = s["update_fact_id"]
    mem = build_memory(s)
    ob = mem.get_belief(fid)
    mem.update_fact(fid, s["new_value"], s["new_confidence"])
    log = []
    if method == "BFS":
        BfsCascadeEngine().cascade(mem, fid)
        pred = set(mem.get_stale_facts())
    else:
        if method == "NTM-BP":
            NtmBpEngine(gamma=0.7, tau=0.7).propagate(
                mem, fid, source_old_belief=ob)
        elif method == "gate-v1":
            eng = SemanticGateEngine(
                llm, tau_gate=0.5, prompt_builder=PROMPT_BUILDERS["v1"])
            eng.propagate(mem, fid, s["facts"][fid][0],
                          s["new_value"], s["new_confidence"])
            log = list(eng.call_log)
        else:  # Structured
            eng = StructuredGateEngine(llm)
            eng.propagate(mem, fid, s["facts"][fid][0],
                          s["new_value"], s["new_confidence"])
            log = list(eng.call_log)
        pred = set(mem.get_reeval_facts())
    return compute_f1(pred, gt), log


def macro(vals):
    return sum(vals) / len(vals)


def paired_bootstrap(a, b, cases, seed=SEED):
    """Paired cluster bootstrap of (a - b) over case clusters.

    ``a`` and ``b`` map case_key -> list of per-instance F1 scores.
    Returns (observed_delta, ci95, two_sided_p).
    """
    rng = random.Random(seed)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        samp = [cases[rng.randrange(len(cases))] for _ in range(len(cases))]
        A = [v for c in samp for v in a[c]]
        Bv = [v for c in samp for v in b[c]]
        diffs.append(sum(A) / len(A) - sum(Bv) / len(Bv))
    obs = (sum(v for c in cases for v in a[c])
           / sum(len(a[c]) for c in cases)
           - sum(v for c in cases for v in b[c])
           / sum(len(b[c]) for c in cases))
    diffs.sort()
    ci = [diffs[int(0.025 * N_BOOTSTRAP)],
          diffs[int(0.975 * N_BOOTSTRAP) - 1]]
    if obs > 0:
        p = 2.0 * sum(1 for d in diffs if d <= 0) / N_BOOTSTRAP
    else:
        p = 2.0 * sum(1 for d in diffs if d >= 0) / N_BOOTSTRAP
    return obs, ci, min(1.0, p)


def descendants(s):
    children = defaultdict(list)
    for p, c in s["dependencies"]:
        children[p].append(c)
    out, stack = set(), [s["update_fact_id"]]
    while stack:
        node = stack.pop()
        for ch in children.get(node, []):
            if ch not in out:
                out.add(ch)
                stack.append(ch)
    return out


def holm_bonferroni(pvals):
    """Holm--Bonferroni adjusted p-values for a list of (label, p)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i][1])
    m = len(pvals)
    out = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i][1])
        running = max(running, adj)
        out[i] = running
    return out


def main():
    cache = load("llm_gate_cache.json")
    llm = CacheOnlyLLM(cache)
    exp = load("expansion500.json")["scenarios"]
    n_inst = len(exp)

    per_inst = {m: [] for m in METHODS}
    by_case = {m: defaultdict(list) for m in METHODS}
    struct_total = 0
    gv_total = 0
    call_kinds = defaultdict(int)
    q1_by_case = {}
    class_counts = defaultdict(int)
    q2_total = q2_yes = 0

    for s in exp:
        ck = case_key(s)
        for m in METHODS:
            f1, log = run_method(m, s, llm)
            per_inst[m].append(f1)
            by_case[m][ck].append(f1)
            if m == "Structured":
                struct_total += len(log)
                if ck not in q1_by_case:
                    q1_by_case[ck] = next(
                        (c["raw_response"].strip().upper()[:1]
                         for c in log if c["kind"] == "change_type"), "?")
                    class_counts[q1_by_case[ck]] += 1
                    for c in log:
                        call_kinds[c["kind"]] += 1
                        if c["kind"] == "relevance":
                            q2_total += 1
                            if "否" not in c["raw_response"][:3]:
                                q2_yes += 1
            elif m == "gate-v1":
                gv_total += len(log)

    cases = sorted(by_case["BFS"].keys())
    n_cases = len(cases)
    overall = {m: macro(per_inst[m]) for m in METHODS}

    # ---- full-set paired tests vs structured ----
    full = {}
    labels = []
    for comp in ("BFS", "NTM-BP", "gate-v1"):
        obs, ci, p = paired_bootstrap(
            by_case["Structured"], by_case[comp], cases)
        key = "structured_minus_" + comp.replace("-", "_").lower()
        full[key] = {"observed_diff": obs, "ci95": ci, "p_value": p,
                     "significant_at_0.05": p < 0.05}
        labels.append((key, p))
    adj = holm_bonferroni(labels)
    full["holm_bonferroni"] = {labels[i][0]: adj[i] for i in range(len(labels))}

    # ---- stratified (unmarked) subset ----
    unmarked_cases = sorted(
        {c for c in cases if not is_marked(c.split("||", 1)[1])})
    subset = {}
    for comp in ("gate-v1", "NTM-BP"):
        obs, ci, p = paired_bootstrap(
            by_case["Structured"], by_case[comp], unmarked_cases)
        subset["unmarked_structured_minus_"
               + comp.replace("-", "_").lower()] = {
            "observed_diff": obs, "ci95": ci, "p_value": p,
            "significant_at_0.05": p < 0.05, "n_cases": len(unmarked_cases)}

    # ---- empty / non-empty instance subsets ----
    def subset_f1(m, nonempty):
        return macro([v for s, v in zip(exp, per_inst[m])
                      if bool(s["ground_truth_reeval"]) == nonempty])
    subset_means = {
        "nonempty": {m: subset_f1(m, True) for m in METHODS},
        "empty": {m: subset_f1(m, False) for m in METHODS},
        "n_nonempty_instances": sum(1 for s in exp if s["ground_truth_reeval"]),
        "n_empty_instances": sum(1 for s in exp if not s["ground_truth_reeval"]),
    }

    # ---- per-case win/loss/tie (final jitter instance as representative) ----
    last = {m: {c: by_case[m][c][-1] for c in cases} for m in METHODS}
    w = l = t = 0
    sub_by_q1 = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
    for c in cases:
        a, b = last["Structured"][c], last["NTM-BP"][c]
        cls = q1_by_case.get(c, "?")
        if a > b:
            w += 1
            sub_by_q1[cls]["wins"] += 1
        elif a < b:
            l += 1
            sub_by_q1[cls]["losses"] += 1
        else:
            t += 1
            sub_by_q1[cls]["ties"] += 1

    # ---- audited call budget ----
    freq = {k: v / n_cases for k, v in class_counts.items()}
    avg_dk = sum(len(descendants(s)) for s in exp) / n_inst
    p_q2_yes = (q2_yes / q2_total) if q2_total else float("nan")
    e_calls = (1.0 + avg_dk * (1.0 - freq.get("C", 0.0))
               + avg_dk * freq.get("B", 0.0) * p_q2_yes)
    call_budget = {
        "avg_downstream_conclusions": avg_dk,
        "P_Q1_C": freq.get("C", 0.0),
        "P_Q1_B": freq.get("B", 0.0),
        "P_Q1_A": freq.get("A", 0.0),
        "P_Q2_yes": p_q2_yes,
        "expected_calls_formula": e_calls,
        "audited_calls_per_update_structured": struct_total / n_inst,
        "audited_calls_per_update_gate_v1": gv_total / n_inst,
        "q1_class_counts": dict(class_counts),
        "structured_call_kinds_per_case": dict(call_kinds),
    }

    results = {
        "metadata": {
            "scope": "Expansion-500, 54 cases x 10 jitter instances = 540",
            "protocol": ("paired cluster bootstrap, case-level clusters, all "
                         "instances within a drawn case, 2000 resamples, seed 42, "
                         "two-sided p = 2*min(P(diff<=0), P(diff>=0))"),
            "per_case_representative": ("final jitter instance (case-level W/L/T "
                                        "only; aggregates are instance-level macro)"),
            "new_api_calls": 0,
            "cache_replays": llm.requests,
        },
        "instance_level_macro_f1": overall,
        "full_set_paired_tests": full,
        "unmarked_subset_paired_tests": subset,
        "subset_means": subset_means,
        "per_case_win_loss_tie_structured_vs_ntm_bp": {
            "win": w, "loss": l, "tie": t,
            "by_q1_class": {k: v for k, v in sub_by_q1.items()},
        },
        "call_budget": call_budget,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("SIGNIFICANCE AUDIT (offline, cache replays={}, api_calls=0)".format(
        llm.requests))
    print("  instance-level macro F1:",
          {m: round(overall[m], 4) for m in METHODS})
    for key, val in full.items():
        if key == "holm_bonferroni":
            continue
        print("  {:<32} d={:.4f} p={:.4f}{}".format(
            key, val["observed_diff"], val["p_value"],
            " *" if val["significant_at_0.05"] else ""))
    print("  Holm-Bonferroni:",
          {k: round(v, 4) for k, v in full["holm_bonferroni"].items()})
    for key, val in subset.items():
        print("  {:<40} d={:.4f} p={:.4f}{}".format(
            key, val["observed_diff"], val["p_value"],
            " *" if val["significant_at_0.05"] else ""))
    print("  per-case W/L/T (structured vs NTM-BP): {} / {} / {}".format(w, l, t))
    print("  Q1 classes:", dict(class_counts))
    print("  calls/update: structured={:.3f} gate-v1={:.3f} |Dk|={:.3f} "
          "E[calls]={:.3f}".format(
              call_budget["audited_calls_per_update_structured"],
              call_budget["audited_calls_per_update_gate_v1"],
              avg_dk, e_calls))
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
