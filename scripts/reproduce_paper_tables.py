#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OFFLINE reproduction of the paper's LLM-derived tables (ZERO API calls).

Replays every frozen LLM prompt from the committed prompt-hash cache
``data/llm_gate_cache.json`` against the committed datasets, recomputes
the paper's Tables 2-4 and 10-12 at the instance level, and asserts each
number against the value reported in the paper.

A cache miss aborts immediately (no network fallback), so a PASS is proof
that the committed cache + datasets + engine reproduce the paper.

Usage:
    python scripts/reproduce_paper_tables.py
Exit code 0 iff every checked number matches within TOL.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
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
SPLIT_PATH = os.path.join(DATA_DIR, "case_split.json")
TAU_GATE = 0.5
TOL = 0.002

# Exact 50-keyword lexical-marking list used for the paper's Table 4 split.
MARKED_KEYWORDS = [
    "恶化", "下滑", "下降", "缩减", "减少", "流失", "罢工", "中断", "缺陷",
    "亏损", "违约", "违规", "处罚", "罚款", "诉讼", "争议", "风险", "威胁",
    "失败", "未达", "不足", "延迟", "延期", "取消", "终止", "退出", "撤回",
    "超标", "异常", "波动", "不确定", "歧义", "矛盾", "冲突", "负面",
    "离职", "辞职", "解雇", "裁员", "关闭", "停产", "破产", "清算",
    "大幅", "严重", "重大", "急剧", "骤降", "暴跌", "崩盘", "危机",
]

NEG_WORDS = [
    "无法", "不能", "未能", "未达", "没有", "不再",
    "恶化", "下降", "下滑", "亏损", "推迟", "延期", "取消", "中断",
    "违约", "被调查", "违规", "风险", "问题", "不足", "离职", "退出",
    "超标", "延迟", "召回", "投诉", "歧义", "质疑", "怀疑", "低于",
    "故障", "缺陷", "流失", "收紧", "冻结", "搁置", "拒绝", "争议",
    "失败", "受挫", "受阻", "紧张", "短缺", "超标",
    "不", "未", "没",
]
NEGATION_PREFIXES = ("无", "不", "没", "非", "未")

# Paper-reported values (instance-level macro F1 / accuracy).
EXPECTED = {
    # Table 2: sealed test
    "sealed/pc500/BFS": 0.854,
    "sealed/pc500/NTM-BP": 0.851,
    "sealed/pc500/gate-v1": 0.854,
    "sealed/pc500/Structured": 0.854,
    "sealed/syndep/BFS": 0.626,
    "sealed/syndep/NTM-BP": 0.703,
    "sealed/syndep/gate-v1": 0.692,
    "sealed/syndep/Structured": 0.807,
    # Table 3: Expansion-500 overall
    "expansion/overall/BFS": 0.315,
    "expansion/overall/NTM-BP": 0.686,
    "expansion/overall/gate-v1": 0.667,
    "expansion/overall/Structured": 0.778,
    # Table 4: Expansion-500 stratified (22 marked / 32 unmarked cases)
    "expansion/marked/BFS": 0.727,
    "expansion/marked/NTM-BP": 0.771,
    "expansion/marked/gate-v1": 0.955,
    "expansion/marked/Structured": 0.909,
    "expansion/unmarked/BFS": 0.031,
    "expansion/unmarked/NTM-BP": 0.628,
    "expansion/unmarked/gate-v1": 0.469,
    "expansion/unmarked/Structured": 0.688,
    # Table 10: zero-cost keyword baselines on Expansion-500
    "kw/always_skip": 0.685,
    "kw/kw_text": 0.833,
    "kw/kw_negation": 0.852,
    "kw/kw_text_conf": 0.944,
    "kw/bfs_subtree": 0.315,
    # Table 11: keyword + confidence on the sealed splits
    "sealed_kw/pc500/kw_text_conf": 0.963,
    "sealed_kw/syndep/kw_text_conf": 0.851,
    # Table 12: routing v0 (post-hoc)
    "routing/routing_v0": 0.750,
    "routing/NTM-BP": 0.686,
    "routing/Structured": 0.778,
}


class CacheOnlyLLM:
    """Deterministic replay from the frozen prompt-hash cache; never calls out."""

    def __init__(self, cache):
        self.cache = cache
        self.requests = 0

    def complete(self, prompt):
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key not in self.cache:
            raise RuntimeError(
                "CACHE MISS: the committed prompt-hash cache does not contain "
                "this prompt; offline reproduction cannot continue.")
        self.requests += 1
        return self.cache[key]["response"]


def case_key(s):
    return "{}||{}".format(s["template_id"], s["new_value"])


def load(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def is_marked(v):
    return any(k in v for k in MARKED_KEYWORDS)


def _occ(text, kw):
    present = negated = False
    start = 0
    while True:
        i = text.find(kw, start)
        if i < 0:
            break
        present = True
        if i > 0 and text[i - 1] in NEGATION_PREFIXES:
            negated = True
        start = i + 1
    return present, negated


def kw_triggered(old, new, handle_flip):
    for kw in NEG_WORDS:
        new_p, new_n = _occ(new, kw)
        if not new_p or new_n:
            continue
        old_p, old_n = _occ(old, kw)
        if not old_p:
            return True
        if handle_flip and old_n:
            return True
    return False


def descendants(deps, source):
    children = defaultdict(list)
    for p, c in deps:
        children[p].append(c)
    out, stack = set(), [source]
    while stack:
        node = stack.pop()
        for ch in children.get(node, []):
            if ch not in out:
                out.add(ch)
                stack.append(ch)
    return out


def run_method(method, s, llm):
    gt = set(s["ground_truth_reeval"])
    fid = s["update_fact_id"]
    mem = build_memory(s)
    ob = mem.get_belief(fid)
    mem.update_fact(fid, s["new_value"], s["new_confidence"])
    if method == "BFS":
        BfsCascadeEngine().cascade(mem, fid)
        pred = set(mem.get_stale_facts())
    else:
        if method == "NTM-BP":
            NtmBpEngine(gamma=0.7, tau=0.7).propagate(mem, fid, source_old_belief=ob)
        elif method == "gate-v1":
            SemanticGateEngine(
                llm, tau_gate=TAU_GATE,
                prompt_builder=PROMPT_BUILDERS["v1"],
            ).propagate(mem, fid, s["facts"][fid][0], s["new_value"], s["new_confidence"])
        elif method == "Structured":
            StructuredGateEngine(llm).propagate(
                mem, fid, s["facts"][fid][0], s["new_value"], s["new_confidence"])
        else:
            raise ValueError(method)
        pred = set(mem.get_reeval_facts())
    return compute_f1(pred, gt), pred, gt


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    cache = load("llm_gate_cache.json")
    llm = CacheOnlyLLM(cache)
    results = {}

    # ---- Table 2: sealed test (from case_split.json) ----
    split = load("case_split.json")
    for name in ("pc500", "syndep"):
        test_cases = set(split["datasets"][name]["test_cases"])
        scenarios = [s for s in load(name + ".json")["scenarios"]
                     if case_key(s) in test_cases]
        for method in ("BFS", "NTM-BP", "gate-v1", "Structured"):
            f1s = [run_method(method, s, llm)[0] for s in scenarios]
            results["sealed/{}/{}".format(name, method)] = mean(f1s)

    # ---- Tables 3-4, 10-12: Expansion-500 ----
    exp = load("expansion500.json")["scenarios"]
    exp_f1 = {}
    for method in ("BFS", "NTM-BP", "gate-v1", "Structured"):
        f1s = [run_method(method, s, llm)[0] for s in exp]
        exp_f1[method] = f1s
        results["expansion/overall/{}".format(method)] = mean(f1s)
        marked = [f for s, f in zip(exp, f1s) if is_marked(s["new_value"])]
        unmarked = [f for s, f in zip(exp, f1s) if not is_marked(s["new_value"])]
        results["expansion/marked/{}".format(method)] = mean(marked)
        results["expansion/unmarked/{}".format(method)] = mean(unmarked)

    VARIANTS = ("always_skip", "kw_text", "kw_negation", "kw_text_conf", "bfs_subtree")
    kw_f1 = {v: [] for v in VARIANTS}
    for s in exp:
        fid = s["update_fact_id"]
        old, oldc = s["facts"][fid]
        desc = descendants(s["dependencies"], fid)
        gt = set(s["ground_truth_reeval"])
        trig = kw_triggered(old, s["new_value"], False)
        trig_n = kw_triggered(old, s["new_value"], True)
        trig_c = trig or (oldc - s["new_confidence"]) >= 0.30
        preds = {
            "always_skip": set(),
            "bfs_subtree": desc,
            "kw_text": desc if trig else set(),
            "kw_negation": desc if trig_n else set(),
            "kw_text_conf": desc if trig_c else set(),
        }
        for v in VARIANTS:
            kw_f1[v].append(compute_f1(preds[v], gt))
    for v in VARIANTS:
        results["kw/{}".format(v)] = mean(kw_f1[v])

    # Table 11: keyword+confidence on the sealed splits
    for name in ("pc500", "syndep"):
        test_cases = set(split["datasets"][name]["test_cases"])
        scenarios = [s for s in load(name + ".json")["scenarios"]
                     if case_key(s) in test_cases]
        vals = []
        for s in scenarios:
            fid = s["update_fact_id"]
            old, oldc = s["facts"][fid]
            trig = kw_triggered(old, s["new_value"], False) or \
                (oldc - s["new_confidence"]) >= 0.30
            pred = set(descendants(s["dependencies"], fid)) if trig else set()
            vals.append(compute_f1(pred, set(s["ground_truth_reeval"])))
        results["sealed_kw/{}/kw_text_conf".format(name)] = mean(vals)

    # Table 12: routing v0 (post-hoc replay of frozen Q1 + numeric/LLM paths)
    routing, numeric, structured = [], [], []
    for s in exp:
        fid = s["update_fact_id"]
        mem = build_memory(s)
        ob = mem.get_belief(fid)
        mem.update_fact(fid, s["new_value"], s["new_confidence"])
        NtmBpEngine(gamma=0.7, tau=0.7).propagate(mem, fid, source_old_belief=ob)
        pred_num = set(mem.get_reeval_facts())
        mem2 = build_memory(s)
        mem2.update_fact(fid, s["new_value"], s["new_confidence"])
        eng = StructuredGateEngine(llm)
        eng.propagate(mem2, fid, s["facts"][fid][0], s["new_value"], s["new_confidence"])
        pred_str = set(mem2.get_reeval_facts())
        cls = next((c["raw_response"].strip().upper()[:1]
                    for c in eng.call_log if c["kind"] == "change_type"), "?")
        pred = set() if cls == "C" else (pred_num if cls == "B" else pred_str)
        gt = set(s["ground_truth_reeval"])
        routing.append(compute_f1(pred, gt))
        numeric.append(compute_f1(pred_num, gt))
        structured.append(compute_f1(pred_str, gt))
    results["routing/routing_v0"] = mean(routing)
    results["routing/NTM-BP"] = mean(numeric)
    results["routing/Structured"] = mean(structured)

    # ---- report ----
    print("OFFLINE REPRODUCTION FROM FROZEN CACHE "
          "(cache replays={}, new_api_calls=0)".format(llm.requests))
    print("{:<34} {:>9} {:>9}  {}".format("key", "expected", "got", ""))
    failures = []
    for key in EXPECTED:
        exp_v, got = EXPECTED[key], results.get(key)
        ok = got is not None and abs(got - exp_v) <= TOL
        if not ok:
            failures.append((key, exp_v, got))
        print("{:<34} {:>9.3f} {:>9.3f}  {}".format(
            key, exp_v, got if got is not None else float("nan"),
            "OK" if ok else "MISMATCH"))
    print("-" * 60)
    if failures:
        print("FAIL: {} number(s) mismatch".format(len(failures)))
        return 1
    print("PASS: all {} numbers reproduce from the committed cache.".format(len(EXPECTED)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
