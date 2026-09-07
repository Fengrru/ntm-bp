# -*- coding: utf-8 -*-
"""Keyword-based gate baseline (zero LLM cost lower bound).

Motivation (paper review): add a rule-based keyword-matching baseline that
detects negation / negative-change cues between the old and new fact value,
costing zero LLM calls. It serves as the lower bound of the
cost--accuracy Pareto frontier.

Variants:
  kw_text        : trigger if a negative keyword appears in the NEW value
                   but not in the OLD value -> re-evaluate all descendants.
  kw_negation    : additionally handles negation flip ("无歧义" -> "存在歧义")
                   by checking whether the keyword was negated in the old value.
  kw_text_conf   : kw_text OR a confidence drop >= 0.30.
  bfs_subtree    : reference point: always re-evaluate all descendants.
  always_skip    : reference point: never re-evaluate.

Evaluation: EXPANSION-500 (540 labeled instances, human-annotated, kappa=1.0).
Cluster bootstrap CI over (template_id, new_value) cases.

No LLM calls are made. Usage:
    python scripts/experiment_keyword_baseline.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ntm_bp.evaluation import compute_f1  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_PATH = os.path.join(DATA_DIR, "expansion500.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_keyword_baseline_results.json")

NEG_WORDS = [
    # generic negation
    "无法", "不能", "未能", "未达", "没有", "不再",
    # negative change / adverse events
    "恶化", "下降", "下滑", "亏损", "推迟", "延期", "取消", "中断",
    "违约", "被调查", "违规", "风险", "问题", "不足", "离职", "退出",
    "超标", "延迟", "召回", "投诉", "歧义", "质疑", "怀疑", "低于",
    "故障", "缺陷", "流失", "收紧", "冻结", "搁置", "拒绝", "争议",
    "失败", "受挫", "受阻", "紧张", "短缺", "超标",
    # single-char negation kept last (substring order matters)
    "不", "未", "没",
]
NEGATION_PREFIXES = ("无", "不", "没", "非", "未")

N_BOOTSTRAP = 2000
RNG_SEED = 20260805


def _occurrences_negated(text: str, kw: str) -> tuple[bool, bool]:
    """Return (present, any_negated) for keyword kw in text."""
    present = False
    negated = False
    start = 0
    while True:
        idx = text.find(kw, start)
        if idx < 0:
            break
        present = True
        if idx > 0 and text[idx - 1] in NEGATION_PREFIXES:
            negated = True
        start = idx + 1
    return present, negated


def keyword_triggered(old_value: str, new_value: str, handle_flip: bool) -> bool:
    """True if a negative cue appears in new_value but not (validly) in old_value."""
    for kw in NEG_WORDS:
        new_present, new_negated = _occurrences_negated(new_value, kw)
        if not new_present or new_negated:
            continue
        old_present, old_negated = _occurrences_negated(old_value, kw)
        if not old_present:
            return True
        # keyword existed but was negated before, now asserted -> flip
        if handle_flip and old_negated:
            return True
    return False


def descendants(deps, source: str) -> set[str]:
    """All transitive children of source."""
    children: dict[str, list[str]] = {}
    for p, c in deps:
        children.setdefault(p, []).append(c)
    out: set[str] = set()
    stack = [source]
    while stack:
        node = stack.pop()
        for ch in children.get(node, []):
            if ch not in out:
                out.add(ch)
                stack.append(ch)
    return out


def predict(scenario: dict, variant: str) -> set[str]:
    old_value, old_conf = scenario["facts"][scenario["update_fact_id"]]
    new_value = scenario["new_value"]
    new_conf = scenario["new_confidence"]
    desc = descendants(scenario["dependencies"], scenario["update_fact_id"])

    if variant == "always_skip":
        return set()
    if variant == "bfs_subtree":
        return desc

    triggered = keyword_triggered(old_value, new_value, handle_flip=False)
    if variant == "kw_negation":
        triggered = keyword_triggered(old_value, new_value, handle_flip=True)
    elif variant == "kw_text_conf":
        triggered = triggered or (old_conf - new_conf) >= 0.30

    return desc if triggered else set()


def cluster_bootstrap_ci(
    f1_values: np.ndarray, cluster_ids: list, n_boot: int, rng: np.random.Generator
) -> list[float]:
    """Percentile CI by resampling clusters (cases) with replacement."""
    clusters: dict = {}
    for i, cid in enumerate(cluster_ids):
        clusters.setdefault(cid, []).append(i)
    keys = list(clusters.keys())
    n_clusters = len(keys)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(n_clusters, size=n_clusters, replace=True)
        idx = [i for k in chosen for i in clusters[keys[k]]]
        stats[b] = f1_values[idx].mean()
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def main() -> None:
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    scenarios = data["scenarios"]
    rng = np.random.default_rng(RNG_SEED)

    variants = ["kw_text", "kw_negation", "kw_text_conf", "bfs_subtree", "always_skip"]
    results: dict = {"per_variant": {}}

    # Precompute per-instance F1/RR per variant
    cluster_ids = [
        f"{s.get('template_id', s['scenario_id'])}||{s['new_value']}" for s in scenarios
    ]

    for variant in variants:
        f1s, rrs = [], []
        for s in scenarios:
            pred = predict(s, variant)
            gt = set(s["ground_truth_reeval"])
            f1 = compute_f1(pred, gt)
            denom = max(1, len(s["facts"]) - 1)  # exclude updated fact itself
            rr = len(pred) / denom
            f1s.append(f1)
            rrs.append(rr)
        f1_arr = np.array(f1s)
        ci = cluster_bootstrap_ci(f1_arr, cluster_ids, N_BOOTSTRAP, rng)
        results["per_variant"][variant] = {
            "avg_f1": float(f1_arr.mean()),
            "ci95": ci,
            "avg_reeval_rate": float(np.mean(rrs)),
            "llm_calls_per_update": 0.0,
            "n_instances": len(f1s),
        }

    results["metadata"] = {
        "dataset": "EXPANSION-500 (54 cases x 10 jitter instances, human labeled, kappa=1.0)",
        "variants": {
            "kw_text": "negative keyword present in new value but absent in old value",
            "kw_negation": "kw_text + negation-flip handling (e.g. '无歧义' -> '存在歧义')",
            "kw_text_conf": "kw_text OR confidence drop >= 0.30",
            "bfs_subtree": "reference: always re-evaluate all descendants",
            "always_skip": "reference: never re-evaluate",
        },
        "n_keyword_list": len(NEG_WORDS),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap": "cluster bootstrap resampling (template_id, new_value) cases",
        "note": "Zero-LLM-cost rule baseline; lower bound of the cost-accuracy frontier.",
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print(json.dumps(results["per_variant"], ensure_ascii=False, indent=1))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
