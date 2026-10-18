#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured-gate component ablation on SynDep (offline, frozen cache).

Recomputes the full Q1/Q2/Q3 structured gate and two ablated variants on the
SynDep development and sealed-test splits, replaying every LLM prompt from
``data/llm_gate_cache.json`` (zero API calls; a cache miss aborts).

Variants
--------
* ``Full``       : ``Q2=yes AND (Q1=A OR (Q1=B AND Q3=opposed))`` -- the
                   shipped :class:`StructuredGateEngine`.
* ``NoQ3``       : ``Q2=yes AND Q1=A`` (B always skips; Q3 is never asked).
* ``ScalarQ2``   : the binary relevance judgment is replaced by the per-target
                   v1 scalar impact prompt thresholded at ``tau_gate``; the
                   Q1/Q3 logic is kept.

The ``Q1 early-exit removed`` variant (C also proceeds) is *not* reproduceable
offline: under the frozen gate no Q2 prompt is issued for type-C cases, so
those prompts are absent from the cache and would require an API call.  It is
therefore omitted here and noted in REPRODUCE.md.

Usage:
    python scripts/experiment_gate_ablation.py
Writes ``data/experiment_gate_ablation_results.json``.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ntm_bp.evaluation import compute_f1
from ntm_bp.experimental.gate_utils import build_memory
from ntm_bp.experimental.semantic_gate import (
    PROMPT_BUILDERS,
    StructuredGateEngine,
    SemanticGateEngine,
    build_prompt_v1,
    collect_descendant_paths,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_PATH = os.path.join(DATA_DIR, "llm_gate_cache.json")
SPLIT_PATH = os.path.join(DATA_DIR, "case_split.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_gate_ablation_results.json")
TAU_GATE = 0.5


class CacheOnlyLLM:
    def __init__(self, cache):
        self.cache = cache
        self.hits = 0

    def complete(self, prompt):
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key not in self.cache:
            raise RuntimeError(
                "CACHE MISS during offline ablation: this variant issues a "
                "prompt absent from the frozen cache and cannot be replayed "
                "without an API call.")
        self.hits += 1
        return self.cache[key]["response"]


class NoQ3Engine(StructuredGateEngine):
    """Q1=B always skips; Q3 is never consulted."""

    def propagate(self, memory, source_fact_id, old_value, new_value,
                  new_confidence=None):
        change = self._classify_change(old_value, new_value)
        for target_id, _path in collect_descendant_paths(
                memory.graph, source_fact_id).items():
            if change == "C":
                continue
            fact = memory.get_fact(target_id)
            target_text = fact.value if fact else target_id
            if not self._judge_relevance(old_value, new_value, target_text):
                continue
            if change == "A":
                memory.mark_reeval(target_id)
        return memory


class ScalarQ2Engine(StructuredGateEngine):
    """Replace the binary Q2 relevance judgment with the v1 scalar impact."""

    def propagate(self, memory, source_fact_id, old_value, new_value,
                  new_confidence=None):
        change = self._classify_change(old_value, new_value)
        for target_id, path in collect_descendant_paths(
                memory.graph, source_fact_id).items():
            if change == "C":
                continue
            fact = memory.get_fact(target_id)
            target_text = fact.value if fact else target_id
            prompt = build_prompt_v1(
                memory, path, target_id, old_value, new_value, new_confidence)
            response = self._ask("scalar_q2", prompt)
            if SemanticGateEngine._parse_float(response) < TAU_GATE:
                continue
            if change == "A":
                memory.mark_reeval(target_id)
            elif self._judge_stance(new_value, target_text) == "opposed":
                memory.mark_reeval(target_id)
        return memory


class GateV1Engine(SemanticGateEngine):
    """Frozen scalar-prompt baseline (gate-v1)."""

    def __init__(self, llm):
        super().__init__(llm, tau_gate=TAU_GATE,
                         prompt_builder=PROMPT_BUILDERS["v1"])


VARIANTS = {
    "Full": StructuredGateEngine,
    "NoQ3": NoQ3Engine,
    "ScalarQ2": ScalarQ2Engine,
    "gate-v1": GateV1Engine,
}


def load(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def run_variant(engine_cls, scenarios, cache):
    llm = CacheOnlyLLM(cache)
    f1s = []
    for s in scenarios:
        gt = set(s["ground_truth_reeval"])
        fid = s["update_fact_id"]
        mem = build_memory(s)
        mem.update_fact(fid, s["new_value"], s["new_confidence"])
        engine_cls(llm).propagate(
            mem, fid, s["facts"][fid][0], s["new_value"], s["new_confidence"])
        f1s.append(compute_f1(set(mem.get_reeval_facts()), gt))
    return sum(f1s) / len(f1s), llm.hits


def main():
    cache = load("llm_gate_cache.json")
    split = load("case_split.json")["datasets"]["syndep"]
    all_scen = load("syndep.json")["scenarios"]
    results = {}
    for split_name, key in (("dev", "dev_cases"), ("test", "test_cases")):
        cases = set(split[key])
        scen = [s for s in all_scen
                if (s["template_id"] + "||" + s["new_value"]) in cases]
        results[split_name] = {}
        for name, cls in VARIANTS.items():
            f1, hits = run_variant(cls, scen, cache)
            results[split_name][name] = {
                "f1": f1, "cache_replays": hits,
                "n_cases": len(cases), "n_instances": len(scen)}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "dataset": "SynDep (dev = 12 cases, sealed test = 18 cases)",
                "tau_gate": TAU_GATE,
                "new_api_calls": 0,
                "note": "Q1-early-exit-removed variant omitted: its type-C Q2 "
                        "prompts are not in the frozen cache.",
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print("Structured-gate component ablation (offline, frozen cache)")
    print("| Variant | dev F1 | test F1 |")
    print("|---|---|---|")
    for name in VARIANTS:
        print("| {} | {:.4f} | {:.4f} |".format(
            name, results["dev"][name]["f1"], results["test"][name]["f1"]))
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
