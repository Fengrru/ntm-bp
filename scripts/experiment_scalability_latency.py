#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-update wall-clock latency vs. graph size (paper Table 7 / Fig. 11-12).

Generates the deterministic scalability graphs (seed 42, 20 graphs per size
for sizes 10..1000) and times one belief-propagation update per graph for
BFS, NTM-BP and CO.  Absolute latencies are hardware- and interpreter-specific;
the paper reports values measured on an Intel i7-12700H / Python 3.11, so the
numbers here are a methodology reproduction, not a byte-identical replay.

Zero network/API calls.

Usage:
    python scripts/experiment_scalability_latency.py
Writes ``data/experiment_scalability_latency_results.json``.
"""
from __future__ import annotations

import io
import json
import os
import statistics
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from generate_scalability_graphs import generate_dataset  # noqa: E402

from ntm_bp.baselines import BfsCascadeEngine  # noqa: E402
from ntm_bp.engine import NtmBpEngine  # noqa: E402
from ntm_bp.experimental.co_engine import CoEngine  # noqa: E402
from ntm_bp.experimental.gate_utils import build_memory  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_scalability_latency_results.json")
SIZES = [10, 50, 100, 200, 500, 1000]
GRAPHS_PER_SIZE = 20


def time_update(engine, s):
    mem = build_memory(s)
    fid = s["update_fact_id"]
    old = mem.get_belief(fid)
    mem.update_fact(fid, s["new_value"], s["new_confidence"])
    t0 = time.perf_counter()
    if hasattr(engine, "propagate"):
        engine.propagate(mem, fid, source_old_belief=old)
    else:  # BFS baseline exposes cascade()
        engine.cascade(mem, fid)
    return (time.perf_counter() - t0) * 1000.0


def main():
    data = generate_dataset(SIZES, GRAPHS_PER_SIZE)
    engines = {
        "BFS": lambda: BfsCascadeEngine(),
        "NTM-BP": lambda: NtmBpEngine(gamma=0.7, tau=0.7, cutoff=0.01),
        "CO": lambda: CoEngine(gamma=0.7, tau=0.7, cutoff=0.01),
    }
    result = {name: {} for name in engines}
    for size in SIZES:
        scenarios = data[str(size)]
        for name, make in engines.items():
            times = []
            for s in scenarios:
                times.append(time_update(make(), s))
            result[name][str(size)] = statistics.median(times)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "sizes": SIZES,
                "graphs_per_size": GRAPHS_PER_SIZE,
                "statistic": "median per-update ms over the graphs at each size",
                "note": "absolute latency is hardware/interpreter specific; "
                        "paper Table 7 measured on i7-12700H / Python 3.11",
                "new_api_calls": 0,
            },
            "latency_ms": result,
        }, f, ensure_ascii=False, indent=2)

    header = "| Method | " + " | ".join(str(s) for s in SIZES) + " |"
    print("Per-update wall-clock latency (ms), median over {} graphs/size".format(
        GRAPHS_PER_SIZE))
    print(header)
    print("|" + "---|" * (len(SIZES) + 1))
    for name in engines:
        row = ["{}".format(name)]
        row += ["{:.2f}".format(result[name][str(s)]) for s in SIZES]
        print("| " + " | ".join(row) + " |")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
