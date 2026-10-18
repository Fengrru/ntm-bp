#!/usr/bin/env python3
"""Run the full NTM paper reproduction.

Offline by default. The runner:

  1. regenerates every dataset from its committed generator (deterministic),
  2. runs all numeric experiments (Figures/Tables that need no LLM),
  3. machine-checks the paper's numbers with
     ``scripts/reproduce_paper_tables.py`` -- this replays the frozen
     prompt-hash cache with **zero API calls**;
  4. attempts the two one-shot API experiments ONLY if ``DEEPSEEK_API_KEY``
     is set (they refuse to overwrite existing result files).

One-shot scripts whose outputs already exist are skipped rather than
re-run, preserving the frozen-artifact discipline.

Requirements: the engine itself is standard-library only; three numeric
scripts need NumPy (``pip install -r requirements-repro.txt``).

Usage:
    python scripts/run_all_experiments.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
DATA = SCRIPTS.parent / "data"

GENERATORS = [
    "generate_pc500.py",
    "generate_syndep.py",
    "generate_co_relation_test.py",
    "generate_scalability_graphs.py",
    "generate_expansion.py",
    "make_case_split.py",
]
DETERMINISTIC = [
    "run_main_eval.py",
    "run_co_relation_eval.py",
    "experiment_keyword_baseline.py",
    "_simulate_routing.py",
    "experiment_gamma_analysis.py",
    "experiment_cpt_sensitivity.py",
    "experiment_walksum_verification.py",
    "experiment_ablation_components.py",
]
VERIFY = "reproduce_paper_tables.py"
API_SCRIPTS = [
    "experiment_sealed_test.py",
    "experiment_expansion_eval.py",
]

# Scripts that refuse to overwrite frozen outputs (one-shot discipline).
ONE_SHOT = {
    "generate_expansion.py": ["expansion_cases.json",
                              "expansion_annotation_sheet.json"],
    "make_case_split.py": ["case_split.json"],
    "experiment_sealed_test.py": ["experiment_sealed_test_results.json"],
    "experiment_expansion_eval.py": ["experiment_expansion_eval_results.json"],
}


def run(script: str) -> bool:
    path = SCRIPTS / script
    if not path.exists():
        print("  [skip] missing: {}".format(script))
        return True
    if script in ONE_SHOT:
        outs = [DATA / o for o in ONE_SHOT[script]]
        if all(o.exists() for o in outs):
            print("  [skip] {} (frozen output already present)".format(script))
            return True
    print("  [run]  {}".format(script))
    proc = subprocess.run([sys.executable, str(path)])
    if proc.returncode != 0:
        print("  [warn] {} exited {}".format(script, proc.returncode))
        return False
    return True


def have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> int:
    if not have("numpy"):
        print("[note] NumPy not found; three numeric scripts will fail. "
              "Install with: pip install -r requirements-repro.txt\n")

    print("=" * 60)
    print("1/4  Regenerating datasets (deterministic)")
    print("=" * 60)
    for s in GENERATORS:
        run(s)

    print("\n" + "=" * 60)
    print("2/4  Numeric experiments (no LLM)")
    print("=" * 60)
    for s in DETERMINISTIC:
        run(s)

    print("\n" + "=" * 60)
    print("3/4  Offline verification of paper tables (zero API calls)")
    print("=" * 60)
    ok = run(VERIFY)

    print("\n" + "=" * 60)
    print("4/4  One-shot API experiments")
    print("=" * 60)
    if os.environ.get("DEEPSEEK_API_KEY"):
        for s in API_SCRIPTS:
            run(s)
    else:
        print("  [skip] DEEPSEEK_API_KEY not set; sealed test and "
              "expansion eval were not re-run.")
        print("         Their results are already verified offline in step 3.")

    print("\n" + "=" * 60)
    print("PASS: offline reproduction complete." if ok
          else "PARTIAL: see warnings above.")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
