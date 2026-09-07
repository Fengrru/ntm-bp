#!/usr/bin/env python3
"""
Run ALL experiments for the NTM paper (release version).
Usage:
    python scripts/run_all_experiments.py
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "generate_pc500.py",
    "generate_syndep.py",
    "generate_co_relation_test.py",
    "generate_scalability_graphs.py",
    "generate_expansion.py",
    "make_case_split.py",
    "run_main_eval.py",
    "run_co_relation_eval.py",
    "experiment_sealed_test.py",
    "experiment_expansion_eval.py",
    "experiment_keyword_baseline.py",
    "_simulate_routing.py",
    "experiment_gamma_analysis.py",
    "experiment_cpt_sensitivity.py",
    "experiment_walksum_verification.py",
    "experiment_ablation_components.py",
]

def main():
    script_dir = Path(__file__).parent
    for s in SCRIPTS:
        path = script_dir / s
        if not path.exists():
            print(f"⚠️  Skipping missing: {s}")
            continue
        print(f"
{'='*60}")
        print(f"Running {s}...")
        print(f"{'='*60}")
        try:
            subprocess.run([sys.executable, str(path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌  {s} failed with exit code {e.returncode}")
            # continue to next script

    print("\n✅  All available experiments completed.")

if __name__ == "__main__":
    main()
