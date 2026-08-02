"""Create the frozen dev/test case split for prompt iteration.

WHY THIS EXISTS (methodological honesty)
----------------------------------------
The gate prompt failure analysis (over-firing on "confirmation of a
known negative" updates) tempts us to edit the prompt. But tuning the
prompt against the SAME data we report on is prompt-level circular
evaluation -- the exact sin P0 removed from the ground truth. So:

* Prompt iteration may ONLY look at dev cases.
* Test cases are SEALED: they are evaluated ONCE, with the final frozen
  prompt, for the paper's reported numbers.

SPLIT DESIGN
------------
* Unit of splitting is the TEMPLATE, not the case: cases of the same
  template share fact texts, so putting siblings on both sides would
  leak tuned-on wording into test.
* pc500 / syndep: templates shuffled with a fixed seed, ~40% to dev.
* co_relation: ALL 8 scenarios go to dev. The current prompt's rule 4
  was explicitly designed from the CO-Relation blind-spot analysis, so
  this set is already contaminated as a design set and can never serve
  as held-out test data.

Usage:
    python scripts/make_case_split.py
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "case_split.json")

SEED = 20260731
DEV_TEMPLATE_FRACTION = 0.4


def load_cases(path: str) -> dict:
    """Return {template_id: sorted list of case_keys}."""
    with open(path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)["scenarios"]
    by_template = defaultdict(set)
    for s in scenarios:
        tid = s["template_id"]
        by_template[tid].add(f"{tid}||{s['new_value']}")
    return {tid: sorted(cases) for tid, cases in by_template.items()}


def split_dataset(name: str, rng: random.Random) -> dict:
    by_template = load_cases(os.path.join(DATA_DIR, f"{name}.json"))
    templates = sorted(by_template.keys())
    rng.shuffle(templates)
    n_dev = max(1, math.ceil(DEV_TEMPLATE_FRACTION * len(templates)))
    dev_templates = sorted(templates[:n_dev])
    test_templates = sorted(templates[n_dev:])
    dev_cases = sorted(c for t in dev_templates for c in by_template[t])
    test_cases = sorted(c for t in test_templates for c in by_template[t])
    print(f"{name}: {len(templates)} templates -> "
          f"dev {len(dev_templates)} templates / {len(dev_cases)} cases, "
          f"test {len(test_templates)} templates / {len(test_cases)} cases")
    return {
        "dev_templates": dev_templates,
        "test_templates": test_templates,
        "dev_cases": dev_cases,
        "test_cases": test_cases,
    }


def main():
    if os.path.isfile(OUTPUT_PATH):
        print(f"REFUSING to overwrite existing split: {OUTPUT_PATH}")
        print("The split must stay frozen. Delete it manually only if you "
              "understand this invalidates all prompt iteration done so far.")
        return

    rng = random.Random(SEED)
    result = {
        "metadata": {
            "seed": SEED,
            "dev_template_fraction": DEV_TEMPLATE_FRACTION,
            "split_unit": "template (cases follow their template)",
            "sealed_note": (
                "TEST cases are sealed: no prompt/threshold tuning may look "
                "at them. They are evaluated once with the final frozen "
                "prompt for reported numbers."
            ),
            "co_relation_note": (
                "All co_relation scenarios are dev-only: the gate prompt's "
                "rule 4 was designed from the CO-Relation blind-spot "
                "analysis, so this set is a design set, not held-out data."
            ),
        },
        "datasets": {},
    }
    for name in ["pc500", "syndep"]:
        result["datasets"][name] = split_dataset(name, rng)

    with open(os.path.join(DATA_DIR, "co_relation_test.json"),
              "r", encoding="utf-8") as f:
        co_ids = sorted(s["scenario_id"]
                        for s in json.load(f)["scenarios"])
    result["datasets"]["co_relation"] = {
        "dev_templates": co_ids,
        "test_templates": [],
        "dev_cases": co_ids,
        "test_cases": [],
    }
    print(f"co_relation: {len(co_ids)} scenarios -> all dev (design set)")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Split saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
