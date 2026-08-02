"""Audit a semantic-gate call log (Prompt cache audit tool).

Reads the JSON call log produced by ``scripts/run_gate_eval.py`` (or any
``call_log`` in the same shape) and checks the scientific-integrity
properties that the paper's "cache audit trail" claims depend on:

1. Parse robustness: every raw LLM response must yield a parseable
   score/answer (a response that fails to parse silently falls back to a
   default -- those calls are flagged).
2. Prompt hygiene: prompts must contain ONLY fact texts. Any occurrence
   of the hand-annotated labels (``supports`` / ``contradicts`` /
   ``neutral`` / ``theta`` / ``relation_type`` / ``ground_truth``) in a
   prompt is a leak and is flagged as FATAL.
3. Completeness: every call must carry prompt + raw_response + parsed
   value (no silent drops).

Usage:
    python scripts/audit_cache.py [path/to/gate_call_log.json]

Exit code 0 = audit passed, 1 = issues found.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Substrings that must never appear inside a gate prompt (they would leak
# the hand-annotated labels the paper promises to keep out of the LLM).
LEAK_PATTERNS = [
    "supports", "contradicts", "neutral",
    "relation_type", "relation types", "theta", "ground_truth",
    "annotation_rationale",
]

DEFAULT_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "gate_call_log.json")


def audit(entries: list) -> dict:
    """Audit a list of call-log entries.

    Each entry is expected to have at least:
        {"kind"|"target_fact_id": ..., "prompt": str, "raw_response": str, ...}
    """
    issues: list[dict] = []
    stats = {"total_calls": len(entries), "parse_failures": 0, "leaks": 0}

    for i, entry in enumerate(entries):
        prompt = entry.get("prompt", "")
        raw = entry.get("raw_response", "")

        # 1. Completeness
        if not prompt.strip():
            issues.append({"index": i, "type": "empty_prompt",
                           "entry_keys": sorted(entry.keys())})
        if not raw.strip():
            issues.append({"index": i, "type": "empty_response"})

        # 2. Parse robustness: a numeric/letter response that the parser
        # could not decode would have fallen back to a default silently.
        kind = entry.get("kind")
        if kind == "change_type":
            ok = bool(re.search(r"[ABC]", raw.upper()))
        elif kind == "relevance":
            ok = bool(re.search(r"是|否", raw))
        elif kind == "stance":
            ok = bool(re.search(r"同向|对立", raw))
        elif kind in (None, ""):
            # Scalar impact gate: parsed_score field tells us directly.
            parsed = entry.get("parsed_score")
            ok = parsed is not None and isinstance(parsed, (int, float))
        else:
            ok = True  # unknown kinds are not audited for parsing
        if not ok:
            stats["parse_failures"] += 1
            issues.append({"index": i, "type": "parse_failure",
                           "kind": kind,
                           "raw_response": raw[:200]})

        # 3. Prompt hygiene (leak detection)
        for pat in LEAK_PATTERNS:
            if pat in prompt.lower():
                stats["leaks"] += 1
                issues.append({"index": i, "type": "leak",
                               "pattern": pat, "severity": "FATAL"})
                break

    return {"stats": stats, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", nargs="?", default=DEFAULT_LOG,
                        help="path to the call log JSON (default: data/gate_call_log.json)")
    args = parser.parse_args()

    if not os.path.isfile(args.log_path):
        print(f"ERROR: no call log found at {args.log_path}")
        print("Run scripts/run_gate_eval.py first (--mock for a plumbing test).")
        sys.exit(1)

    with open(args.log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("calls", []) if isinstance(data, dict) else data
    result = audit(entries)
    stats = result["stats"]

    print("=" * 60)
    print("Prompt cache audit")
    print("=" * 60)
    print(f"  calls audited:      {stats['total_calls']}")
    print(f"  parse failures:     {stats['parse_failures']}")
    print(f"  annotation leaks:   {stats['leaks']}")

    if result["issues"]:
        print(f"\nIssues found: {len(result['issues'])}")
        for issue in result["issues"][:10]:
            print(f"  [{issue['index']}] {issue['type']}: {issue}")
        print("\nAUDIT FAILED")
        sys.exit(1)

    print("\nNo issues found -- audit passed.")
    print("Prompts contain only fact texts; every response parsed cleanly.")
    sys.exit(0)


if __name__ == "__main__":
    main()
