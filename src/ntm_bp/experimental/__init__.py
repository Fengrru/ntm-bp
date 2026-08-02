"""Experimental algorithms (NOT part of the stable core API).

Everything under :mod:`ntm_bp.experimental` is research code ported from
the NTM paper project.  It is kept off the main import path on purpose:

* :mod:`ntm_bp.experimental.co_engine` -- the CO (Cognitive Override)
  engine degrades on ``neutral`` relation chains (F1 drops to 0.709 on
  PC-500); use the standard :class:`ntm_bp.NtmBpEngine` unless you know
  your relation annotations are reliable.
* :mod:`ntm_bp.experimental.semantic_gate` -- LLM-based semantic gates
  (Q1/Q2/Q3 structured gate) that require a real LLM client and whose
  prompts were frozen after dev-set iteration.

Importing ``ntm_bp.experimental`` does not import the engines themselves;
use ``from ntm_bp.experimental.co_engine import CoEngine`` explicitly.
"""

from __future__ import annotations

__all__: list[str] = []
