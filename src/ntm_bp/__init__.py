"""ntm-bp: a zero-dependency belief-propagation engine.

NTM-BP (Neural Truth Maintenance with Belief Propagation) replaces the
binary STALE flag of classical Truth Maintenance Systems with a
continuous belief state and propagates confidence *decay* along the
edges of a probabilistic dependency graph.  Only conclusions whose
belief drops below a threshold are marked REEVAL_REQUIRED, so an LLM
re-evaluation budget is spent where it matters.

Submodules
----------
* ``ntm_bp.evaluation`` -- benchmark datasets, metrics and the
  experiment runner that reproduces the paper's Figure 1.
* ``ntm_bp.baselines`` -- the BFS cascade baseline (binary STALE).
* ``ntm_bp.llm`` -- LLM provider abstraction (mock + key-less
  DeepSeek client; the API key comes from the environment).
* ``ntm_bp.experimental`` -- CO engine and LLM semantic gates
  (research code, NOT part of the stable API -- see its docstring).
"""

from ntm_bp.engine import NtmBpEngine
from ntm_bp.graph import DependencyGraph
from ntm_bp.memory import Fact, FactStatus, FactVersion, MemoryState
from ntm_bp.propagation import (
    DecayCalculator,
    PropagationMessage,
    PropagationQueue,
)

__version__ = "0.1.0"

__all__ = [
    "DecayCalculator",
    "DependencyGraph",
    "Fact",
    "FactStatus",
    "FactVersion",
    "MemoryState",
    "NtmBpEngine",
    "PropagationMessage",
    "PropagationQueue",
]
