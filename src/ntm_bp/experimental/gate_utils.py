"""Gate evaluation utilities: CachedLLM wrapper and build_memory helper.

Extracted from research repo's experiment_gate_generalization.py for use by
sealed test, expansion eval, keyword baseline, routing simulation, etc.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from ntm_bp.graph import DependencyGraph
from ntm_bp.llm import DeepSeekClient
from ntm_bp.memory import MemoryState


class CachedLLM:
    """Prompt-hash cache around DeepSeekClient.

    Identical prompts (same case, different jitter instance) hit the
    cache; only unique prompts reach the API. Exact, not approximate.
    """

    def __init__(self, client: DeepSeekClient, cache: Dict[str, Any]) -> None:
        self._client = client
        self.cache = cache
        self.api_calls = 0
        self.requests = 0

    def complete(self, prompt: str) -> str:
        self.requests += 1
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key not in self.cache:
            self.cache[key] = {
                "prompt": prompt,
                "response": self._client.complete(prompt),
            }
            self.api_calls += 1
        return self.cache[key]["response"]


def build_memory(s: Dict[str, Any]) -> MemoryState:
    """Build MemoryState + DependencyGraph from a scenario dict."""
    mem = MemoryState(tau=0.7)
    graph = DependencyGraph()
    for fid, (val, conf) in s["facts"].items():
        mem.add_fact(fid, val, conf)
        graph.add_node(fid)
    for p, c in s["dependencies"]:
        graph.add_edge(p, c)
    mem.graph = graph
    return mem