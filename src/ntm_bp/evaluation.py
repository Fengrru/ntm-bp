"""Evaluation pipeline: datasets, metrics and experiment runner.

Ported from the NTM paper project (``ntm.evaluation`` +
``ntm.experiments.runner``) with zero third-party dependencies.  The
ground truth in both datasets is manually annotated per
(template, update_value) case, independent of any propagation algorithm
-- see the generator scripts under ``scripts/``.

Modes are plain strings: ``"bfs"`` (BFS baseline), ``"ntm_bp"`` (the
paper's method), ``"co"`` (experimental CO engine, imported lazily from
:mod:`ntm_bp.experimental.co_engine`).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ntm_bp.baselines import BfsCascadeEngine
from ntm_bp.engine import NtmBpEngine
from ntm_bp.graph import DependencyGraph
from ntm_bp.memory import MemoryState

# ----------------------------------------------------------------------
# Experiment modes (plain strings so scripts stay import-light)
# ----------------------------------------------------------------------

MODE_BFS = "bfs"
MODE_NTM_BP = "ntm_bp"
MODE_CO = "co"
ALL_MODES: Tuple[str, ...] = (MODE_BFS, MODE_NTM_BP, MODE_CO)


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------

@dataclass
class Scenario:
    """A single evaluation scenario.

    Attributes:
        scenario_id: Unique identifier.
        facts: Initial facts (fact_id -> (value, confidence)).
        dependencies: List of (premise_id, conclusion_id) tuples.
        update_fact_id: The fact to update.
        new_value: The new value for the updated fact.
        new_confidence: The new confidence for the updated fact.
        ground_truth_reeval: Set of fact IDs that should be re-evaluated
            (manually annotated, algorithm-independent).
        ground_truth_values: Expected final values (fact_id -> value).
        theta: Annotated semantic impact degree [0, 1].
        relation_types: Mapping of "{parent}__{child}" -> relation_type string.
    """

    scenario_id: str
    facts: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    dependencies: List[Tuple[str, str]] = field(default_factory=list)
    update_fact_id: str = ""
    new_value: str = ""
    new_confidence: float = 0.5
    ground_truth_reeval: Set[str] = field(default_factory=set)
    ground_truth_values: Dict[str, str] = field(default_factory=dict)
    theta: float = 0.5
    relation_types: Dict[str, str] = field(default_factory=dict)


class DatasetLoader:
    """Base class for dataset loaders."""

    def load(self, path: str) -> List[Scenario]:
        """Load scenarios from a file path.

        Args:
            path: Path to the dataset file.

        Returns:
            List of Scenario objects.
        """
        raise NotImplementedError

    @staticmethod
    def load_from_json(data: Dict[str, Any]) -> List[Scenario]:
        """Parse a JSON dict into a list of Scenario objects.

        Expected JSON format::

            {
              "scenarios": [
                {
                  "scenario_id": "...",
                  "facts": {"f1": ["value", 0.9], ...},
                  "dependencies": [["parent", "child"], ...],
                  "update_fact_id": "...",
                  "new_value": "...",
                  "new_confidence": 0.5,
                  "ground_truth_reeval": ["f2", "f3"],
                  "ground_truth_values": {"f1": "..."},
                  "theta": 0.4
                },
                ...
              ]
            }

        Args:
            data: Parsed JSON dictionary.

        Returns:
            List of Scenario objects.
        """
        scenarios = []
        for s in data.get("scenarios", []):
            # Parse facts: {"f1": ["value", 0.9]} -> {"f1": ("value", 0.9)}
            facts = {}
            for fid, (value, confidence) in s.get("facts", {}).items():
                facts[fid] = (value, float(confidence))

            # Parse dependencies: [["parent", "child"]] -> [("parent", "child")]
            deps = [tuple(d) for d in s.get("dependencies", [])]

            # Parse relation_types: can be dict {"f1__f2": "supports"} or
            # list [["f1", "f2", "supports"]]
            rt_input = s.get("relation_types", {})
            if isinstance(rt_input, list):
                relation_types = {f"{p}__{c}": r for p, c, r in rt_input}
            else:
                relation_types = dict(rt_input)

            scenarios.append(
                Scenario(
                    scenario_id=s.get("scenario_id", ""),
                    facts=facts,
                    dependencies=deps,
                    update_fact_id=s.get("update_fact_id", ""),
                    new_value=s.get("new_value", ""),
                    new_confidence=float(s.get("new_confidence", 0.5)),
                    ground_truth_reeval=set(s.get("ground_truth_reeval", [])),
                    ground_truth_values=s.get("ground_truth_values", {}),
                    theta=float(s.get("theta", 0.5)),
                    relation_types=relation_types,
                )
            )
        return scenarios


class SynDepLoader(DatasetLoader):
    """Loader for the SynDep synthetic dependency dataset.

    SynDep is a fully synthetic benchmark (formerly misnamed
    "LongMemEval"; it is unrelated to the public benchmark of that
    name). If path points to a valid JSON file, it will be parsed using
    load_from_json(). Otherwise, returns a single synthetic scenario.
    """

    def load(self, path: str) -> List[Scenario]:
        """Load SynDep scenarios.

        If path is a valid JSON file, parse and return real scenarios.
        Otherwise, returns a single synthetic scenario for testing.
        """
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return self.load_from_json(data)
            except (json.JSONDecodeError, IOError):
                pass

        # Fallback: single synthetic scenario
        return [
            Scenario(
                scenario_id="syndep_fallback_001",
                facts={
                    "f1": ("Acme Corp 信誉良好", 0.9),
                    "f2": ("合同 A 条款合规", 0.85),
                    "f3": ("Contract A 低风险", 0.8),
                    "f4": ("推荐投资", 0.75),
                },
                dependencies=[
                    ("f1", "f3"),
                    ("f2", "f3"),
                    ("f3", "f4"),
                ],
                update_fact_id="f1",
                new_value="Acme Corp 被调查",
                new_confidence=0.4,
                ground_truth_reeval={"f3", "f4"},
                ground_truth_values={
                    "f1": "Acme Corp 被调查",
                    "f2": "合同 A 条款合规",
                },
                theta=0.7,
            )
        ]


class PartialContradictionLoader(DatasetLoader):
    """Loader for the PC-500 custom dataset.

    If the path points to a valid JSON file, it will be parsed using
    load_from_json(). Otherwise, synthetic scenarios are returned.
    """

    def load(self, path: str) -> List[Scenario]:
        """Load PC-500 scenarios.

        If path is a valid JSON file, parse and return real scenarios.
        Otherwise, return 3 synthetic scenarios covering different theta ranges.
        """
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return self.load_from_json(data)
            except (json.JSONDecodeError, IOError):
                pass

        # Fallback: synthetic scenarios
        return [
            Scenario(
                scenario_id="pc_001",
                facts={
                    "f1": ("Acme Corp 信誉良好", 0.9),
                    "f2": ("Contract A 低风险", 0.85),
                    "f3": ("推荐投资", 0.8),
                },
                dependencies=[("f1", "f2"), ("f2", "f3")],
                update_fact_id="f1",
                new_value="Acme Corp 被调查",
                new_confidence=0.4,
                ground_truth_reeval={"f2", "f3"},
                ground_truth_values={"f1": "Acme Corp 被调查"},
                theta=0.7,
            ),
            Scenario(
                scenario_id="pc_002",
                facts={
                    "f1": ("供应商 A 质量可靠", 0.9),
                    "f2": ("使用供应商 A", 0.85),
                },
                dependencies=[("f1", "f2")],
                update_fact_id="f1",
                new_value="供应商 A 偶尔延迟交货",
                new_confidence=0.7,
                ground_truth_reeval=set(),
                ground_truth_values={"f1": "供应商 A 偶尔延迟交货"},
                theta=0.15,
            ),
            Scenario(
                scenario_id="pc_003",
                facts={
                    "f1": ("市场处于牛市", 0.9),
                    "f2": ("投资策略偏进取", 0.85),
                    "f3": ("高仓位操作", 0.8),
                },
                dependencies=[("f1", "f2"), ("f2", "f3")],
                update_fact_id="f1",
                new_value="市场转为熊市",
                new_confidence=0.1,
                ground_truth_reeval={"f2", "f3"},
                ground_truth_values={"f1": "市场转为熊市"},
                theta=0.85,
            ),
        ]


# ----------------------------------------------------------------------
# Metrics (Section 7.3 of the whitepaper)
# ----------------------------------------------------------------------

def compute_recompute_rate(memory: MemoryState) -> float:
    """Compute the Recompute Rate (RR).

    RR = (# of STALE + # of REEVAL_REQUIRED facts) / total facts

    Args:
        memory: The memory state after cascade.

    Returns:
        Recompute rate in [0.0, 1.0].
    """
    if memory.fact_count == 0:
        return 0.0
    stale_or_reeval = len(memory.get_stale_facts()) + len(memory.get_reeval_facts())
    return stale_or_reeval / memory.fact_count


def compute_binary_recompute_rate(memory: MemoryState) -> float:
    """Compute RR for the binary STALE system.

    RR = #STALE / total facts
    """
    if memory.fact_count == 0:
        return 0.0
    return len(memory.get_stale_facts()) / memory.fact_count


def compute_accuracy(
    predictions: List[str],
    ground_truth: List[str],
) -> float:
    """Compute decision accuracy.

    Accuracy = (# matching predictions) / total predictions

    Args:
        predictions: List of predicted labels/values.
        ground_truth: List of ground truth labels/values.

    Returns:
        Accuracy in [0.0, 1.0].
    """
    if len(predictions) == 0:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions)


def compute_fnr_fpr(
    predicted_reeval: Set[str],
    actual_need_reeval: Set[str],
    total_facts: Set[str],
) -> Tuple[float, float]:
    """Compute False Negative Rate and False Positive Rate.

    FNR = miss-identified re-eval needs / actual re-eval needs
    FPR = false alarms / facts that didn't need re-eval

    Args:
        predicted_reeval: Facts marked as needing re-evaluation.
        actual_need_reeval: Facts that actually needed re-evaluation.
        total_facts: All facts in the system.

    Returns:
        Tuple of (FNR, FPR).
    """
    tp = len(predicted_reeval & actual_need_reeval)
    fn = len(actual_need_reeval - predicted_reeval)
    fp = len(predicted_reeval - actual_need_reeval)
    tn = len(total_facts - actual_need_reeval - predicted_reeval)

    fnr = fn / len(actual_need_reeval) if len(actual_need_reeval) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return fnr, fpr


def compute_f1(
    predicted_reeval: Set[str],
    actual_need_reeval: Set[str],
) -> float:
    """Compute F1 score for re-evaluation detection.

    F1 = 2 * P * R / (P + R)

    When both sets are empty the prediction is perfect (correctly
    identified that nothing needs re-evaluation), so F1 = 1.0.
    """
    if not predicted_reeval and not actual_need_reeval:
        return 1.0

    tp = len(predicted_reeval & actual_need_reeval)
    fp = len(predicted_reeval - actual_need_reeval)
    fn = len(actual_need_reeval - predicted_reeval)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_agm_distance(memory_before: MemoryState, memory_after: MemoryState) -> float:
    """Compute the AGM distance between two memory states.

    d_AGM(M, M') = sum(|B(f) - B'(f)|) over all f in F

    As defined in Appendix B of the whitepaper.
    """
    all_facts = set(memory_before.facts.keys()) | set(memory_after.facts.keys())
    total_distance = 0.0
    for fid in all_facts:
        b_before = memory_before.get_belief(fid) or 0.0
        b_after = memory_after.get_belief(fid) or 0.0
        total_distance += abs(b_before - b_after)
    return total_distance


def compute_cognitive_consistency(memory: MemoryState) -> float:
    """Compute the cognitive consistency score C(M).

    A simple heuristic: the variance of beliefs across the graph.
    Lower variance = higher consistency. (stdlib port of the original
    numpy-based implementation.)
    """
    beliefs = [memory.get_belief(fid) or 0.0 for fid in memory.facts]
    if len(beliefs) <= 1:
        return 1.0
    mean = sum(beliefs) / len(beliefs)
    variance = sum((b - mean) ** 2 for b in beliefs) / len(beliefs)
    return float(1.0 - variance)


def compute_token_savings(
    baseline_tokens: int,
    method_tokens: int,
) -> float:
    """Compute Token Savings (TS) ratio.

    TS = (baseline_tokens - method_tokens) / baseline_tokens
    """
    if baseline_tokens == 0:
        return 0.0
    return (baseline_tokens - method_tokens) / baseline_tokens


def compute_accuracy_retain_rate(
    retained_facts: Set[str],
    still_correct_facts: Set[str],
) -> float:
    """Compute the Accuracy of Retained (Acc_retain).

    Acc_retain = #correct_retained / #total_retained
    """
    if not retained_facts:
        return 1.0
    correct_retained = len(retained_facts & still_correct_facts)
    return correct_retained / len(retained_facts)


# ----------------------------------------------------------------------
# Experiment runner
# ----------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    dataset_path: str = ""
    loader: Optional[DatasetLoader] = None
    tau: float = 0.7
    gamma: float = 0.7
    propagation_cutoff: float = 0.01
    mode: str = MODE_BFS
    description: str = ""
    engine_factory: Optional[Callable[[], Any]] = None


@dataclass
class ExperimentResult:
    """Results from running an experiment."""

    config: ExperimentConfig
    scenario_results: List[Dict[str, Any]] = field(default_factory=list)
    wall_time_seconds: float = 0.0

    @property
    def summary(self) -> Dict[str, float]:
        """Aggregate metrics across all scenarios."""
        if not self.scenario_results:
            return {}

        rr_values = [r.get("recompute_rate", 0.0) for r in self.scenario_results]
        agm_values = [r.get("agm_distance", 0.0) for r in self.scenario_results]
        fnr_values = [r.get("fnr", 0.0) for r in self.scenario_results]
        fpr_values = [r.get("fpr", 0.0) for r in self.scenario_results]
        f1_values = [r.get("f1", 0.0) for r in self.scenario_results]

        return {
            "avg_recompute_rate": sum(rr_values) / len(rr_values) if rr_values else 0.0,
            "avg_agm_distance": sum(agm_values) / len(agm_values) if agm_values else 0.0,
            "avg_fnr": sum(fnr_values) / len(fnr_values) if fnr_values else 0.0,
            "avg_fpr": sum(fpr_values) / len(fpr_values) if fpr_values else 0.0,
            "avg_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
            "scenario_count": len(self.scenario_results),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "config": {
                "tau": self.config.tau,
                "gamma": self.config.gamma,
                "mode": self.config.mode,
                "description": self.config.description,
            },
            "summary": self.summary,
            "scenarios": self.scenario_results,
            "wall_time_seconds": self.wall_time_seconds,
        }


class ExperimentRunner:
    """Orchestrates running experiments on evaluation datasets."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        if config.engine_factory is not None:
            self._engine: Any = config.engine_factory()
        elif config.mode == MODE_NTM_BP:
            self._engine = NtmBpEngine(
                tau=config.tau,
                gamma=config.gamma,
                cutoff=config.propagation_cutoff,
            )
        elif config.mode == MODE_CO:
            # Lazy import: the CO engine lives in the experimental
            # namespace and is not part of the stable API.
            from ntm_bp.experimental.co_engine import CoEngine

            self._engine = CoEngine(
                gamma=config.gamma,
                tau=config.tau,
                cutoff=config.propagation_cutoff,
            )
        else:
            self._engine = BfsCascadeEngine()

    def run(self) -> ExperimentResult:
        """Execute the experiment.

        Returns:
            ExperimentResult containing per-scenario and aggregated metrics.
        """
        start_time = time.time()

        if self.config.loader is None:
            raise ValueError("No dataset loader configured.")

        scenarios = self.config.loader.load(self.config.dataset_path)
        result = ExperimentResult(config=self.config)

        for scenario in scenarios:
            scenario_result = self._run_single(scenario)
            result.scenario_results.append(scenario_result)

        result.wall_time_seconds = time.time() - start_time
        return result

    def _run_single(self, scenario: Scenario) -> Dict[str, Any]:
        """Run a single evaluation scenario."""
        # Initialize memory
        memory = MemoryState(tau=self.config.tau)

        # Build dependency graph
        graph = DependencyGraph()

        # Add facts and dependencies
        for fid, (value, confidence) in scenario.facts.items():
            memory.add_fact(fid, value, confidence)
            graph.add_node(fid)

        for parent_id, child_id in scenario.dependencies:
            edge_key = f"{parent_id}__{child_id}"
            rel_type = scenario.relation_types.get(edge_key, "default")
            graph.add_edge(parent_id, child_id, relation_type=rel_type)

        # Attach graph to memory for the engine
        memory.graph = graph

        # Snapshot before update (for AGM distance)
        before_snapshot = memory.to_snapshot()
        before_state = MemoryState.from_snapshot(before_snapshot)

        # Save old belief before update (for source_change scaling)
        old_belief = memory.get_belief(scenario.update_fact_id)

        # Perform update
        memory.update_fact(scenario.update_fact_id, scenario.new_value, scenario.new_confidence)

        # Run propagation/cascade
        try:
            if self.config.mode in (MODE_NTM_BP, MODE_CO):
                self._engine.propagate(memory, scenario.update_fact_id, source_old_belief=old_belief)
            else:
                self._engine.cascade(memory, scenario.update_fact_id)
        except RuntimeError:
            # Fallback if graph not properly attached
            pass

        # Compute metrics
        if self.config.mode in (MODE_NTM_BP, MODE_CO):
            rr = compute_recompute_rate(memory)
            predicted_reeval = set(memory.get_reeval_facts())
        else:
            rr = compute_binary_recompute_rate(memory)
            predicted_reeval = set(memory.get_stale_facts())
        agm_dist = compute_agm_distance(before_state, memory)
        actual_need_reeval = scenario.ground_truth_reeval
        total_facts = set(memory.facts.keys())

        fnr, fpr = compute_fnr_fpr(predicted_reeval, actual_need_reeval, total_facts)
        f1 = compute_f1(predicted_reeval, actual_need_reeval)

        return {
            "scenario_id": scenario.scenario_id,
            "theta": scenario.theta,
            "recompute_rate": rr,
            "agm_distance": agm_dist,
            "fnr": fnr,
            "fpr": fpr,
            "f1": f1,
            "predicted_reeval": list(predicted_reeval),
            "actual_need_reeval": list(actual_need_reeval),
        }
