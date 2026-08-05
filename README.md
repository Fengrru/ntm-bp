# ntm-bp

**Zero-dependency belief-propagation engine for selective LLM re-evaluation in agentic memory.**

`ntm-bp` implements the NTM-BP algorithm (Neural Truth Maintenance with Belief
Propagation) from the NTM paper ("Neural Truth Maintenance: From Binary
Stale Marking to Probabilistic Belief Propagation for LLM Agent Memory",
Yuhan Feng, 2026): when a fact in an agent's memory is
updated, confidence **decay** is propagated along the edges of a probabilistic
dependency graph. Only conclusions whose belief drops below a threshold
`tau` are marked `REEVAL_REQUIRED`, so LLM re-evaluation budget is spent
where it actually matters.

```
pip install -e .
```

That's the whole dependency story: **the runtime has zero third-party
dependencies** (pure Python 3.9+ standard library). The original research
implementation wrapped `networkx`; this port uses plain dict adjacency lists.
Copy `src/ntm_bp/` into your project if you don't even want pip.

---

## Why

Classical Truth Maintenance Systems use a binary STALE flag: one premise
update **blanket-invalidates every transitive conclusion**. With LLM-backed
memory, re-evaluating all of them is expensive (one LLM call per fact).

NTM-BP instead treats confidence as a continuous quantity and transmits a
decay factor along each edge:

| step | formula |
| --- | --- |
| initial delta (direct children) | `delta_init = source_change_ratio * P(child \| parent)` |
| next-hop delta | `delta_next = delta * gamma * P(child \| parent)` |
| updated belief | `B_new = B_old * (1 - delta)` |
| re-evaluation trigger | `B_new < tau` → `REEVAL_REQUIRED` |

- `P(child | parent)` comes from the edge's Conditional Probability Table
  (CPT); unknown entries default to the uniform `0.5`.
- `gamma` (default `0.7`) damps decay per hop.
- `source_change_ratio` scales the initial delta by how much the source
  fact's belief actually changed (sigmoid-shaped, so small changes still
  produce a noticeable but small cascade).
- `cutoff` (default `0.01`) stops propagation when the delta becomes
  negligible; a priority queue processes larger deltas first.

## Quick start

```python
from ntm_bp import MemoryState, NtmBpEngine
from ntm_bp.graph import DependencyGraph

mem = MemoryState(tau=0.7)
mem.add_fact("f1", "Acme Corp 信誉良好", 0.9)
mem.add_fact("f2", "Contract A 低风险", 0.85)
mem.add_fact("f3", "推荐投资", 0.8)

graph = DependencyGraph()
# Strong first edge: f2 is fully determined by f1 (P=1.0).
graph.add_edge("f1", "f2", cpt={"Acme Corp 信誉良好": {"Contract A 低风险": 1.0}})
# Weak second edge: f3 is nearly independent of f2 (P=0.1).
graph.add_edge("f2", "f3", cpt={"Contract A 低风险": {"推荐投资": 0.1}})
mem.graph = graph

old = mem.get_belief("f1")
mem.update_fact("f1", "Acme Corp 信誉良好", 0.4)  # downgrade: 0.9 -> 0.4

engine = NtmBpEngine(gamma=0.7, tau=0.7)
mem, report = engine.propagate_with_report(mem, "f1", source_old_belief=old)

print(report["reeval_facts"])       # ['f2'] -- only f2 needs an LLM call
print(mem.facts["f3"].status)       # FactStatus.ACTIVE -- f3 survives
```

Run the bundled example with:

```
python examples/quickstart.py
```

## API surface

| symbol | description |
| --- | --- |
| `NtmBpEngine(gamma, tau, cutoff, max_iterations, lazy)` | the propagation engine; `propagate(memory, fact_id, source_old_belief)` mutates and returns the memory, `propagate_with_report(...)` additionally returns affected facts and belief changes |
| `MemoryState(facts, belief, time, tau, graph)` | fact store + belief map; `add_fact`, `update_fact`, `get_belief`, `set_belief`, `mark_reeval`, `get_reeval_facts`, `to_snapshot`/`from_snapshot` |
| `DependencyGraph` | directed graph with per-edge CPTs and optional `relation_type` tags; `add_edge(parent, child, cpt, relation_type)`, `get_conditional_probability`, `get_relation_type`, `children`, `descendants`, `has_cycle`, `to_dict`/`from_dict` |
| `Fact`, `FactVersion`, `FactStatus` | data models (versioned values, `ACTIVE`/`STALE`/`REEVAL_REQUIRED`/`PENDING`) |
| `DecayCalculator`, `PropagationQueue`, `PropagationMessage` | the decay arithmetic and priority queue (subclass-friendly) |
| `ntm_bp.baselines.BfsCascadeEngine` | the BFS baseline: marks every reachable descendant STALE (binary system) |
| `ntm_bp.evaluation` | `Scenario`, dataset loaders, metrics (RR/F1/FNR/FPR/AGM), `ExperimentRunner` -- reproduces the paper's Figure 1 |
| `ntm_bp.llm` | `LLMInterface`, `MockLLMClient`, key-less `DeepSeekClient` (reads `DEEPSEEK_API_KEY` from the environment) |
| `ntm_bp.experimental` | CO engine and LLM semantic gates -- research code, **not** part of the stable API |

## Datasets, generators and reproduction

The paper's numbers come from two synthetic benchmarks whose **full
generators are shipped in `scripts/`** (deterministic, seed 42, including
the ±0.05 per-instance confidence jitter), so every number in the paper's
Table 1 / Table 2 / Figure 1 can be reproduced byte-for-byte:

```
python scripts/generate_pc500.py            # 500 instances, 6 templates x 3 update values
python scripts/generate_syndep.py           # 200 instances, 10 templates x 3 update values
python scripts/generate_scalability_graphs.py  # 10..1000 nodes, 20 graphs per size
python scripts/generate_co_relation_test.py # 8 relation-type scenarios (design set)
python scripts/make_case_split.py           # frozen dev/test split (seed 20260731)
```

The ground truth (`reeval` sets + `theta`) is **manually annotated per
(template, update_value) case and is algorithm-independent** -- no engine
output was used to produce labels (see the generator docstrings). The
±0.05 jitter is clamped to `[0.05, 0.99]` so annotated semantics stay
valid for every instance.

### Reproduce Figure 1 (three-way comparison)

```
python scripts/run_main_eval.py
```

Runs BFS / NTM-BP / CO on both datasets with `tau=0.7, gamma=0.7,
cutoff=0.01` and prints Markdown tables. Expected key numbers on PC-500:

| metric | BFS | NTM-BP | CO |
| --- | --- | --- | --- |
| RR | 0.694 | 0.491 | 0.491 |
| F1 | 0.759 | 0.818 | 0.818 |
| AGM | 2.385 | 0.956 | 0.956 |

(CO == NTM-BP here because these datasets carry no relation annotations,
so every edge is `default`.)

### CO relation-type ablation (negative result)

```
python scripts/run_co_relation_eval.py
```

Reproduces the paper's negative result (Sec 6.7): CO on all-`neutral`
edges degrades badly (F1 ~ 0.709 on PC-500). Note the paper's CO
table uses *oracle* relation labels -- the numbers bound the value of
perfect edge typing and exclude classification cost. This is exactly
why the CO engine lives under `ntm_bp.experimental` instead of the
stable API.

## LLM gate, prompts and the cache audit tool

The structured semantic gate (Q1 change type / Q2 relevance / Q3 stance)
and the frozen v1/v2/v3 scalar prompts are documented in
docs/prompts.md. Every LLM call is recorded (`prompt` + `raw_response` +
parsed value) in an audit trail:

```
# plumbing test with a deterministic mock (NOT experimental evidence)
python scripts/run_gate_eval.py --mock --dataset pc500

# real experiment (needs DEEPSEEK_API_KEY in the environment)
python scripts/run_gate_eval.py --dataset pc500

# audit the call log: parse robustness + prompt-hygiene (leak) checks
python scripts/audit_cache.py
```

The audit tool fails on any prompt that contains the hand-annotated
labels (`supports`/`contradicts`/`neutral`/`theta`/...), enforcing the
paper's promise that the LLM sees only fact texts.

## Input format (DependencyGraph JSON Schema)

Real-world graph extraction is future work (paper Sec 7.3,
"Dependency graph extraction" in Limitations): the graph is
*given* to the engine. To make the engine pluggable into RAG / agent
memory systems today, the interchange format is standardized in
[schemas/dependency_graph.schema.json](schemas/dependency_graph.schema.json)
-- nodes, edges, CPT matrices and relation types, plus the full scenario
format used by the benchmarks. `DependencyGraph.from_dict`/`to_dict`
round-trip this format, so any system that can emit it can run NTM-BP.

## Tests

```
pip install -e .[test]
pytest
```

The suite covers the decay formulas, cutoff truncation, threshold
behavior, cyclic graphs, partial overthrow (strong edges keep a grandchild
ACTIVE), gamma sensitivity, snapshot round-trips, the BFS baseline, the
evaluation pipeline, the CO engine and the semantic gates.

## Relation to the research project

This package is the lightweight, dependency-free extraction of the
numerical engine evaluated in the NTM paper, plus the deterministic
evaluation harness for the PC-500 and SynDep benchmarks (paper
Tables/Figures for BFS, NTM-BP, CO). The LLM-gate experiments
(sealed test with frozen prompts, Expansion-500 external validation,
keyword baselines, call-count audit, cost model, significance tests)
and the prompt-iteration history live in the research repository; the
frozen prompts ship here under `ntm_bp.experimental.semantic_gate` and
docs/prompts.md, with the algorithm unchanged.

Note: the committed `data/gate_call_log.json` is the **mock** plumbing
run (`"mock": true` in its metadata) -- it demonstrates the audit
format, it is not experimental evidence.

## License

MIT. See [LICENSE](LICENSE).
