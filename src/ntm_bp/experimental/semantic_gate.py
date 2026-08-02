"""Semantic Gate: LLM-based semantic re-evaluation for NTM.

EXPERIMENTAL -- not part of the stable core API.  Requires a real LLM
client (e.g. :class:`ntm_bp.llm.DeepSeekClient`) and the prompts below
were FROZEN after dev-set iteration (see ``scripts/make_case_split.py``
and ``docs/prompts.md``).

Motivation (from the CO-Relation error analysis): all numeric engines
(BFS / NTM-BP / CO) are value-blind -- they propagate over confidence
deltas, CPTs and gamma, but never look at WHAT the update says. This
produces three systematic failure modes:

  1. A semantically unrelated (neutral) link should block even a strong
     update, but numeric engines propagate anyway.
  2. A reinforcing update over a contradicts relation causes no
     confidence drop, so numeric engines stay silent -- yet the
     contradicted downstream conclusion is now MORE doubtful.
  3. A mild update (e.g. one late delivery) barely changes the premise,
     but numeric engines still cascade.

The SemanticGateEngine addresses this by asking an LLM, for every fact
reachable from the updated source, whether the update semantically
undermines that downstream conclusion, given the textual dependency
path. Facts whose impact score >= tau_gate are marked REEVAL_REQUIRED.

HONESTY NOTES
-------------
* The LLM prompt receives ONLY the fact texts along the dependency
  path. It does NOT receive the hand-annotated relation types
  (supports/contradicts/neutral) nor the theta annotation -- the
  semantic relation must be inferred from the text itself.
* Every LLM call (prompt + raw response + parsed score) is recorded in
  `self.call_log` so experiment outputs can be audited.
* MockLLMClient may be used to test the plumbing, but results obtained
  with a mock are NOT experimental evidence.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Callable, Dict, List, Optional

from ntm_bp.graph import DependencyGraph
from ntm_bp.llm import LLMInterface
from ntm_bp.memory import MemoryState


def collect_descendant_paths(
    graph: DependencyGraph, source_fact_id: str
) -> Dict[str, List[str]]:
    """BFS from source, returning {descendant_id: path_from_source}.

    The path includes both endpoints. Shared by SemanticGateEngine and
    StructuredGateEngine.
    """
    paths: Dict[str, List[str]] = {}
    queue = deque([(source_fact_id, [source_fact_id])])
    visited = {source_fact_id}
    while queue:
        fid, path = queue.popleft()
        for child_id in graph.children(fid):
            if child_id in visited:
                continue
            visited.add(child_id)
            child_path = path + [child_id]
            paths[child_id] = child_path
            queue.append((child_id, child_path))
    return paths


def _path_blocks(
    memory: MemoryState,
    path: List[str],
    target_id: str,
    new_confidence: Optional[float],
) -> tuple:
    """Shared prompt fragments: (target_text, chain_block, conf_line)."""
    target_fact = memory.get_fact(target_id)
    target_text = target_fact.value if target_fact else target_id

    chain_lines = []
    for fid in path[1:-1]:
        fact = memory.get_fact(fid)
        chain_lines.append(f"- {fact.value if fact else fid}")
    chain_block = (
        "该结论通过以下中间结论间接依赖于被更新的事实：\n"
        + "\n".join(chain_lines) + "\n"
    ) if chain_lines else "该结论直接依赖于被更新的事实。\n"

    conf_line = (
        f"（对新陈述本身的置信度约为 {new_confidence:.2f}）\n"
        if new_confidence is not None else ""
    )
    return target_text, chain_block, conf_line


def build_prompt_v1(
    memory: MemoryState,
    path: List[str],
    target_id: str,
    old_value: str,
    new_value: str,
    new_confidence: Optional[float],
) -> str:
    """Original gate prompt (FROZEN -- used in all pre-split experiments).

    Known failure mode (full-dataset generalization run): rule 4 makes
    the LLM over-fire on updates that merely confirm an already-known
    negative. Do NOT edit this builder; add a new version instead.
    """
    target_text, chain_block, conf_line = _path_blocks(
        memory, path, target_id, new_confidence)
    return (
        "一个记忆系统中，某条事实刚被更新：\n"
        f"旧陈述：{old_value}\n"
        f"新陈述：{new_value}\n"
        f"{conf_line}\n"
        f"记忆中存在一条下游结论：{target_text}\n"
        f"{chain_block}\n"
        "请判断：这次更新在语义上是否实质动摇了该下游结论的有效性，"
        "使其需要重新评估？注意：\n"
        "1. 如果更新与该结论语义上基本无关，冲击应接近 0。\n"
        "2. 如果更新只是轻微波动、不改变原有整体判断，冲击应很低。\n"
        "3. 如果更新使该结论所依据的前提被否定或被明显削弱，冲击应很高。\n"
        "4. 即使更新没有降低源事实本身的可信度（例如进一步确认了某个"
        "不利条件），只要它使该结论更加站不住脚，冲击也应很高。\n\n"
        "只输出一个 0.0 到 1.0 之间的数字，不要输出其他内容。\n"
        "冲击强度："
    )


def build_prompt_v2(
    memory: MemoryState,
    path: List[str],
    target_id: str,
    old_value: str,
    new_value: str,
    new_confidence: Optional[float],
) -> str:
    """v2 gate prompt: comparative anchoring to fix confirmation over-fire.

    Developed against DEV cases only (see data/case_split.json). Core
    change vs v1: the score is anchored on the DELTA between old and new
    statements with respect to the downstream conclusion, so an update
    that merely restates/confirms what the conclusion already accounted
    for scores low -- while still firing when the update makes
    information that CONTRADICTS the conclusion more solid (the
    CO-Relation blind spot that motivated v1's rule 4).
    """
    target_text, chain_block, conf_line = _path_blocks(
        memory, path, target_id, new_confidence)
    return (
        "一个记忆系统中，某条事实刚被更新：\n"
        f"旧陈述：{old_value}\n"
        f"新陈述：{new_value}\n"
        f"{conf_line}\n"
        f"记忆中存在一条下游结论：{target_text}\n"
        f"{chain_block}\n"
        "请判断：相比旧陈述，这次更新是否为该下游结论带来了实质性的新信息，"
        "动摇了它的有效性，使其需要重新评估？\n"
        "先思考两个问题（不要输出思考过程）：\n"
        "A. 新旧陈述相比，对该下游结论的支持或反对程度是否发生了实质变化？\n"
        "B. 该下游结论在形成时是否已经把新陈述所说的情况考虑在内？\n"
        "打分规则：\n"
        "1. 更新与该结论语义上基本无关：冲击接近 0。\n"
        "2. 新陈述只是确认、复述旧陈述已表达的情况，或仅有轻微波动"
        "（即使内容是负面的，只要结论形成时已考虑到该情况）：冲击很低。\n"
        "3. 更新否定或明显削弱了该结论赖以成立的前提：冲击很高。\n"
        "4. 更新使与该结论相矛盾的信息变得更确凿（矛盾从存疑变为坐实）："
        "冲击很高。\n\n"
        "只输出一个 0.0 到 1.0 之间的数字，不要输出其他内容。\n"
        "冲击强度："
    )


def build_prompt_v3(
    memory: MemoryState,
    path: List[str],
    target_id: str,
    old_value: str,
    new_value: str,
    new_confidence: Optional[float],
) -> str:
    """v3 gate prompt: adds stance analysis for confirmation updates.

    Dev diagnosis of v2 showed one structural gap causing both new
    failures: whether a *confirmation* update shakes a downstream
    conclusion depends on the STANCE between them --
    * conclusion drawn FROM the confirmed situation (aligned): no new
      information, low impact (fixes v1's over-fire on pc500/syndep);
    * conclusion OPPOSED by the confirmed situation (contradiction
      hardens): high impact (keeps the CO-Relation blind-spot win
      that v2 lost).
    Also clarifies that withdrawing a trend is a substantive change,
    not a "minor fluctuation". Developed on DEV cases only.
    """
    target_text, chain_block, conf_line = _path_blocks(
        memory, path, target_id, new_confidence)
    return (
        "一个记忆系统中，某条事实刚被更新：\n"
        f"旧陈述：{old_value}\n"
        f"新陈述：{new_value}\n"
        f"{conf_line}\n"
        f"记忆中存在一条下游结论：{target_text}\n"
        f"{chain_block}\n"
        "请判断：这次更新是否实质动摇了该下游结论的有效性，使其需要"
        "重新评估？按以下步骤思考（不要输出思考过程）：\n"
        "第一步，相关性：更新所述情况与该结论之间是否存在实质的影响"
        "机制？若只是表面上同属一个主体但无实质影响，冲击接近 0。\n"
        "第二步，变化性质：新陈述相对旧陈述是以下哪种？\n"
        "  (a) 实质改变：原情况被否定、反转、明显削弱或明显加重；"
        "注意，若旧陈述是一个明确趋势或状态，而新陈述表明该趋势不再"
        "成立（即使变化幅度不大），这也属于实质改变。\n"
        "  (b) 确认复述：只是确认、重申或进一步坐实旧陈述已表达的情况。\n"
        "  (c) 轻微波动：只是小幅、偶发的变化，旧陈述的整体判断仍成立。\n"
        "第三步，方向关系（仅当第二步为 (b) 时需要）：该下游结论与被"
        "确认的情况是什么关系？\n"
        "  同向：结论本来就是基于该情况得出的，已把它考虑在内。\n"
        "  对立：该情况与结论相矛盾，结论的成立需要该情况不真或存疑。\n"
        "打分规则：\n"
        "1. 无实质影响机制：接近 0。\n"
        "2. (a) 实质改变且与结论相关：冲击很高。\n"
        "3. (b) 确认复述 + 同向：没有带来新信息，冲击很低。\n"
        "4. (b) 确认复述 + 对立：矛盾从存疑变为坐实，冲击很高。\n"
        "5. (c) 轻微波动：冲击很低。\n\n"
        "只输出一个 0.0 到 1.0 之间的数字，不要输出其他内容。\n"
        "冲击强度："
    )


PROMPT_BUILDERS: Dict[str, Callable] = {
    "v1": build_prompt_v1,
    "v2": build_prompt_v2,
    "v3": build_prompt_v3,
}


class SemanticGateEngine:
    """LLM semantic-judgment engine for re-evaluation marking.

    For each fact reachable from the updated source, builds the textual
    dependency path and asks the LLM for a semantic impact score in
    [0, 1]. Facts scoring >= tau_gate are marked REEVAL_REQUIRED.

    Attributes:
        llm: Any LLMInterface implementation (DeepSeekClient for real
            experiments, MockLLMClient for pipeline tests only).
        tau_gate: Impact threshold above which a fact is marked for
            re-evaluation (default 0.5).
        call_log: Audit trail of all LLM calls made by this engine.
    """

    def __init__(
        self,
        llm: LLMInterface,
        tau_gate: float = 0.5,
        prompt_builder: Callable = build_prompt_v1,
    ) -> None:
        self.llm = llm
        self.tau_gate = tau_gate
        self.prompt_builder = prompt_builder
        self.call_log: List[Dict] = []

    def propagate(
        self,
        memory: MemoryState,
        source_fact_id: str,
        old_value: str,
        new_value: str,
        new_confidence: Optional[float] = None,
    ) -> MemoryState:
        """Judge every descendant of the updated fact via the LLM.

        Args:
            memory: Memory state (will be mutated with REEVAL marks).
            source_fact_id: The fact that was just updated.
            old_value: The source fact's value BEFORE the update.
            new_value: The source fact's value AFTER the update.
            new_confidence: Confidence assigned to the new statement
                (included in the prompt as context; may be None).

        Returns:
            Updated memory with REEVAL_REQUIRED marks.

        Raises:
            RuntimeError: If no dependency graph is attached to memory.
            KeyError: If source_fact_id is not in the graph.
        """
        graph = memory.graph
        if graph is None:
            raise RuntimeError(
                "No dependency graph available. "
                "Set memory.graph before calling propagate()."
            )
        if source_fact_id not in graph.nodes:
            raise KeyError(
                f"Source fact '{source_fact_id}' not found in dependency graph."
            )

        # BFS to collect each reachable descendant with its path from source
        paths = collect_descendant_paths(graph, source_fact_id)

        for target_id, path in paths.items():
            score = self._judge_impact(
                memory, path, target_id, old_value, new_value, new_confidence
            )
            if score >= self.tau_gate:
                memory.mark_reeval(target_id)

        return memory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _judge_impact(
        self,
        memory: MemoryState,
        path: List[str],
        target_id: str,
        old_value: str,
        new_value: str,
        new_confidence: Optional[float],
    ) -> float:
        """Ask the LLM whether the update undermines the target fact."""
        prompt = self.prompt_builder(
            memory, path, target_id, old_value, new_value, new_confidence
        )
        response = self.llm.complete(prompt)
        score = self._parse_float(response, default=0.5)
        self.call_log.append({
            "target_fact_id": target_id,
            "path": list(path),
            "prompt": prompt,
            "raw_response": response,
            "parsed_score": score,
        })
        return score

    @staticmethod
    def _parse_float(text: str, default: float = 0.5) -> float:
        """Parse an impact score in [0, 1] from LLM text output."""
        try:
            match = re.search(r"0\.\d+|1\.0+|[01]", text)
            if match:
                return max(0.0, min(1.0, float(match.group())))
        except (ValueError, TypeError):
            pass
        return default


class StructuredGateEngine:
    """Decomposed semantic gate: discrete judgments + deterministic logic.

    Motivation (dev iteration v1->v3): a single scalar-impact prompt is
    brittle -- each rule added to fix one failure mode broke another
    (whack-a-mole). This engine instead asks the LLM three SMALL
    single-choice questions and combines them deterministically,
    mirroring the annotation guideline's own decision procedure:

      Q1 (per update, once): what kind of change is new vs old?
         A = substantive change (negated / reversed / clearly weakened
             or worsened / a stated trend no longer holds)
         B = confirmation (restates or further solidifies what the old
             statement already said)
         C = minor fluctuation (old overall judgment still holds)
      Q2 (per target): is there a real influence mechanism between the
         updated matter and the target conclusion? (yes/no)
      Q3 (per target, only if Q1 == B): is the target conclusion
         aligned with or opposed to the confirmed situation?

    Verdict: reeval iff  Q2 == yes  AND  (Q1 == A  OR  (Q1 == B AND
    Q3 == opposed)).

    Same honesty rules as SemanticGateEngine: prompts contain only fact
    texts (never annotated relation types / theta); every call is
    recorded in call_log.
    """

    def __init__(self, llm: LLMInterface) -> None:
        self.llm = llm
        self.call_log: List[Dict] = []

    def propagate(
        self,
        memory: MemoryState,
        source_fact_id: str,
        old_value: str,
        new_value: str,
        new_confidence: Optional[float] = None,
    ) -> MemoryState:
        graph = memory.graph
        if graph is None:
            raise RuntimeError(
                "No dependency graph available. "
                "Set memory.graph before calling propagate()."
            )
        if source_fact_id not in graph.nodes:
            raise KeyError(
                f"Source fact '{source_fact_id}' not found in dependency graph."
            )

        change = self._classify_change(old_value, new_value)
        paths = collect_descendant_paths(graph, source_fact_id)
        for target_id, path in paths.items():
            if change == "C":
                continue  # minor fluctuation never triggers reeval
            target_fact = memory.get_fact(target_id)
            target_text = target_fact.value if target_fact else target_id
            if not self._judge_relevance(old_value, new_value, target_text):
                continue
            if change == "A":
                memory.mark_reeval(target_id)
            elif self._judge_stance(new_value, target_text) == "opposed":
                memory.mark_reeval(target_id)
        return memory

    # ------------------------------------------------------------------
    # Discrete judgments
    # ------------------------------------------------------------------

    def _ask(self, kind: str, prompt: str) -> str:
        response = self.llm.complete(prompt)
        self.call_log.append(
            {"kind": kind, "prompt": prompt, "raw_response": response})
        return response.strip()

    def _classify_change(self, old_value: str, new_value: str) -> str:
        # Dev iteration note: "明显加重" originally sat in A, but the
        # annotation guideline treats same-direction intensification as a
        # REINFORCING update (shakes only opposed conclusions), so it
        # belongs in B. A is reserved for direction-changing updates.
        prompt = (
            "某条事实记录刚被更新：\n"
            f"旧陈述：{old_value}\n"
            f"新陈述：{new_value}\n\n"
            "新陈述相对旧陈述属于以下哪种？\n"
            "A. 方向性改变：原情况被否定、反转或明显削弱；或旧陈述所述"
            "的趋势/状态整体上不再成立（即使变化幅度不大）。注意："
            "单次、偶发的小事件不构成趋势/状态的改变，应归入 C。\n"
            "B. 确认或加强：确认、重申、进一步坐实旧陈述已表达的情况，"
            "或表明原情况沿同一方向进一步发展/加重。\n"
            "C. 轻微波动：只是小幅、偶发、不改变整体判断的变化（例如"
            "一次性的小插曲、单次轻微异常），旧陈述的整体判断仍然成立。\n\n"
            "只输出一个字母（A、B 或 C），不要输出其他内容。\n"
            "答案："
        )
        response = self._ask("change_type", prompt)
        match = re.search(r"[ABC]", response.upper())
        return match.group() if match else "A"  # fail-open to reeval

    def _judge_relevance(
        self, old_value: str, new_value: str, target_text: str
    ) -> bool:
        # Dev iteration note 2: an attempt to fix neutral-blocking cases by
        # requiring a "清晰的业务影响链条" over-corrected badly (clear
        # premise->conclusion links got rejected too), so it was rolled back
        # to the counterfactual phrasing below. See prompt_dev_run5.log.
        prompt = (
            f"已知情况：{new_value}\n"
            f"某条结论：{target_text}\n\n"
            "假如这一情况属实，它是否会影响对该结论的判断（无论支持还是"
            "反对，直接影响或通过合理的连锁影响都算）？\n"
            "仅当两者在业务上互不相干（例如只是同属一个公司但没有任何"
            "影响链条）时才答否。\n\n"
            "只输出“是”或“否”，不要输出其他内容。\n"
            "答案："
        )
        response = self._ask("relevance", prompt)
        return "否" not in response[:3]  # fail-open to relevant

    def _judge_stance(self, new_value: str, target_text: str) -> str:
        # Dev iteration note: the earlier aligned-definition included "本来
        # 就是基于该情况得出的", which dragged conclusion-formation history
        # into the judgment (contradicts-reinforce misread as aligned).
        # Stance must be judged on CONTENT alone.
        prompt = (
            f"已知情况：{new_value}\n"
            f"某条结论：{target_text}\n\n"
            "只看内容本身，这一情况与该结论是什么关系？\n"
            "同向：两者一致或互相印证，该情况成立会让该结论更站得住。\n"
            "对立：两者冲突，该情况成立会让该结论更难成立、面临阻碍或"
            "风险。\n\n"
            "只输出“同向”或“对立”，不要输出其他内容。\n"
            "答案："
        )
        response = self._ask("stance", prompt)
        return "opposed" if "对立" in response[:4] else "aligned"
