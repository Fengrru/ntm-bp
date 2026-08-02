"""Generate the CO relation-type test dataset with independent annotations.

Each scenario carries REAL semantic content and edge-level relation
types (supports / contradicts / neutral). Ground truth is manually
annotated by judging the semantics of each update against each
downstream fact — NOT by replaying the CO engine's gamma-modulation
rules (the previous version of this dataset was written to match CO's
own propagation behavior, which made the evaluation circular).

ANNOTATION POLICY (algorithm-independent)
-----------------------------------------
  * reeval: downstream facts whose validity is semantically cast into
    doubt by the update. Neutral edges usually block impact; a
    reinforcing update combined with a contradicts edge can still
    REQUIRE re-evaluation (the contradicted fact becomes more doubtful).
  * theta: annotated semantic impact strength in [0, 1].
  * rationale: recorded justification per case.

The dataset intentionally contains cases that are hard for ALL current
engines (e.g. reinforcing update on a contradicts edge produces no
confidence drop yet demands re-evaluation).

Usage:
    python scripts/generate_co_relation_test.py
"""

from __future__ import annotations

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

ANNOTATION_POLICY = (
    "Ground truth manually annotated per scenario from the semantics of the "
    "update and each downstream fact; relation types are edge metadata, not "
    "a label-generation rule. No engine output was used to produce labels."
)


def build_scenario(scenario_id, facts, deps, rel_types, update_fid, new_val,
                   new_conf, gt_reeval, theta, rationale):
    """Build a Scenario-compatible dict with annotation metadata."""
    return {
        "scenario_id": scenario_id,
        "facts": {fid: [v, conf] for fid, (v, conf) in facts.items()},
        "dependencies": [[p, c] for p, c in deps],
        "update_fact_id": update_fid,
        "new_value": new_val,
        "new_confidence": new_conf,
        "ground_truth_reeval": sorted(gt_reeval),
        "theta": theta,
        "annotation_rationale": rationale,
        "relation_types": {f"{p}__{c}": rt for (p, c), rt in zip(deps, rel_types)},
    }


def main():
    scenarios = []
    S = "supports"
    N = "neutral"
    C = "contradicts"

    # S1: strong contradiction through a supports chain -> full re-eval
    scenarios.append(build_scenario(
        "co_supports_chain",
        {"f1": ["核心工程师团队稳定", 0.9], "f2": ["项目按期交付有保障", 0.85],
         "f3": ["客户续约可能性高", 0.8]},
        [("f1", "f2"), ("f2", "f3")], [S, S],
        "f1", "核心工程师批量离职", 0.2, {"f2", "f3"}, 0.85,
        "团队稳定前提被否定，交付保障与续约判断均需重估。"))

    # S2: minor update over a neutral edge -> nothing to re-eval
    scenarios.append(build_scenario(
        "co_neutral_context",
        {"f1": ["总部办公区装修", 0.9], "f2": ["销售季度目标维持不变", 0.85],
         "f3": ["预期营收稳定", 0.8]},
        [("f1", "f2"), ("f2", "f3")], [N, S],
        "f1", "装修工期延长一个月", 0.85, set(), 0.1,
        "装修进度与销售目标仅弱相关（neutral），轻微更新不动摇下游。"))

    # S3: strong update but first edge is neutral -> still nothing
    scenarios.append(build_scenario(
        "co_neutral_blocks_strong_update",
        {"f1": ["行业展会照常举办", 0.9], "f2": ["市场部预算充足", 0.85],
         "f3": ["全年获客目标可达成", 0.8]},
        [("f1", "f2"), ("f2", "f3")], [N, S],
        "f1", "行业展会取消", 0.15, set(), 0.3,
        "展会与预算之间是 neutral 关系：展会取消不影响预算充足的判断及其下游。"))

    # S4: reinforcing update on a contradicts edge -> re-eval STILL required
    scenarios.append(build_scenario(
        "co_contradicts_reinforce",
        {"f1": ["数据出境监管收紧", 0.9], "f2": ["海外数据中心扩容计划推进", 0.85],
         "f3": ["海外市场服务能力提升", 0.8]},
        [("f1", "f2"), ("f2", "f3")], [C, S],
        "f1", "数据出境监管进一步收紧", 0.9, {"f2", "f3"}, 0.7,
        "强化型更新叠加 contradicts 边：监管更紧使被其反对的扩容计划及下游更存疑，"
        "尽管源事实置信度几乎未变。"))

    # S5: deep supports chain with a hard contradiction -> full re-eval
    scenarios.append(build_scenario(
        "co_3hop_supports",
        {"f1": ["核心原料供应稳定", 0.9], "f2": ["生产计划可执行", 0.85],
         "f3": ["订单可按期交付", 0.8], "f4": ["大客户合同可续签", 0.75]},
        [("f1", "f2"), ("f2", "f3"), ("f3", "f4")], [S, S, S],
        "f1", "核心原料断供", 0.1, {"f2", "f3", "f4"}, 0.9,
        "断供否定供应稳定，生产、交付、续签整条支持链全部存疑。"))

    # S6: neutral tail edge truncates the impact -> partial re-eval
    scenarios.append(build_scenario(
        "co_3hop_neutral_tail",
        {"f1": ["品牌宣传片上线", 0.9], "f2": ["官网访问量上升", 0.85],
         "f3": ["销售线索增加", 0.8], "f4": ["IT 运维排班不变", 0.75]},
        [("f1", "f2"), ("f2", "f3"), ("f3", "f4")], [S, S, N],
        "f1", "宣传片因版权问题下架", 0.2, {"f2", "f3"}, 0.6,
        "访问量与线索判断失效；IT 排班与线索仅 neutral 相关，不受影响（部分集）。"))

    # S7: pressure relieved -> contradicted conclusion becomes MORE stable
    scenarios.append(build_scenario(
        "co_mixed_supports_contradicts",
        {"f1": ["竞品发布重大更新", 0.9], "f2": ["我方面临市场压力", 0.85],
         "f3": ["维持当前定价", 0.8]},
        [("f1", "f2"), ("f2", "f3")], [S, C],
        "f1", "竞品更新遭遇大量差评", 0.75, {"f2"}, 0.5,
        "压力判断需重估；压力若减弱，被其反对的维持定价反而更稳固，不需重估（部分集）。"))

    # S8: mild reinforcing update over supports chain -> nothing
    scenarios.append(build_scenario(
        "co_weak_update_supports",
        {"f1": ["供应商交付记录良好", 0.9], "f2": ["采购计划维持", 0.85],
         "f3": ["库存周转正常", 0.8]},
        [("f1", "f2"), ("f2", "f3")], [S, S],
        "f1", "供应商本月一次轻微延迟", 0.8, set(), 0.2,
        "单次轻微延迟不动摇交付记录良好的整体判断，下游无需重估。"))

    # Save as JSON
    output_path = os.path.join(DATA_DIR, "co_relation_test.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "dataset": "CO-Relation",
                "instances": len(scenarios),
                "annotation_policy": ANNOTATION_POLICY,
            },
            "scenarios": scenarios,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(scenarios)} scenarios to {output_path}")


if __name__ == "__main__":
    main()
