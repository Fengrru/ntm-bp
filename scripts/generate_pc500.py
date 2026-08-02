"""Generate the PC-500 synthetic dataset as a JSON file.

Produces 500 partial-contradiction scenario instances from 6 templates
x 3 update values = 18 annotated cases, with per-instance jitter on
initial confidences so instances are not exact copies.

GROUND TRUTH POLICY (algorithm-independent)
-------------------------------------------
The ground truth is NOT produced by any propagation engine. Each
(template, update_value) pair is manually annotated by judging, case by
case, whether the update semantically casts doubt on the validity of
each downstream conclusion:

  * reeval: the set of downstream facts whose validity becomes
    questionable and therefore requires re-evaluation. Reinforcing or
    semantically minor updates yield an EMPTY set even though the
    dependency graph would mechanically reach all descendants.
  * theta: annotated semantic impact strength in [0, 1] (NOT derived
    arithmetically from 1 - new_confidence).
  * rationale: a short justification recorded with each annotation.

The annotator may override noisy graph edges (i.e., a descendant in the
graph need not be in the reeval set if the update does not actually
undermine it). ground_truth_belief is intentionally absent: belief is an
internal quantity of the algorithms under test and has no independent
ground truth.

CONFIDENCE COHERENCE: new_confidence is the assertion confidence of the
new statement itself. Reinforcing/confirming statements carry high
confidence (0.85-0.9); contradicting statements may carry low confidence
(e.g. an unverified rumor). This replaces the earlier incoherent scheme
where confidence was reverse-engineered from impact (1 - theta).

Usage:
    python scripts/generate_pc500.py
"""

from __future__ import annotations

import json
import os
import random


# ------------------------------------------------------------------
# Scenario templates
# ------------------------------------------------------------------

SCENARIO_TEMPLATES = [
    {
        "template_id": "chain_business",
        "facts": [
            ("f1", "Acme Corp 信誉良好", 0.9),
            ("f2", "Contract A 低风险", 0.85),
            ("f3", "推荐投资", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3")],
        "update_id": "f1",
        "new_values": [
            ("Acme Corp 被调查", 0.4),
            ("Acme Corp 破产传闻", 0.2),
            ("Acme Corp 营收下滑", 0.5),
        ],
    },
    {
        "template_id": "chain_market",
        "facts": [
            ("f1", "市场处于牛市", 0.9),
            ("f2", "投资策略偏进取", 0.85),
            ("f3", "高仓位操作", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3")],
        "update_id": "f1",
        "new_values": [
            ("市场转为熊市", 0.1),
            ("市场震荡整理", 0.4),
            ("市场小幅回调", 0.85),
        ],
    },
    {
        "template_id": "chain_supplier",
        "facts": [
            ("f1", "供应商 A 质量可靠", 0.9),
            ("f2", "使用供应商 A", 0.85),
            ("f3", "降低生产成本", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3")],
        "update_id": "f1",
        "new_values": [
            ("供应商 A 偶尔延迟交货", 0.85),
            ("供应商 A 出现质量问题", 0.3),
            ("供应商 A 产能不足", 0.5),
        ],
    },
    {
        "template_id": "branch_tech",
        "facts": [
            ("f1", "技术方案 A 成熟稳定", 0.9),
            ("f2", "采用技术方案 A", 0.85),
            ("f3", "项目进度有保障", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f1", "f3")],
        "update_id": "f1",
        "new_values": [
            ("技术方案 A 存在安全漏洞", 0.2),
            ("技术方案 A 需要额外定制", 0.5),
            ("技术方案 A 已被淘汰", 0.1),
        ],
    },
    {
        "template_id": "chain_regulation",
        "facts": [
            ("f1", "新法规即将出台", 0.9),
            ("f2", "产品需要合规调整", 0.85),
            ("f3", "延迟产品上线", 0.8),
            ("f4", "影响季度营收", 0.75),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3"), ("f3", "f4")],
        "update_id": "f1",
        "new_values": [
            ("新法规推迟实施", 0.3),
            ("新法规要求更严格", 0.85),
            ("新法规已取消", 0.1),
        ],
    },
    {
        "template_id": "branch_hr",
        "facts": [
            ("f1", "核心员工提出离职", 0.9),
            ("f2", "招聘新员工", 0.85),
            ("f3", "项目进度受影响", 0.8),
            ("f4", "加班赶工应对", 0.75),
        ],
        "dependencies": [("f1", "f2"), ("f1", "f3"), ("f3", "f4")],
        "update_id": "f1",
        "new_values": [
            ("核心员工撤销离职申请", 0.1),
            ("核心员工被加薪留任", 0.3),
            ("核心员工开始工作交接", 0.9),
        ],
    },
]


# ------------------------------------------------------------------
# Independent semantic annotations (manual, algorithm-agnostic)
# Key: (template_id, new_value)
# Value: reeval set, semantic impact theta, rationale
# ------------------------------------------------------------------

GROUND_TRUTH_ANNOTATIONS = {
    # --- chain_business: f1 信誉良好 -> f2 合同低风险 -> f3 推荐投资 ---
    ("chain_business", "Acme Corp 被调查"): {
        "reeval": ["f2", "f3"],
        "theta": 0.7,
        "rationale": "被调查直接动摇信誉前提，低风险判断与投资推荐都需重新评估。",
    },
    ("chain_business", "Acme Corp 破产传闻"): {
        "reeval": ["f2", "f3"],
        "theta": 0.85,
        "rationale": "破产传闻严重冲击信誉，下游合同风险与投资结论全部存疑。",
    },
    ("chain_business", "Acme Corp 营收下滑"): {
        "reeval": ["f2", "f3"],
        "theta": 0.5,
        "rationale": "营收下滑削弱但不完全否定信誉，风险与投资判断需谨慎复核。",
    },
    # --- chain_market: f1 牛市 -> f2 进取策略 -> f3 高仓位 ---
    ("chain_market", "市场转为熊市"): {
        "reeval": ["f2", "f3"],
        "theta": 0.9,
        "rationale": "牛熊反转直接否定前提，进取策略与高仓位必须重估。",
    },
    ("chain_market", "市场震荡整理"): {
        "reeval": ["f2", "f3"],
        "theta": 0.55,
        "rationale": "震荡使牛市判断存疑，激进策略与仓位需重新考量。",
    },
    ("chain_market", "市场小幅回调"): {
        "reeval": [],
        "theta": 0.25,
        "rationale": "牛市中的小幅回调属正常波动，不动摇牛市前提及下游策略。",
    },
    # --- chain_supplier: f1 质量可靠 -> f2 使用供应商A -> f3 降成本 ---
    ("chain_supplier", "供应商 A 偶尔延迟交货"): {
        "reeval": [],
        "theta": 0.2,
        "rationale": "偶尔延迟是交期问题而非质量问题，不推翻质量可靠及其下游。",
    },
    ("chain_supplier", "供应商 A 出现质量问题"): {
        "reeval": ["f2", "f3"],
        "theta": 0.75,
        "rationale": "质量问题直接否定前提，是否继续使用及成本结论均存疑。",
    },
    ("chain_supplier", "供应商 A 产能不足"): {
        "reeval": ["f2", "f3"],
        "theta": 0.5,
        "rationale": "产能不足虽非质量问题，但影响继续使用的可行性与成本优势。",
    },
    # --- branch_tech: f1 方案成熟 -> f2 采用方案A; f1 -> f3 进度有保障 ---
    ("branch_tech", "技术方案 A 存在安全漏洞"): {
        "reeval": ["f2", "f3"],
        "theta": 0.8,
        "rationale": "安全漏洞否定成熟稳定，采用决定与进度保障均需重估。",
    },
    ("branch_tech", "技术方案 A 需要额外定制"): {
        "reeval": ["f3"],
        "theta": 0.45,
        "rationale": "需定制不否定采用方案本身，但额外工作量动摇进度保障（部分集）。",
    },
    ("branch_tech", "技术方案 A 已被淘汰"): {
        "reeval": ["f2", "f3"],
        "theta": 0.9,
        "rationale": "方案被淘汰彻底否定前提，两个下游结论全部失效。",
    },
    # --- chain_regulation: f1 新法规出台 -> f2 合规调整 -> f3 延迟上线 -> f4 影响营收 ---
    ("chain_regulation", "新法规推迟实施"): {
        "reeval": ["f2", "f3", "f4"],
        "theta": 0.7,
        "rationale": "推迟实施改变时间前提，调整、延迟与营收影响的必要性均存疑。",
    },
    ("chain_regulation", "新法规要求更严格"): {
        "reeval": [],
        "theta": 0.25,
        "rationale": "更严格是强化型更新：合规调整、延迟上线、营收受影响的结论方向仍成立。",
    },
    ("chain_regulation", "新法规已取消"): {
        "reeval": ["f2", "f3", "f4"],
        "theta": 0.95,
        "rationale": "法规取消彻底移除前提，整条推理链全部作废。",
    },
    # --- branch_hr: f1 员工离职 -> f2 招聘; f1 -> f3 进度受影响 -> f4 加班应对 ---
    ("branch_hr", "核心员工撤销离职申请"): {
        "reeval": ["f2", "f3", "f4"],
        "theta": 0.9,
        "rationale": "撤销离职直接否定前提，招聘、进度影响与加班应对全部失去依据。",
    },
    ("branch_hr", "核心员工被加薪留任"): {
        "reeval": ["f2", "f3", "f4"],
        "theta": 0.85,
        "rationale": "留任等同于离职前提被推翻，下游三项应对措施均需撤回或重估。",
    },
    ("branch_hr", "核心员工开始工作交接"): {
        "reeval": [],
        "theta": 0.1,
        "rationale": "开始交接是离职前提被证实的强化型更新，下游应对依然成立。",
    },
}

ANNOTATION_POLICY = (
    "Ground truth manually annotated per (template, update_value): reeval = "
    "downstream facts whose validity is semantically cast into doubt "
    "(reinforcing/minor updates -> empty set); theta = annotated semantic "
    "impact in [0,1], independent of any propagation algorithm; annotator may "
    "override noisy graph edges. No engine output was used to produce labels."
)

# Per-instance jitter applied to initial confidences (kept small so the
# annotated semantics remain valid for every instance).
JITTER = 0.05


def generate_scenario(
    template: dict,
    new_value: str,
    new_confidence: float,
    scenario_index: int,
    rng: random.Random,
) -> dict:
    """Generate a single scenario instance with jittered confidences."""
    annotation = GROUND_TRUTH_ANNOTATIONS[(template["template_id"], new_value)]

    facts_dict = {}
    for fid, val, conf in template["facts"]:
        jittered = round(
            min(0.99, max(0.05, conf + rng.uniform(-JITTER, JITTER))), 4
        )
        facts_dict[fid] = [val, jittered]

    return {
        "scenario_id": f"pc_syn_{scenario_index:04d}",
        "template_id": template["template_id"],
        "facts": facts_dict,
        "dependencies": [[p, c] for p, c in template["dependencies"]],
        "update_fact_id": template["update_id"],
        "new_value": new_value,
        "new_confidence": new_confidence,
        "ground_truth_reeval": sorted(annotation["reeval"]),
        "ground_truth_values": {template["update_id"]: new_value},
        "theta": annotation["theta"],
        "annotation_rationale": annotation["rationale"],
    }


def main():
    rng = random.Random(42)
    scenarios = []

    # Round-robin over the 18 annotated cases until 500 instances exist.
    cases = [
        (template, new_value, new_confidence)
        for template in SCENARIO_TEMPLATES
        for new_value, new_confidence in template["new_values"]
    ]
    index = 0
    while index < 500:
        for template, new_value, new_confidence in cases:
            if index >= 500:
                break
            scenarios.append(
                generate_scenario(template, new_value, new_confidence, index + 1, rng)
            )
            index += 1

    rng.shuffle(scenarios)

    # Write output
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pc500.json")

    payload = {
        "metadata": {
            "dataset": "PC-500",
            "unique_cases": len(cases),
            "instances": len(scenarios),
            "annotation_policy": ANNOTATION_POLICY,
        },
        "scenarios": scenarios,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(scenarios)} scenarios ({len(cases)} unique annotated cases) -> {output_path}")

    # Stats
    theta_values = [s["theta"] for s in scenarios]
    print(f"  theta range: [{min(theta_values):.2f}, {max(theta_values):.2f}]")
    print(f"  avg theta: {sum(theta_values) / len(theta_values):.2f}")
    empty_gt = sum(1 for s in scenarios if not s["ground_truth_reeval"])
    print(f"  scenarios with empty reeval GT: {empty_gt} ({empty_gt / len(scenarios):.0%})")
    chain_count = sum(1 for s in scenarios if len(s["dependencies"]) == len(s["facts"]) - 1)
    branch_count = len(scenarios) - chain_count
    print(f"  chain topologies: {chain_count}, branch topologies: {branch_count}")


if __name__ == "__main__":
    main()
