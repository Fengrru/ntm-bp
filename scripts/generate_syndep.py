"""Generate the SynDep (Synthetic Dependency) benchmark as a JSON file.

SynDep produces 200 scenario instances from 10 templates x 3 update
values = 30 annotated cases, featuring multi-premise convergence
patterns (4-6 facts per scenario). Instances get per-instance jitter on
initial confidences so they are not exact copies.

NOTE: This dataset was previously (mis)named "LongMemEval". It is fully
synthetic and unrelated to the public LongMemEval benchmark; it has been
renamed to SynDep to avoid misrepresentation.

GROUND TRUTH POLICY (algorithm-independent)
-------------------------------------------
The ground truth is NOT produced by any propagation engine. Each
(template, update_value) pair is manually annotated by judging, case by
case, whether the update semantically casts doubt on the validity of
each downstream conclusion:

  * reeval: the set of downstream facts whose validity becomes
    questionable. Reinforcing or semantically minor updates yield an
    EMPTY set; partial impact yields a strict subset of descendants.
  * theta: annotated semantic impact strength in [0, 1] (NOT derived
    arithmetically from 1 - new_confidence).
  * rationale: a short justification recorded with each annotation.

The annotator may override noisy graph edges. ground_truth_belief is
intentionally absent: belief is an internal quantity of the algorithms
under test and has no independent ground truth.

CONFIDENCE COHERENCE: new_confidence is the assertion confidence of the
new statement itself. Reinforcing/confirming statements carry high
confidence (0.85-0.9); contradicting statements may carry low confidence
(e.g. an unverified rumor). This replaces the earlier incoherent scheme
where confidence was reverse-engineered from impact (1 - theta).

Usage:
    python scripts/generate_syndep.py
"""

from __future__ import annotations

import json
import os
import random


# ------------------------------------------------------------------
# Scenario templates (4-6 facts with multi-premise convergence)
# ------------------------------------------------------------------

SCENARIO_TEMPLATES = [
    # === 4-fact templates ===
    {
        "template_id": "converge_investment",
        "facts": [
            ("f1", "公司营收连续增长", 0.9),
            ("f2", "行业前景看好", 0.85),
            ("f3", "管理层经验丰富", 0.8),
            ("f4", "推荐增持股票", 0.75),
        ],
        # f1, f2, f3 all support f4 (three premises -> one conclusion)
        "dependencies": [("f1", "f4"), ("f2", "f4"), ("f3", "f4")],
        "update_id": "f1",
        "new_values": [
            ("公司营收大幅下滑", 0.2),
            ("公司营收增速放缓", 0.5),
            ("公司营收持平预期", 0.85),
        ],
    },
    {
        "template_id": "converge_product",
        "facts": [
            ("f1", "市场需求旺盛", 0.9),
            ("f2", "研发团队能力强", 0.85),
            ("f3", "资金充裕", 0.8),
            ("f4", "新产品值得投资", 0.75),
        ],
        "dependencies": [("f1", "f4"), ("f2", "f4"), ("f3", "f4")],
        "update_id": "f2",
        "new_values": [
            ("研发团队核心成员离职", 0.2),
            ("研发团队需外部招聘", 0.5),
            ("研发团队得到加强", 0.9),
        ],
    },
    {
        "template_id": "chain_branch_legal",
        "facts": [
            ("f1", "合同条款存在歧义", 0.9),
            ("f2", "双方理解不一致", 0.85),
            ("f3", "法律团队介入", 0.8),
            ("f4", "诉讼风险上升", 0.75),
        ],
        # Chain: f1 -> f2 -> f4, with branch f2 -> f3
        "dependencies": [("f1", "f2"), ("f2", "f3"), ("f2", "f4")],
        "update_id": "f1",
        "new_values": [
            ("合同条款已澄清", 0.1),
            ("合同条款需重新谈判", 0.4),
            ("合同条款存在争议", 0.85),
        ],
    },
    # === 5-fact templates ===
    {
        "template_id": "converge_multi_chain",
        "facts": [
            ("f1", "经济数据表现良好", 0.9),
            ("f2", "央行维持宽松政策", 0.85),
            ("f3", "信贷环境宽松", 0.8),
            ("f4", "企业融资成本降低", 0.75),
            ("f5", "建议扩大投资规模", 0.7),
        ],
        # f1->f3, f2->f3, f3->f4, f4->f5
        "dependencies": [("f1", "f3"), ("f2", "f3"), ("f3", "f4"), ("f4", "f5")],
        "update_id": "f1",
        "new_values": [
            ("经济数据不及预期", 0.2),
            ("经济数据符合预期", 0.85),
            ("经济数据超预期", 0.9),
        ],
    },
    {
        "template_id": "converge_tech_decision",
        "facts": [
            ("f1", "现有系统维护成本高", 0.9),
            ("f2", "新技术方案成熟度足够", 0.85),
            ("f3", "团队具备学习能力", 0.8),
            ("f4", "迁移风险可控", 0.75),
            ("f5", "启动技术架构升级", 0.7),
        ],
        # f1->f5, f2->f4, f2->f5, f3->f4, f4->f5
        "dependencies": [("f1", "f5"), ("f2", "f4"), ("f2", "f5"), ("f3", "f4"), ("f4", "f5")],
        "update_id": "f2",
        "new_values": [
            ("新技术方案存在安全漏洞", 0.2),
            ("新技术方案学习曲线陡峭", 0.5),
            ("新技术方案优势明显", 0.9),
        ],
    },
    {
        "template_id": "deep_chain_market",
        "facts": [
            ("f1", "原材料价格上涨", 0.9),
            ("f2", "生产成本增加", 0.85),
            ("f3", "产品提价压力大", 0.8),
            ("f4", "市场份额可能下滑", 0.75),
            ("f5", "利润预期调低", 0.7),
        ],
        # Deep chain: f1->f2->f3->f4->f5
        "dependencies": [("f1", "f2"), ("f2", "f3"), ("f3", "f4"), ("f4", "f5")],
        "update_id": "f1",
        "new_values": [
            ("原材料价格回落至正常水平", 0.1),
            ("原材料价格小幅波动", 0.4),
            ("原材料价格持续攀升", 0.9),
        ],
    },
    # === 6-fact templates ===
    {
        "template_id": "converge_risk_assessment",
        "facts": [
            ("f1", "合作方财务状况恶化", 0.9),
            ("f2", "合作方存在合规问题", 0.85),
            ("f3", "市场环境不利", 0.8),
            ("f4", "内部风控能力不足", 0.75),
            ("f5", "合同保护条款有限", 0.7),
            ("f6", "建议终止合作关系", 0.65),
        ],
        # f1->f6, f2->f6, f3->f6, f4->f5, f5->f6
        "dependencies": [
            ("f1", "f6"), ("f2", "f6"), ("f3", "f6"),
            ("f4", "f5"), ("f5", "f6"),
        ],
        "update_id": "f1",
        "new_values": [
            ("合作方财务状况改善", 0.1),
            ("合作方财务问题可控", 0.4),
            ("合作方财务问题加剧", 0.9),
        ],
    },
    {
        "template_id": "branch_merge_project",
        "facts": [
            ("f1", "客户需求变更频繁", 0.9),
            ("f2", "开发人员加班过多", 0.85),
            ("f3", "代码质量下降", 0.8),
            ("f4", "测试覆盖率不足", 0.75),
            ("f5", "上线日期临近", 0.7),
            ("f6", "项目存在延期风险", 0.65),
        ],
        # f1->f2, f2->f3, f2->f4, f1->f5, f3->f6, f4->f6, f5->f6
        "dependencies": [
            ("f1", "f2"), ("f2", "f3"), ("f2", "f4"),
            ("f1", "f5"), ("f3", "f6"), ("f4", "f6"), ("f5", "f6"),
        ],
        "update_id": "f1",
        "new_values": [
            ("客户需求已冻结", 0.1),
            ("客户需求小幅调整", 0.5),
            ("客户需求仍在增加", 0.9),
        ],
    },
    {
        "template_id": "converge_supply_chain",
        "facts": [
            ("f1", "主要供应商产能下降", 0.9),
            ("f2", "物流成本上升", 0.85),
            ("f3", "库存水平偏低", 0.8),
            ("f4", "生产排期紧张", 0.75),
            ("f5", "交付延迟风险高", 0.7),
        ],
        "dependencies": [("f1", "f3"), ("f2", "f5"), ("f3", "f4"), ("f4", "f5")],
        "update_id": "f1",
        "new_values": [
            ("主要供应商产能恢复", 0.15),
            ("主要供应商产能部分恢复", 0.45),
            ("主要供应商产能继续恶化", 0.9),
        ],
    },
    {
        "template_id": "converge_regulatory",
        "facts": [
            ("f1", "监管新规征求意见稿发布", 0.9),
            ("f2", "行业自律标准提高", 0.85),
            ("f3", "现有产品不符合新要求", 0.8),
            ("f4", "需要大规模调整产品线", 0.75),
            ("f5", "合规成本大幅增加", 0.7),
        ],
        "dependencies": [("f1", "f3"), ("f2", "f3"), ("f3", "f4"), ("f4", "f5")],
        "update_id": "f1",
        "new_values": [
            ("监管新规被推迟实施", 0.2),
            ("监管新规要求比预期宽松", 0.5),
            ("监管新规要求比预期严格", 0.9),
        ],
    },
]


# ------------------------------------------------------------------
# Independent semantic annotations (manual, algorithm-agnostic)
# Key: (template_id, new_value)
# Value: reeval set, semantic impact theta, rationale
# ------------------------------------------------------------------

GROUND_TRUTH_ANNOTATIONS = {
    # --- converge_investment: f1,f2,f3 -> f4 推荐增持 ---
    ("converge_investment", "公司营收大幅下滑"): {
        "reeval": ["f4"],
        "theta": 0.85,
        "rationale": "营收大幅下滑严重削弱增持理由，尽管 f2/f3 仍成立，结论必须重估。",
    },
    ("converge_investment", "公司营收增速放缓"): {
        "reeval": ["f4"],
        "theta": 0.5,
        "rationale": "增速放缓部分削弱三前提之一，增持结论需谨慎复核。",
    },
    ("converge_investment", "公司营收持平预期"): {
        "reeval": [],
        "theta": 0.25,
        "rationale": "持平预期属轻微更新，另两前提未变，增持结论不需强制重估。",
    },
    # --- converge_product: f1,f2,f3 -> f4 值得投资（更新 f2） ---
    ("converge_product", "研发团队核心成员离职"): {
        "reeval": ["f4"],
        "theta": 0.8,
        "rationale": "核心成员离职直接否定研发能力前提，投资结论存疑。",
    },
    ("converge_product", "研发团队需外部招聘"): {
        "reeval": ["f4"],
        "theta": 0.5,
        "rationale": "需外部招聘暗示现有能力有缺口，投资结论需复核。",
    },
    ("converge_product", "研发团队得到加强"): {
        "reeval": [],
        "theta": 0.1,
        "rationale": "强化型更新：研发能力前提被增强，结论无需重估。",
    },
    # --- chain_branch_legal: f1 歧义 -> f2 理解不一致 -> f3 法律介入 / f4 诉讼风险 ---
    ("chain_branch_legal", "合同条款已澄清"): {
        "reeval": ["f2", "f3", "f4"],
        "theta": 0.9,
        "rationale": "歧义前提被消除，理解不一致、法律介入与诉讼风险全部失去依据。",
    },
    ("chain_branch_legal", "合同条款需重新谈判"): {
        "reeval": ["f4"],
        "theta": 0.45,
        "rationale": "进入谈判轨道后理解不一致与法律介入仍然成立，但诉讼风险走向存疑（部分集）。",
    },
    ("chain_branch_legal", "合同条款存在争议"): {
        "reeval": [],
        "theta": 0.2,
        "rationale": "存在争议与存在歧义语义相近，属确认型更新，下游不需重估。",
    },
    # --- converge_multi_chain: f1,f2 -> f3 信贷宽松 -> f4 融资成本低 -> f5 扩大投资 ---
    ("converge_multi_chain", "经济数据不及预期"): {
        "reeval": ["f3", "f4", "f5"],
        "theta": 0.7,
        "rationale": "数据不及预期动摇宽松环境判断，整条下游链均需重估。",
    },
    ("converge_multi_chain", "经济数据符合预期"): {
        "reeval": [],
        "theta": 0.2,
        "rationale": "符合预期属中性确认，前提未被动摇。",
    },
    ("converge_multi_chain", "经济数据超预期"): {
        "reeval": [],
        "theta": 0.15,
        "rationale": "超预期是强化型更新，下游结论方向不变。",
    },
    # --- converge_tech_decision: 更新 f2 成熟度（f2->f4, f2->f5, f4->f5） ---
    ("converge_tech_decision", "新技术方案存在安全漏洞"): {
        "reeval": ["f4", "f5"],
        "theta": 0.85,
        "rationale": "安全漏洞否定成熟度，迁移风险可控与启动升级均存疑。",
    },
    ("converge_tech_decision", "新技术方案学习曲线陡峭"): {
        "reeval": ["f4", "f5"],
        "theta": 0.5,
        "rationale": "学习曲线陡峭增大迁移不确定性，风险评估与升级决定需复核。",
    },
    ("converge_tech_decision", "新技术方案优势明显"): {
        "reeval": [],
        "theta": 0.1,
        "rationale": "强化型更新，前提被增强，下游无需重估。",
    },
    # --- deep_chain_market: f1 原材料涨 -> f2 成本 -> f3 提价 -> f4 份额 -> f5 利润 ---
    ("deep_chain_market", "原材料价格回落至正常水平"): {
        "reeval": ["f2", "f3", "f4", "f5"],
        "theta": 0.9,
        "rationale": "价格回落否定涨价前提，成本、提价、份额、利润整条链作废。",
    },
    ("deep_chain_market", "原材料价格小幅波动"): {
        "reeval": ["f2", "f3", "f4", "f5"],
        "theta": 0.6,
        "rationale": "小幅波动使持续上涨前提存疑，深链下游结论均需复核。",
    },
    ("deep_chain_market", "原材料价格持续攀升"): {
        "reeval": [],
        "theta": 0.15,
        "rationale": "持续攀升是强化型更新，链上所有结论依然成立。",
    },
    # --- converge_risk_assessment: f1 财务恶化 -> f6 终止合作（f2,f3,f5 亦 -> f6） ---
    ("converge_risk_assessment", "合作方财务状况改善"): {
        "reeval": ["f6"],
        "theta": 0.8,
        "rationale": "关键负面前提被逆转，终止合作的建议需要重估（其余前提不受影响）。",
    },
    ("converge_risk_assessment", "合作方财务问题可控"): {
        "reeval": ["f6"],
        "theta": 0.55,
        "rationale": "财务问题可控弱化终止合作的主要依据，结论需复核。",
    },
    ("converge_risk_assessment", "合作方财务问题加剧"): {
        "reeval": [],
        "theta": 0.1,
        "rationale": "强化型更新：终止合作的结论被进一步支持。",
    },
    # --- branch_merge_project: f1 需求变更频繁 -> f2 加班 -> f3/f4; f1->f5; f3,f4,f5 -> f6 ---
    ("branch_merge_project", "客户需求已冻结"): {
        "reeval": ["f2", "f3", "f4", "f6"],
        "theta": 0.8,
        "rationale": "需求冻结否定变更前提，加班、质量、测试及延期风险需重估；"
        "f5 上线日期是日历事实，不因需求冻结而失效（标注者覆盖噪声边）。",
    },
    ("branch_merge_project", "客户需求小幅调整"): {
        "reeval": ["f2", "f3", "f4", "f6"],
        "theta": 0.5,
        "rationale": "小幅调整使'频繁变更'的严重性存疑，压力链下游需复核；f5 同上不受影响。",
    },
    ("branch_merge_project", "客户需求仍在增加"): {
        "reeval": [],
        "theta": 0.1,
        "rationale": "强化型更新：变更频繁前提被证实，下游结论依然成立。",
    },
    # --- converge_supply_chain: f1 产能下降 -> f3 库存低 -> f4 排期紧 -> f5 延迟风险（f2->f5） ---
    ("converge_supply_chain", "主要供应商产能恢复"): {
        "reeval": ["f3", "f4", "f5"],
        "theta": 0.85,
        "rationale": "产能恢复否定下降前提，库存、排期、延迟风险整条链需重估。",
    },
    ("converge_supply_chain", "主要供应商产能部分恢复"): {
        "reeval": ["f3", "f4", "f5"],
        "theta": 0.5,
        "rationale": "部分恢复使紧张程度存疑，下游链需复核。",
    },
    ("converge_supply_chain", "主要供应商产能继续恶化"): {
        "reeval": [],
        "theta": 0.1,
        "rationale": "强化型更新：产能下降前提被证实，下游无需重估。",
    },
    # --- converge_regulatory: f1 新规征求意见 -> f3 产品不合规 -> f4 调整产品线 -> f5 成本增加 ---
    ("converge_regulatory", "监管新规被推迟实施"): {
        "reeval": ["f4", "f5"],
        "theta": 0.5,
        "rationale": "推迟实施不改变不合规事实（f3 仍成立），但大规模调整的紧迫性与成本时间线存疑（部分集）。",
    },
    ("converge_regulatory", "监管新规要求比预期宽松"): {
        "reeval": ["f3", "f4", "f5"],
        "theta": 0.65,
        "rationale": "要求放宽使不合规判断本身存疑，整条下游链需重估。",
    },
    ("converge_regulatory", "监管新规要求比预期严格"): {
        "reeval": [],
        "theta": 0.15,
        "rationale": "强化型更新：不合规、调整与成本增加的结论方向不变。",
    },
}

ANNOTATION_POLICY = (
    "Ground truth manually annotated per (template, update_value): reeval = "
    "downstream facts whose validity is semantically cast into doubt "
    "(reinforcing/minor updates -> empty set; partial impact -> strict "
    "subset); theta = annotated semantic impact in [0,1], independent of any "
    "propagation algorithm; annotator may override noisy graph edges. No "
    "engine output was used to produce labels."
)

# Per-instance jitter applied to initial confidences.
JITTER = 0.05


def generate_scenario(
    template: dict,
    new_value: str,
    new_confidence: float,
    scenario_index: int,
    rng: random.Random,
) -> dict:
    """Generate a single SynDep scenario instance with jittered confidences."""
    annotation = GROUND_TRUTH_ANNOTATIONS[(template["template_id"], new_value)]

    facts_dict = {}
    for fid, val, conf in template["facts"]:
        jittered = round(
            min(0.99, max(0.05, conf + rng.uniform(-JITTER, JITTER))), 4
        )
        facts_dict[fid] = [val, jittered]

    return {
        "scenario_id": f"syndep_{scenario_index:04d}",
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

    # Round-robin over the 30 annotated cases until 200 instances exist.
    cases = [
        (template, new_value, new_confidence)
        for template in SCENARIO_TEMPLATES
        for new_value, new_confidence in template["new_values"]
    ]
    index = 0
    while index < 200:
        for template, new_value, new_confidence in cases:
            if index >= 200:
                break
            scenarios.append(
                generate_scenario(template, new_value, new_confidence, index + 1, rng)
            )
            index += 1

    rng.shuffle(scenarios)

    # Write output
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "syndep.json")

    payload = {
        "metadata": {
            "dataset": "SynDep",
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
    total_edges = sum(len(s["dependencies"]) for s in scenarios)
    print(f"  avg edges per scenario: {total_edges / len(scenarios):.1f}")
    fact_counts = [len(s["facts"]) for s in scenarios]
    print(f"  fact count range: [{min(fact_counts)}, {max(fact_counts)}]")


if __name__ == "__main__":
    main()
