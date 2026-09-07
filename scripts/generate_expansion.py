"""Generate the expansion dataset (Stage-2): candidate cases + blind sheet.

HONESTY POLICY
--------------
* This script outputs cases WITHOUT any ground-truth labels. reeval is a
  human annotation product, never a generator output. The generated
  annotation sheet is BLIND (no labels, no type hints, no rationale).
* Template comments label the intended update TYPE (A/B/C) only to keep
  the template library balanced across the three update classes; these
  comments are code-level and never appear in the JSON outputs.
* The second annotator must be an independent human who has not seen
  the existing labels or any engine output (see scripts/annotation_review.py).

Design goals (vs PC-500 / SynDep):
* Broader domains: healthcare, insurance, software release, logistics,
  energy, real estate, aviation, education, food, banking, manufacturing,
  retail, telecom, agriculture, tourism, pharma, e-commerce, AI projects.
* Richer topologies: chain, branch, converge, deep-chain, mixed (with
  diamond and neutral-blocking edges).
* Each template has exactly 3 update values covering the three classes:
  directional change / confirmation-or-strengthening / minor fluctuation.

Usage:
    python scripts/generate_expansion.py
"""

from __future__ import annotations

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CASES_PATH = os.path.join(DATA_DIR, "expansion_cases.json")
SHEET_PATH = os.path.join(DATA_DIR, "expansion_annotation_sheet.json")

# ---------------------------------------------------------------
# Templates. Each entry: facts (fid, text, conf), dependencies,
# update_id, and 3 new values (A: directional change, B: strengthen,
# C: minor fluctuation) with assertion confidence.
# ---------------------------------------------------------------

PC_STYLE_TEMPLATES = [
    {
        "template_id": "med_chain",
        "topology": "chain",
        "facts": [
            ("f1", "患者病情稳定", 0.9),
            ("f2", "采用保守治疗方案", 0.85),
            ("f3", "计划近期出院", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3")],
        "update_id": "f1",
        "new_values": [
            ("患者病情明显恶化", 0.3),       # A
            ("复查结果确认病情稳定", 0.9),   # B
            ("患者出现偶发低烧", 0.8),       # C
        ],
    },
    {
        "template_id": "insurance_branch",
        "topology": "branch",
        "facts": [
            ("f1", "保单条款清晰无歧义", 0.9),
            ("f2", "理赔流程顺畅", 0.85),
            ("f3", "赔付快速到位", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f1", "f3")],
        "update_id": "f1",
        "new_values": [
            ("保单条款存在歧义", 0.3),       # A
            ("保司发布条款补充说明", 0.9),   # B
            ("保单模板例行措辞更新", 0.85),  # C
        ],
    },
    {
        "template_id": "software_release_chain",
        "topology": "chain",
        "facts": [
            ("f1", "自动化测试覆盖充分", 0.9),
            ("f2", "发布候选版本质量达标", 0.85),
            ("f3", "产品按期上线", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3")],
        "update_id": "f1",
        "new_values": [
            ("测试中发现关键安全缺陷", 0.2),   # A
            ("全量回归测试通过", 0.9),         # B
            ("测试中发现少量低优先级小缺陷", 0.85),  # C
        ],
    },
    {
        "template_id": "logistics_chain",
        "topology": "chain",
        "facts": [
            ("f1", "港口运转正常", 0.9),
            ("f2", "海运时效有保障", 0.85),
            ("f3", "订单能按时交付", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3")],
        "update_id": "f1",
        "new_values": [
            ("港口工人开始罢工", 0.2),        # A
            ("港口吞吐量创新高", 0.9),        # B
            ("港口某日出现小幅拥堵", 0.8),    # C
        ],
    },
    {
        "template_id": "energy_chain",
        "topology": "chain",
        "facts": [
            ("f1", "天然气供应充足", 0.9),
            ("f2", "能源成本保持稳定", 0.85),
            ("f3", "维持当前产品定价", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3")],
        "update_id": "f1",
        "new_values": [
            ("天然气管道发生供应中断", 0.3),  # A
            ("长期供应协议顺利续签", 0.9),    # B
            ("天然气短期价格小幅波动", 0.8),  # C
        ],
    },
    {
        "template_id": "realestate_branch",
        "topology": "branch",
        "facts": [
            ("f1", "小区周边配套完善", 0.9),
            ("f2", "房产具备升值预期", 0.85),
            ("f3", "当前值得购入", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f1", "f3")],
        "update_id": "f1",
        "new_values": [
            ("周边商业配套规划被取消", 0.2),  # A
            ("周边新建学校正式动工", 0.9),    # B
            ("周边新开一家便利店", 0.85),     # C
        ],
    },
    {
        "template_id": "aviation_chain",
        "topology": "chain",
        "facts": [
            ("f1", "该航线航班准点率高", 0.9),
            ("f2", "商务出行选择该航线可靠", 0.85),
            ("f3", "续签年度出行合作协议", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3")],
        "update_id": "f1",
        "new_values": [
            ("空管人员罢工导致准点率骤降", 0.3),   # A
            ("该航线准点率创历史新高", 0.9),       # B
            ("该航线某周准点率略有下降", 0.8),     # C
        ],
    },
    {
        "template_id": "education_branch",
        "topology": "branch",
        "facts": [
            ("f1", "课程教学质量高", 0.9),
            ("f2", "学员满意度高", 0.85),
            ("f3", "学员续报率高", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f1", "f3")],
        "update_id": "f1",
        "new_values": [
            ("王牌讲师离职", 0.3),            # A
            ("学员对课程打出满分评价", 0.9),  # B
            ("本周有一次课程临时调课", 0.85),  # C
        ],
    },
    {
        "template_id": "food_chain",
        "topology": "chain",
        "facts": [
            ("f1", "原料安全合规", 0.9),
            ("f2", "产品可放心销售", 0.85),
            ("f3", "市场份额保持稳定", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3")],
        "update_id": "f1",
        "new_values": [
            ("一批原料被检出超标", 0.2),      # A
            ("第三方复检全部合格", 0.9),      # B
            ("单批次标签印刷有瑕疵", 0.85),   # C
        ],
    },
]

SYNDEP_STYLE_TEMPLATES = [
    {
        "template_id": "bank_converge",
        "topology": "converge",
        "facts": [
            ("f1", "基准利率下调", 0.9),
            ("f2", "存款流失速度放缓", 0.85),
            ("f3", "合规成本保持稳定", 0.8),
            ("f4", "净息差有望改善", 0.75),
        ],
        "dependencies": [("f1", "f4"), ("f2", "f4"), ("f3", "f4")],
        "update_id": "f1",
        "new_values": [
            ("基准利率大幅下调", 0.4),        # A
            ("基准利率按预期下调", 0.9),      # B
            ("基准利率微调", 0.85),           # C
        ],
    },
    {
        "template_id": "manufacturing_multi",
        "topology": "mixed",
        "facts": [
            ("f1", "订单量持续增长", 0.9),
            ("f2", "自动化产线已投产", 0.85),
            ("f3", "熟练工人充足", 0.8),
            ("f4", "产能利用率提升", 0.75),
            ("f5", "利润水平改善", 0.7),
        ],
        "dependencies": [("f1", "f4"), ("f2", "f4"), ("f3", "f4"), ("f4", "f5")],
        "update_id": "f1",
        "new_values": [
            ("订单量大幅下滑", 0.3),          # A
            ("订单量稳步增长", 0.9),          # B
            ("订单量出现小幅波动", 0.8),      # C
        ],
    },
    {
        "template_id": "retail_deep",
        "topology": "deep",
        "facts": [
            ("f1", "线上流量持续增长", 0.9),
            ("f2", "转化率稳步提升", 0.85),
            ("f3", "客单价上升", 0.8),
            ("f4", "营收增长", 0.75),
            ("f5", "盈利状况改善", 0.7),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3"), ("f3", "f4"), ("f4", "f5")],
        "update_id": "f1",
        "new_values": [
            ("线上流量大幅下滑", 0.2),        # A
            ("线上流量持续增长", 0.9),        # B
            ("线上流量单日波动", 0.85),       # C
        ],
    },
    {
        "template_id": "telecom_chain",
        "topology": "chain",
        "facts": [
            ("f1", "频谱资源充足", 0.9),
            ("f2", "5G 设备部署到位", 0.85),
            ("f3", "5G 商用服务启动", 0.8),
            ("f4", "用户向 5G 迁移加速", 0.75),
            ("f5", "网络负载压力上升", 0.7),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3"), ("f3", "f4"), ("f4", "f5")],
        "update_id": "f1",
        "new_values": [
            ("频谱资源被大幅缩减", 0.3),      # A
            ("额外频谱资源获批", 0.9),        # B
            ("频谱分配例行调整", 0.85),       # C
        ],
    },
    {
        "template_id": "agri_chain",
        "topology": "chain",
        "facts": [
            ("f1", "产区天气正常", 0.9),
            ("f2", "产量符合预期", 0.85),
            ("f3", "市场供给稳定", 0.8),
            ("f4", "农产品价格平稳", 0.75),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3"), ("f3", "f4")],
        "update_id": "f1",
        "new_values": [
            ("极端天气预警发布", 0.3),        # A
            ("气象良好丰收在望", 0.9),        # B
            ("出现短期降雨", 0.85),           # C
        ],
    },
    {
        "template_id": "tourism_branch",
        "topology": "branch",
        "facts": [
            ("f1", "签证政策宽松", 0.9),
            ("f2", "出境游预订量增长", 0.85),
            ("f3", "目的地热度上升", 0.8),
        ],
        "dependencies": [("f1", "f2"), ("f1", "f3")],
        "update_id": "f1",
        "new_values": [
            ("签证政策大幅收紧", 0.3),        # A
            ("签证政策放宽并延长有效期", 0.9),  # B
            ("签证受理时间微调", 0.85),        # C
        ],
    },
    {
        "template_id": "pharma_deep",
        "topology": "deep",
        "facts": [
            ("f1", "临床试验进展顺利", 0.9),
            ("f2", "审批流程正常推进", 0.85),
            ("f3", "药品上市时间可期", 0.8),
            ("f4", "营收预期乐观", 0.75),
        ],
        "dependencies": [("f1", "f2"), ("f2", "f3"), ("f3", "f4")],
        "update_id": "f1",
        "new_values": [
            ("临床试验结果未达主要终点", 0.2),  # A
            ("临床试验中期数据积极", 0.9),      # B
            ("临床试验日程小幅调整", 0.85),     # C
        ],
    },
    {
        "template_id": "ecommerce_converge",
        "topology": "converge",
        "facts": [
            ("f1", "物流网络覆盖完善", 0.9),
            ("f2", "支付体验良好", 0.85),
            ("f3", "复购率高", 0.8),
            ("f4", "平台 GMV 持续增长", 0.75),
        ],
        "dependencies": [("f1", "f3"), ("f2", "f3"), ("f3", "f4")],
        "update_id": "f1",
        "new_values": [
            ("仓库网点大幅缩减", 0.3),        # A
            ("新物流中心投入使用", 0.9),      # B
            ("个别网点布局调整", 0.85),       # C
        ],
    },
    {
        "template_id": "ai_project_merge",
        "topology": "mixed",
        "facts": [
            ("f1", "项目需求明确", 0.9),
            ("f2", "研发团队稳定", 0.85),
            ("f3", "技术方案可行", 0.8),
            ("f4", "里程碑如期达成", 0.75),
            ("f5", "项目按期交付", 0.7),
        ],
        "dependencies": [("f1", "f4"), ("f2", "f4"), ("f3", "f4"), ("f4", "f5")],
        "update_id": "f1",
        "new_values": [
            ("项目需求发生重大变更", 0.3),    # A
            ("需求经评审确认细化", 0.9),      # B
            ("需求文档措辞微调", 0.85),       # C
        ],
    },
]

SCENARIO_TEMPLATES = PC_STYLE_TEMPLATES + SYNDEP_STYLE_TEMPLATES

ANNOTATION_INSTRUCTIONS = (
    "对每个案例：阅读初始事实、依赖边和更新内容，按语义判断哪些下游事实的"
    "结论有效性被这次更新实质动摇（需要重新评估），把这些 fact_id 填入 "
    "reeval 列表；若没有则填空列表。判断依据只有文本语义，不使用任何算法"
    "输出。强化型/轻微更新通常不动摇下游；使前提被否定或明显削弱的更新"
    "通常动摇整条依赖链；语义无关的依赖边会阻断影响。"
)


def build_case(template: dict, new_value: str, new_confidence: float) -> dict:
    facts_dict = {fid: [val, conf] for fid, val, conf in template["facts"]}
    return {
        "scenario_id": f"exp_{template['template_id']}",
        "template_id": template["template_id"],
        "topology": template["topology"],
        "facts": facts_dict,
        "dependencies": [[p, c] for p, c in template["dependencies"]],
        "update_fact_id": template["update_id"],
        "new_value": new_value,
        "new_confidence": new_confidence,
        # NO ground-truth fields: labels are a human annotation product.
    }


def main() -> None:
    if os.path.isfile(CASES_PATH) or os.path.isfile(SHEET_PATH):
        print(f"REFUSING to overwrite existing outputs: "
              f"{CASES_PATH} / {SHEET_PATH}")
        raise SystemExit(1)

    scenarios = []
    sheet_cases = []
    for template in SCENARIO_TEMPLATES:
        old_value = next(
            val for fid, val, _ in template["facts"]
            if fid == template["update_id"]
        )
        for new_value, new_confidence in template["new_values"]:
            s = build_case(template, new_value, new_confidence)
            scenarios.append(s)
            sheet_cases.append({
                "case_id": f"expansion::{template['template_id']}||{new_value}",
                "dataset": "expansion",
                "facts": s["facts"],
                "dependencies": s["dependencies"],
                "update_fact_id": s["update_fact_id"],
                "old_value": old_value,
                "new_value": new_value,
                "new_confidence": new_confidence,
                "reeval": None,  # to be filled by human annotators
            })

    cases_payload = {
        "metadata": {
            "dataset": "EXPANSION (Stage-2 candidates)",
            "n_templates": len(SCENARIO_TEMPLATES),
            "n_cases": len(scenarios),
            "topologies": {t: sum(1 for x in SCENARIO_TEMPLATES
                                  if x["topology"] == t)
                           for t in sorted({x["topology"]
                                            for x in SCENARIO_TEMPLATES})},
            "label_status": "UNLABELED -- ground truth requires human "
                            "annotation via expansion_annotation_sheet.json",
        },
        "scenarios": scenarios,
    }
    with open(CASES_PATH, "w", encoding="utf-8") as f:
        json.dump(cases_payload, f, ensure_ascii=False, indent=2)

    sheet = {
        "instructions": ANNOTATION_INSTRUCTIONS,
        "note": ("BLIND sheet for expansion cases: no labels, no type hints, "
                 "no rationale. Fill 'reeval' with a list of fact_ids (may be "
                 "empty). Must be annotated independently by both annotators "
                 "before labels are accepted."),
        "cases": sheet_cases,
    }
    with open(SHEET_PATH, "w", encoding="utf-8") as f:
        json.dump(sheet, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(scenarios)} unlabeled cases -> {CASES_PATH}")
    print(f"Generated blind annotation sheet ({len(sheet_cases)} cases) "
          f"-> {SHEET_PATH}")
    print("Topologies:", cases_payload["metadata"]["topologies"])


if __name__ == "__main__":
    main()
