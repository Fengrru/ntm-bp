#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the ADVERSARIAL BALANCED held-out set (candidate cases + blind sheet).

Purpose (review response): Expansion-500 is 37/54 empty, so macro-F1 is
dominated by "correctly skip" and a zero-cost rule dominates. This set is
designed to be informative instead:

  * ~50% of cases have a NON-EMPTY ground-truth re-evaluation set R*;
  * many non-empty cases use NEUTRAL surface text and a SMALL confidence
    change (no negative keyword, no large confidence drop) so that a
    keyword/confidence rule cannot detect them;
  * many empty cases use NEGATIVE-sounding surface text (a keyword or a
    large confidence drop) that should NOT trigger re-evaluation;
  * several empty cases are STRENGTHENING updates (supersede-style), the
    structured gate's known failure mode.

HONESTY POLICY
--------------
* This script emits cases WITHOUT ground-truth labels. `reeval` is a human
  annotation product. The sheet is BLIND: no labels, no intent, no hints.
* The `intent` field below is used ONLY to balance the design and is never
  written to any output JSON.

Usage:
    python scripts/generate_balanced_expansion.py
Writes data/expansion_balanced_cases.json and
       data/expansion_balanced_sheet.json
"""
from __future__ import annotations

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CASES_PATH = os.path.join(DATA_DIR, "expansion_balanced_cases.json")
SHEET_PATH = os.path.join(DATA_DIR, "expansion_balanced_sheet.json")

# ----------------------------------------------------------------------
# Templates. intent: "M" = intended material/non-empty R*,
#                     "I" = intended immaterial/empty R*.
# ----------------------------------------------------------------------
TEMPLATES = [
    {
        "template_id": "bal_hospital_shift", "topology": "chain",
        "facts": [("f1", "急诊值班医护人员充足", 0.9),
                  ("f2", "急诊分诊流程顺畅", 0.85),
                  ("f3", "患者平均等待时间达标", 0.8)],
        "dependencies": [("f1", "f2"), ("f2", "f3")], "update_id": "f1",
        "updates": [
            {"value": "急诊骨干医护被整体抽调至分院", "conf": 0.72,
             "intent": "M", "note": "neutral text, small conf drop, material"},
            {"value": "一名患者家属就候诊体验提出个别投诉", "conf": 0.85,
             "intent": "I", "note": "negative word 投诉, immaterial"},
        ],
    },
    {
        "template_id": "bal_grid_dispatch", "topology": "chain",
        "facts": [("f1", "区域电网负荷处于安全区间", 0.9),
                  ("f2", "调度方案可正常执行", 0.85),
                  ("f3", "居民供电保持稳定", 0.8)],
        "dependencies": [("f1", "f2"), ("f2", "f3")], "update_id": "f1",
        "updates": [
            {"value": "区域电网负荷已超出安全上限", "conf": 0.74,
             "intent": "M", "note": "neutral text, material"},
            {"value": "某片区出现短时电压波动并已自动恢复", "conf": 0.9,
             "intent": "I", "note": "negative word 波动 but self-recovered"},
        ],
    },
    {
        "template_id": "bal_ecom_warehouse", "topology": "branch",
        "facts": [("f1", "华东仓库存充足", 0.9),
                  ("f2", "订单可次日达", 0.85),
                  ("f3", "退货处理及时", 0.8)],
        "dependencies": [("f1", "f2"), ("f1", "f3")], "update_id": "f1",
        "updates": [
            {"value": "华东仓库存已清空待补货", "conf": 0.7,
             "intent": "M", "note": "neutral text, material"},
            {"value": "华东仓个别订单配送小幅延迟，整体时效未受影响", "conf": 0.6,
             "intent": "I", "note": "negative word 延迟 + large conf drop, immaterial"},
        ],
    },
    {
        "template_id": "bal_mine_safety", "topology": "chain",
        "facts": [("f1", "矿区瓦斯浓度正常", 0.9),
                  ("f2", "作业面安全达标", 0.85),
                  ("f3", "本季度生产计划可完成", 0.8)],
        "dependencies": [("f1", "f2"), ("f2", "f3")], "update_id": "f1",
        "updates": [
            {"value": "矿区瓦斯浓度持续升高至临界值", "conf": 0.5,
             "intent": "M", "note": "neutral-ish text, material"},
            {"value": "矿区发生轻微违规操作，未造成任何影响", "conf": 0.85,
             "intent": "I", "note": "negative word 违规, immaterial"},
        ],
    },
    {
        "template_id": "bal_film_production", "topology": "chain",
        "facts": [("f1", "主演档期已锁定", 0.9),
                  ("f2", "拍摄进度按计划推进", 0.85),
                  ("f3", "影片可按期交付", 0.8)],
        "dependencies": [("f1", "f2"), ("f2", "f3")], "update_id": "f1",
        "updates": [
            {"value": "主演已转投其他项目", "conf": 0.72,
             "intent": "M", "note": "neutral text, material"},
            {"value": "剧组因现场噪音被居民投诉，已协调解决", "conf": 0.85,
             "intent": "I", "note": "negative word 投诉, immaterial"},
        ],
    },
    {
        "template_id": "bal_sports_league", "topology": "branch",
        "facts": [("f1", "核心球员身体健康", 0.9),
                  ("f2", "球队进攻火力充足", 0.85),
                  ("f3", "季后赛席位稳固", 0.8)],
        "dependencies": [("f1", "f2"), ("f1", "f3")], "update_id": "f1",
        "updates": [
            {"value": "核心球员已转会离队", "conf": 0.7,
             "intent": "M", "note": "neutral text, material"},
            {"value": "球队进攻火力创赛季新高", "conf": 0.95,
             "intent": "I", "note": "strengthening, supersede-style (gate failure probe)"},
        ],
    },
    {
        "template_id": "bal_bank_credit", "topology": "chain",
        "facts": [("f1", "企业客户还款能力稳定", 0.9),
                  ("f2", "贷款风险可控", 0.85),
                  ("f3", "信贷额度可维持", 0.8)],
        "dependencies": [("f1", "f2"), ("f2", "f3")], "update_id": "f1",
        "updates": [
            {"value": "企业客户已进入资产处置程序", "conf": 0.7,
             "intent": "M", "note": "neutral text, material"},
            {"value": "企业客户变更了注册办公地址", "conf": 0.9,
             "intent": "I", "note": "unrelated"},
        ],
    },
    {
        "template_id": "bal_air_cargo", "topology": "converge",
        "facts": [("f1", "货运航线运力充足", 0.9),
                  ("f2", "地面装卸效率正常", 0.85),
                  ("f3", "海关清关顺畅", 0.8),
                  ("f4", "货物可按期交付", 0.75)],
        "dependencies": [("f1", "f4"), ("f2", "f4"), ("f3", "f4")],
        "update_id": "f1",
        "updates": [
            {"value": "货运航线运力被大幅压缩", "conf": 0.6,
             "intent": "M", "note": "material (chain to f4)"},
            {"value": "货运单据统一改为电子版", "conf": 0.9,
             "intent": "I", "note": "unrelated process change"},
        ],
    },
    {
        "template_id": "bal_pharma_coldchain", "topology": "chain",
        "facts": [("f1", "冷链温度全程达标", 0.9),
                  ("f2", "疫苗效力有保障", 0.85),
                  ("f3", "接种计划可执行", 0.8)],
        "dependencies": [("f1", "f2"), ("f2", "f3")], "update_id": "f1",
        "updates": [
            {"value": "冷链运输车制冷机组中途故障", "conf": 0.55,
             "intent": "M", "note": "material"},
            {"value": "冷库更换了照明灯具", "conf": 0.9,
             "intent": "I", "note": "unrelated"},
        ],
    },
    {
        "template_id": "bal_datacenter", "topology": "chain",
        "facts": [("f1", "机房电力供应稳定", 0.9),
                  ("f2", "服务器在线率达标", 0.85),
                  ("f3", "云服务可用性达 SLA", 0.8)],
        "dependencies": [("f1", "f2"), ("f2", "f3")], "update_id": "f1",
        "updates": [
            {"value": "机房主备电源同时切换失败", "conf": 0.5,
             "intent": "M", "note": "material"},
            {"value": "机房更新了门禁记录系统", "conf": 0.9,
             "intent": "I", "note": "unrelated"},
        ],
    },
    {
        "template_id": "bal_env_monitor", "topology": "branch",
        "facts": [("f1", "企业排放指标合规", 0.9),
                  ("f2", "周边水质达标", 0.85),
                  ("f3", "社区投诉率为零", 0.8)],
        "dependencies": [("f1", "f2"), ("f1", "f3")], "update_id": "f1",
        "updates": [
            {"value": "企业排放指标被监测超标", "conf": 0.55,
             "intent": "M", "note": "material"},
            {"value": "企业组织了一次环保宣传讲座", "conf": 0.9,
             "intent": "I", "note": "unrelated"},
        ],
    },
    {
        "template_id": "bal_auto_parts", "topology": "chain",
        "facts": [("f1", "关键零部件供应稳定", 0.9),
                  ("f2", "整车产线不停线", 0.85),
                  ("f3", "月度交付达成", 0.8)],
        "dependencies": [("f1", "f2"), ("f2", "f3")], "update_id": "f1",
        "updates": [
            {"value": "关键零部件供应商工厂发生火灾", "conf": 0.3,
             "intent": "M", "note": "material, no keyword"},
            {"value": "供应商更换了企业标识", "conf": 0.9,
             "intent": "I", "note": "unrelated"},
        ],
    },
    {
        "template_id": "bal_restaurant", "topology": "branch",
        "facts": [("f1", "中央厨房供应正常", 0.9),
                  ("f2", "门店菜品出品稳定", 0.85),
                  ("f3", "顾客复购率稳定", 0.8)],
        "dependencies": [("f1", "f2"), ("f1", "f3")], "update_id": "f1",
        "updates": [
            {"value": "中央厨房因食品安全问题被责令停业", "conf": 0.3,
             "intent": "M", "note": "material"},
            {"value": "中央厨房调整了餐具供应商", "conf": 0.9,
             "intent": "I", "note": "unrelated"},
        ],
    },
    {
        "template_id": "bal_realestate_lease", "topology": "chain",
        "facts": [("f1", "写字楼主力租户续约", 0.9),
                  ("f2", "出租率保持稳定", 0.85),
                  ("f3", "租金收入达标", 0.8)],
        "dependencies": [("f1", "f2"), ("f2", "f3")], "update_id": "f1",
        "updates": [
            {"value": "写字楼主力租户提前解约搬离", "conf": 0.65,
             "intent": "M", "note": "material, neutral-ish"},
            {"value": "写字楼大堂更换了绿植", "conf": 0.9,
             "intent": "I", "note": "unrelated"},
        ],
    },
    {
        "template_id": "bal_reinsurance", "topology": "converge",
        "facts": [("f1", "原保险公司偿付能力充足", 0.9),
                  ("f2", "再保分出比例合理", 0.85),
                  ("f3", "巨灾风险敞口可控", 0.8),
                  ("f4", "再保合约可续签", 0.75)],
        "dependencies": [("f1", "f4"), ("f2", "f4"), ("f3", "f4")],
        "update_id": "f1",
        "updates": [
            {"value": "原保险公司偿付能力评级被下调", "conf": 0.6,
             "intent": "M", "note": "material"},
            {"value": "再保合约文本格式修订", "conf": 0.9,
             "intent": "I", "note": "unrelated"},
        ],
    },
    {
        "template_id": "bal_exam_admin", "topology": "chain",
        "facts": [("f1", "题库与试卷审核完成", 0.9),
                  ("f2", "考试系统压力测试通过", 0.85),
                  ("f3", "考试可如期举行", 0.8)],
        "dependencies": [("f1", "f2"), ("f2", "f3")], "update_id": "f2",
        "updates": [
            {"value": "考试系统压力测试未达通过标准", "conf": 0.4,
             "intent": "M", "note": "material (update f2)"},
            {"value": "考试系统压力测试通过率创历史新高", "conf": 0.95,
             "intent": "I", "note": "strengthening (gate failure probe)"},
        ],
    },
    {
        "template_id": "bal_flood_control", "topology": "chain",
        "facts": [("f1", "水库水位低于警戒线", 0.9),
                  ("f2", "堤坝安全有保障", 0.85),
                  ("f3", "下游城镇防洪安全", 0.8)],
        "dependencies": [("f1", "f2"), ("f2", "f3")], "update_id": "f1",
        "updates": [
            {"value": "流域上游持续强降雨，水库水位逼近警戒线", "conf": 0.6,
             "intent": "M", "note": "material, few keywords"},
            {"value": "水库管理站组织了一次防汛演练", "conf": 0.9,
             "intent": "I", "note": "unrelated"},
        ],
    },
    {
        "template_id": "bal_telecom_base", "topology": "deep",
        "facts": [("f1", "基站设备供电稳定", 0.9),
                  ("f2", "基站信号覆盖达标", 0.85),
                  ("f3", "用户通话质量良好", 0.8),
                  ("f4", "客户投诉率低于目标", 0.75)],
        "dependencies": [("f1", "f2"), ("f2", "f3"), ("f3", "f4")],
        "update_id": "f1",
        "updates": [
            {"value": "基站供电线路被施工挖断", "conf": 0.4,
             "intent": "M", "note": "material"},
            {"value": "基站外观进行了美化涂装", "conf": 0.9,
             "intent": "I", "note": "unrelated"},
        ],
    },
    {
        "template_id": "bal_port_customs", "topology": "chain",
        "facts": [("f1", "港口清关流程通畅", 0.9),
                  ("f2", "货物滞港时间达标", 0.85),
                  ("f3", "客户交付时效有保障", 0.8)],
        "dependencies": [("f1", "f2"), ("f2", "f3")], "update_id": "f1",
        "updates": [
            {"value": "港口海关系统因网络攻击中断", "conf": 0.5,
             "intent": "M", "note": "material"},
            {"value": "港口增设了司机休息区", "conf": 0.9,
             "intent": "I", "note": "unrelated"},
        ],
    },
    {
        "template_id": "bal_semiconductor", "topology": "mixed",
        "facts": [("f1", "晶圆代工产能充足", 0.9),
                  ("f2", "流片排期可满足", 0.85),
                  ("f3", "客户芯片可按期量产", 0.8),
                  ("f4", "客户营收预期乐观", 0.75)],
        "dependencies": [("f1", "f2"), ("f2", "f3"), ("f3", "f4")],
        "update_id": "f1",
        "updates": [
            {"value": "晶圆厂因设备故障暂停部分产线", "conf": 0.45,
             "intent": "M", "note": "material"},
            {"value": "晶圆厂举办了开放日活动", "conf": 0.9,
             "intent": "I", "note": "unrelated"},
        ],
    },
]

ANNOTATION_INSTRUCTIONS = (
    "对每个案例：阅读初始事实、依赖边和更新内容，按语义判断哪些下游事实的"
    "结论有效性被这次更新实质动摇（需要重新评估），把这些 fact_id 填入 "
    "reeval 列表；若没有则填空列表。判断依据只有文本语义，不使用任何算法"
    "输出，也不要参考置信度数值。强化型/轻微更新通常不动摇下游；使前提被"
    "否定或明显削弱的更新通常动摇整条依赖链；与结论语义无关的更新即使字面"
    "负面也不动摇下游。"
)


def build_case(template, value, conf):
    facts_dict = {fid: [val, c] for fid, val, c in template["facts"]}
    return {
        "scenario_id": f"bal_{template['template_id']}",
        "template_id": template["template_id"],
        "topology": template["topology"],
        "facts": facts_dict,
        "dependencies": [[p, c] for p, c in template["dependencies"]],
        "update_fact_id": template["update_id"],
        "new_value": value,
        "new_confidence": conf,
    }


def main():
    if os.path.isfile(CASES_PATH) or os.path.isfile(SHEET_PATH):
        raise SystemExit(f"REFUSING to overwrite {CASES_PATH} / {SHEET_PATH}")

    scenarios, sheet_cases = [], []
    n_m = n_i = 0
    for t in TEMPLATES:
        old_value = next(v for fid, v, _ in t["facts"] if fid == t["update_id"])
        for u in t["updates"]:
            if u["intent"] == "M":
                n_m += 1
            else:
                n_i += 1
            s = build_case(t, u["value"], u["conf"])
            scenarios.append(s)
            sheet_cases.append({
                "case_id": f"balanced::{t['template_id']}||{u['value']}",
                "dataset": "expansion_balanced",
                "facts": s["facts"],
                "dependencies": s["dependencies"],
                "update_fact_id": s["update_fact_id"],
                "old_value": old_value,
                "new_value": u["value"],
                "new_confidence": u["conf"],
                "reeval": None,
            })

    cases_payload = {
        "metadata": {
            "dataset": "EXPANSION-BALANCED (adversarial held-out candidates)",
            "n_templates": len(TEMPLATES),
            "n_cases": len(scenarios),
            "label_status": "UNLABELED -- ground truth requires human annotation",
        },
        "scenarios": scenarios,
    }
    with open(CASES_PATH, "w", encoding="utf-8") as f:
        json.dump(cases_payload, f, ensure_ascii=False, indent=2)

    sheet = {
        "instructions": ANNOTATION_INSTRUCTIONS,
        "note": ("BLIND sheet: no labels, no intent, no hints. Fill 'reeval' "
                 "with a list of fact_ids (may be empty)."),
        "cases": sheet_cases,
    }
    with open(SHEET_PATH, "w", encoding="utf-8") as f:
        json.dump(sheet, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(scenarios)} cases ({n_m} intended-nonempty, "
          f"{n_i} intended-empty) -> {CASES_PATH}")
    print(f"Blind sheet -> {SHEET_PATH}")


if __name__ == "__main__":
    main()
