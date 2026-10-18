# 第 3 步：智能路由研究设计（Smart Routing, v0 草案）

> 目标：把"按更新类型选择处理路径"做成论文的核心方法贡献——在成本
> （LLM 调用数）与精度（reeval F1）之间显式权衡，取代"一刀切"范式。
> 启动时间：2026 年 10 月。本文件是设计蓝图，规则冻结后才能评测。

## 1. 动机（来自第 1、2 步的实证证据）

三层证据表明不同方法在不同更新类型上各有绝对优势：

| 更新类型（Q1 分类） | ntm_bp（数值） | gate_structured（LLM） | 证据 |
|---|---|---|---|
| C 轻微波动 | 误报（数值衰减过度敏感） | 完美（Q1 早停） | expansion：8/8 C 类案例 structured 全胜，ntm_bp 全误标 |
| B 强确认型（supersede） | 正确（数值不衰减，不标） | 误报（Q1 把"创新高/满分"判为 A） | expansion：8/8 案例 structured 全败，ntm_bp 全对 |
| B 普通确认/对立 | 部分 | 强（Q3 stance 判断） | sealed test syndep：B+对立 案例 structured 赢 |
| A 方向性改变 | 部分（深链漏标） | 强（反事实 relevance） | expansion：A 类 structured 无系统性失败 |

结论：**没有单一方法在所有类型上最优；Q1 分类（1 次 LLM 调用）足以作为
路由信号**。

## 2. 路由 v0 定义（已被事后模拟否决，见第 6 节）

```
route_update(mem, fid, old, new, conf):
    cls = Q1(old, new)                      # 1 次 LLM 调用
    if cls == C:  return                    # 不 reeval（跳过数值与下游判断）
    if cls == B:
        R = ntm_bp_propagate(mem, fid)      # 数值路径：B 类时数值信号可靠
        mark_reeval(R)                      # 含 B+对立 案例（数值在 B 类不误报）
        return
    # cls == A: LLM 路径（structured 逻辑）
    for target in descendants(fid):
        if relevant(old, new, target):      # 反事实问法
            mark_reeval(target)
```

**该规则已被 `_simulate_routing.py` 的事后模拟否决（0.750 < structured 0.778，
5 输 1 赢，全部失败发生在 B 类）**。核心假设"B 类时数值信号可靠"不成立：
阈值传播无法区分"确认后尘埃落定"与"确认且仍在变动"，在 GT 非空的
B 类案例上既不精确也不召回。

设计动机（先验，非数据拟合）：
- C 类：变化不实质，任何标记都是误报——直接跳过是最优路径；
- B 类：更新与旧陈述同向，数值传播的衰减方向与"确认"语义一致
  （强化不削弱），故数值路径天然保守、可靠；
- A 类：需要语义阻断判断（CO neutral 边教训），只能走 LLM relevance。

## 3. 诚实性纪律（关键）

1. **规则冻结原则**：路由规则必须在正式评测前声明并冻结。本文件的
   规则 v0 即冻结版本；任何基于评测结果的修改都必须走 dev 迭代流程。
2. **数据分层**：
   - sealed test（27 案例）与 expansion（54 案例）已被冻结方法评测过，
     只能用于：方法对比报告（已完成）、路由的**事后可行性分析**（明确
     标注 NOT ex-ante，不进论文结果表）。
   - 路由的正式结果必须来自**第三批数据**（见第 5 节），ex-ante 切分
     dev/test，规则冻结后一次性跑 test。
3. **不得复用**：expansion 已全量用作方法 test，不能再当路由的 test；
   也不能在 expansion 上迭代 Q1（supersede 修复需要新 dev 数据）。

## 4. 成本模型

- 路径成本（LLM 调用数/更新）：C 类 = 1（仅 Q1）；B 类 = 1 + 0（数值
  零 LLM）；A 类 = 1 + 每目标 1 次 relevance。
- 对比基线：BFS = 0 LLM 但精度 0.31（expansion）；structured = 平均
  ~2.9 调用/更新；v1 = 每目标 1 次（无早停）。
- 论文呈现：成本-精度帕累托前沿（x=调用数/更新，y=F1），四个点：
  BFS、ntm_bp、v1、structured、**routing v0**。

## 5. 第三批数据计划（10 月执行）

1. 用 `generate_expansion.py` 模式扩展 10-15 个新模板（重点覆盖：
   supersede 型强确认、B+对立、A 类 neutral 阻断——即路由的三个关键
   判别面），生成第三批候选案例。
2. 双标注者标注（复用盲表流程 + kappa2 工具）。
3. ex-ante 切分（新 seed，拒绝覆盖），dev/test。
4. dev 用于（可选）Q1 supersede 修复（若路由 v0 在 dev 上暴露该短板）；
   test 一次性评测路由 v0（及全部基线）。

## 6. 立即可做的验证（零成本，已完成）

- `scripts/_simulate_routing.py`：用 llm_gate_cache 离线回放 expansion
  540 实例（1550 次缓存回放，0 次新 API 调用），按路由 v0 规则合成
  pred。结果存档：`data/routing_v0_simulation.json`。

### 6.1 结果（事后分析，NOT ex-ante，不进论文结果表）

| 方法 | F1 | 95% CI |
|---|---|---|
| routing_v0 | 0.7500 | [0.637, 0.854] |
| ntm_bp | 0.6863 | [0.597, 0.759] |
| structured | 0.7778 | [0.667, 0.889] |

vs structured：赢 1 / 输 5 / 平 48。

### 6.2 失败定位与结论

- 5 个 loss 全部是 Q1 判 B 的案例（bank_converge、pharma_deep、
  retail_deep、tourism_branch、ai_project_merge、manufacturing_multi
  中除 1 个 win 外全部）——切到数值路径后全面劣于 structured 的
  stance 判断。唯一 win（bank_converge||基准利率按预期下调）也只是
  数值与 LLM 双双不完美下的侥幸。
- 此前"ntm_bp 在 B 类正确"的印象只对 supersede 型（GT=空）成立：
  数值不标=碰巧正确；对 GT 非空的 B 类案例（同向加剧需要重评的
  场景），阈值传播既不精确也不召回。
- 根本问题不是路径选择而是 Q1 的 supersede 误分类：8/54 案例 Q1
  把"同向强化到新水平"误判为 A，触发 relevance 全链误报。若 Q1 判
  对 B，structured 的 stance 路径本身全部正确。

### 6.3 修订方向（v1，10 月执行）

1. **放弃数值路径**：B 类维持 structured 的 stance 判断（v0 的
   B→数值假设被数据否决，写为论文负面结果："naive routing fails——
   数值信号在强化型更新上不可靠，因为它无法区分确认后尘埃落定与
   确认且仍在变动"）。
2. **聚焦 Q1 修复**：supersede 型误判是 structured 唯一系统性弱点，
   修复 = 在 Q1 定义中把"旧值被超越但方向未反转"显式归入 B（需
   第三批数据 dev 上迭代，ex-ante 评测）。
3. **成本层独立成贡献**：C 类 1 调用早停已是最优路由；论文呈现
   成本-精度帕累托前沿（BFS 0 调用 / ntm_bp 0 调用 / v1 / structured
   / routing 1+按需调用），其中 structured 与 routing 同路径但成本
   核算显式化。
