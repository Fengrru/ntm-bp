# Frozen Prompt Templates (Semantic Gate)

> **Status: FROZEN.** These prompts are part of the paper's experimental
> record. Do not edit them and re-run the *sealed* test split: prompt
> iteration may only look at dev cases (`data/case_split.json`). If you
> want to improve the gate, add a new builder version and evaluate it on
> dev cases first.

## Honesty rules (non-negotiable)

1. **Prompts contain only fact texts.** They never receive the
   hand-annotated `relation_type` (supports/contradicts/neutral), the
   `theta` annotation, or any ground-truth label. The semantic relation
   must be inferred from the text itself.
2. **Every LLM call is recorded** (`prompt` + `raw_response` + parsed
   value) in the engine's `call_log`; `scripts/audit_cache.py` enforces
   rule 1 mechanically and flags any parse fallback.

## The Q1/Q2/Q3 structured gate (recommended)

Instead of one scalar-impact prompt, the `StructuredGateEngine` asks
three SMALL single-choice questions and combines the answers
deterministically -- mirroring the annotation guideline's own decision
procedure:

| # | question | asked | answers |
| --- | --- | --- | --- |
| Q1 | what kind of change is new vs old? | once per update | A = substantive change / B = confirmation / C = minor fluctuation |
| Q2 | real influence mechanism between update and target? | per descendant | 是 / 否 |
| Q3 | target aligned with or opposed to the confirmed situation? | per descendant, only if Q1 == B | 同向 / 对立 |

**Verdict:** `reeval` iff `Q2 == yes AND (Q1 == A OR (Q1 == B AND Q3 == opposed))`.

Design rationale (from dev iteration):

- Q1 excludes same-direction intensification from A: the annotation
  guideline treats it as a REINFORCING update that shakes only opposed
  conclusions.
- Q1's A includes "a stated trend/state no longer holds" -- withdrawing
  a trend is substantive, not a minor fluctuation, even if the numeric
  change is small.
- Q2 is phrased counterfactually ("假如这一情况属实…") and explicitly
  rejects merely-sharing-a-subject as a mechanism. An earlier stricter
  phrasing ("清晰的业务影响链条") over-corrected and was rolled back.
- Q3 judges stance on CONTENT alone (not on conclusion-formation
  history), which fixed the contradicts-reinforce case being misread as
  aligned.

### Q1 prompt (change type, per update)

```
某条事实记录刚被更新：
旧陈述：{old_value}
新陈述：{new_value}

新陈述相对旧陈述属于以下哪种？
A. 方向性改变：原情况被否定、反转或明显削弱；或旧陈述所述的趋势/状态整体上不再成立（即使变化幅度不大）。注意：单次、偶发的小事件不构成趋势/状态的改变，应归入 C。
B. 确认或加强：确认、重申、进一步坐实旧陈述已表达的情况，或表明原情况沿同一方向进一步发展/加重。
C. 轻微波动：只是小幅、偶发、不改变整体判断的变化（例如一次性的小插曲、单次轻微异常），旧陈述的整体判断仍然成立。

只输出一个字母（A、B 或 C），不要输出其他内容。
答案：
```

Fail-open rule: unparseable response → `A` (re-evaluate).

### Q2 prompt (relevance, per descendant)

```
已知情况：{new_value}
某条结论：{target_text}

假如这一情况属实，它是否会影响对该结论的判断（无论支持还是反对，直接影响或通过合理的连锁影响都算）？
仅当两者在业务上互不相干（例如只是同属一个公司但没有任何影响链条）时才答否。

只输出"是"或"否"，不要输出其他内容。
答案：
```

Fail-open rule: response not starting with "否" → relevant.

### Q3 prompt (stance, per descendant when Q1 == B)

```
已知情况：{new_value}
某条结论：{target_text}

只看内容本身，这一情况与该结论是什么关系？
同向：两者一致或互相印证，该情况成立会让该结论更站得住。
对立：两者冲突，该情况成立会让该结论更难成立、面临阻碍或风险。

只输出"同向"或"对立"，不要输出其他内容。
答案：
```

## Scalar-impact prompts (v1 / v2 / v3)

The earlier scalar prompts ask for one impact score in [0, 1] with
`tau_gate = 0.5`; `score >= tau_gate` marks re-evaluation. They share a
common preamble built from the dependency path:

```
一个记忆系统中，某条事实刚被更新：
旧陈述：{old_value}
新陈述：{new_value}
（对新陈述本身的置信度约为 {new_confidence:.2f}）
记忆中存在一条下游结论：{target_text}
该结论直接依赖于被更新的事实。        # or: 该结论通过以下中间结论间接依赖于被更新的事实：
- {intermediate fact text}             #    one line per hop

{...version-specific rules...}

只输出一个 0.0 到 1.0 之间的数字，不要输出其他内容。
冲击强度：
```

### v1 (original, pre-split experiments)

Rules: (1) semantically unrelated → ~0; (2) minor fluctuation → low;
(3) premise negated/weakened → high; (4) even without a source
confidence drop, if the update makes the conclusion less tenable → high.

Known failure mode: rule 4 over-fires on updates that merely confirm an
already-known negative (a "confirmation of a known negative" update).

### v2 (comparative anchoring, dev-iterated)

Replaces rule 4 with a DELTA-based judgment: score is anchored on
whether the update brings substantive NEW information relative to the
old statement. Rule 4 keeps the "contradiction hardens" case (the
CO-Relation blind spot that motivated v1's rule 4).

### v3 (stance analysis, dev-iterated)

Explicit three-step reasoning (relevance → change nature → stance when
(b)), plus the clarification that a withdrawn trend is a substantive
change. Fixes both v1's over-fire (confirmation + aligned → low) and
v2's regression on the CO blind spot (confirmation + opposed → high).

## Prompt iteration discipline

- Only dev cases may be inspected when iterating (`data/case_split.json`).
- The sealed test split is evaluated ONCE with the final frozen prompt.
- `co_relation` scenarios are all dev: the gate's rule 4 was designed
  from the CO-Relation blind-spot analysis, so that set is a design set,
  not held-out data.
