# Annotation Guidelines — Expansion-500 Ground Truth

This document records the frozen policy used to produce the ground-truth
re-evaluation sets `R*` for the Expansion-500 held-out set. It is included
so that the labels can be audited independently of any engine output.

## Task

Each case is a synthetic dependency graph (a DAG of facts/conclusions) plus
one fact update. The label is the set `R*` of downstream `fact_id`s whose
conclusion validity is *materially* undermined by the update and therefore
requires re-evaluation. `R*` may be empty.

## Frozen annotation instruction (verbatim, from `expansion_annotation_sheet.json`)

> 对每个案例：阅读初始事实、依赖边和更新内容，按语义判断哪些下游事实的结论有效性被这次更新实质动摇（需要重新评估），把这些 fact_id 填入 reeval 列表；若没有则填空列表。判断依据只有文本语义，不使用任何算法输出。强化型/轻微更新通常不动摇下游；使前提被否定或明显削弱的更新通常动摇整条依赖链；语义无关的依赖边会阻断影响。

In English:

- Judge from **text semantics alone**; never use engine/algorithm output.
- A **directional / negating / materially weakening** update usually shakes
  the affected dependency chain.
- A **strengthening or confirmation** update usually leaves downstream
  conclusions untouched, except conclusions that are *opposed* to the
  strengthened situation.
- A **minor fluctuation** or a **semantically unrelated edge** usually
  yields the empty set.
- The direction is judged from the *text*, not from the numeric confidence
  value (confidence is a noisy signal and is deliberately withheld from the
  semantic gate).

## Update classes (template design)

The generator balances each of the 18 templates across three update classes
(`A` directional change / `B` confirmation-or-strengthening / `C` minor
fluctuation), giving 18 × 3 = 54 cases. The class labels are code-level
comments only; they never appear in the JSON handed to annotators, and the
**annotators labelled `R*` per case, not the A/B/C class**.

## Blank sheet and independence

- The blind sheet `annotation_sheet_blind.json` contains no labels, no type
  hints and no rationale (`"reeval": null`).
- First annotator: the author (wrote the policy).
- Second annotator: **Li Zixuan (advisor)**, a blinded second coder who
  filled the sheet after all existing labels and all engine outputs were
  removed, with no access to the first annotator's labels.
- Agreement: **Cohen's κ = 1.0** at both the re-evaluation-set and per-fact
  levels.
- Caveat (stated in the paper): the second annotator is an in-school
  advisor, not a fully disinterested third party, so the perfect agreement
  is read as a *consistency check on the annotation policy* rather than as
  independent external validation.

## Files

| file | contents |
| --- | --- |
| `data/expansion_cases.json` | 54 unlabeled cases (blind), 18 templates × 3 updates |
| `data/expansion_annotation_sheet.json` | blind sheet with the verbatim instructions |
| `data/annotation_sheet_blind.json` | alternative blind sheet |
| `data/expansion_sheet_annotator1.json` | first annotator's filled sheet |
| `data/expansion_sheet_annotator2.json` | second annotator's filled sheet |
| `data/expansion500.json` | final labelled cases × 10 jitter instances (540 instances) |

## Reproducing the labels' effect

The labels are an input, not an output: no script regenerates them. The
agreement and the resulting metrics are reproduced by
`scripts/reproduce_paper_tables.py` (Tables 3–4) from the committed
`data/expansion500.json`.
