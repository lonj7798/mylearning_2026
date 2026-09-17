---
chapter: ch-50
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/tulu-3.md
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision — numbers replaced with the report's tables)"
---

# Excerpt: Tülu 3 — per-stage table, and the development/unseen split

**Source library:** `wiki/raw-data/llm-training/model-reports/tulu-3.md`
**Artifact:** Tülu 3: Pushing Frontiers in Open Language Model Post-Training (Allen AI), arXiv:2411.15124.
**Checked on 2026-09-15** against the arXiv PDF; all values below are quoted at the table or section given.

> **Correction carried by this excerpt.** The library card's `## Technical Details — What RLVR buys`
> states "+5–10pp on GSM8K, +~4pp on IFEval" for the RLVR stage relative to the DPO checkpoint. Table 6 and
> Table 23 both give 8B GSM8K 84.3 → 87.6 (+3.3) and IFEval 81.1 → 82.4 (+1.3). The earlier version of this
> excerpt repeated the card's figures. Where the card and the tables disagree, the tables govern.

---

## Why ch-50 uses this source

Two of the chapter's three worked examples come from this report: the per-stage per-benchmark table
(ch-50 §1) and the paired development/unseen evaluation design (ch-50 §2).

---

## Per-stage per-benchmark table — Table 6 (8B, released checkpoints)

| Benchmark (eval setting) | Tülu 3 8B SFT | Tülu 3 8B DPO | Tülu 3 8B (final) |
|---|---|---|---|
| Avg. | 60.6 | 64.7 | 65.1 |
| MMLU (0-shot, CoT) | 65.9 | 68.7 | 68.2 |
| PopQA (15-shot) | 29.3 | 29.3 | 29.1 |
| TruthfulQA (6-shot) | 46.8 | 56.1 | 55.0 |
| BigBenchHard (3-shot, CoT) | 69.7 | 68.7 | 69.0 |
| DROP (3-shot) | 61.3 | 62.5 | 62.6 |
| MATH (4-shot CoT, Flex) | 31.5 | 42.0 | 43.7 |
| GSM8K (8-shot, CoT) | 76.2 | 84.3 | 87.6 |
| HumanEval (pass@10) | 86.2 | 83.9 | 83.9 |
| HumanEval+ (pass@10) | 81.4 | 78.6 | 79.2 |
| IFEval (prompt loose) | 72.8 | 81.1 | 82.4 |
| AlpacaEval 2 (LC % win) | 12.4 | 33.5 | 34.5 |
| Safety (6-task avg.) | 93.1 | 87.2 | 85.5 |

Table 23 (§6.4) reports the same DPO → RLVR comparison at 8B with an average of 64.4 → 64.8; it differs
from Table 6 on the average and on BigBenchHard because some rows use different eval settings
(IFEval Strict rather than prompt loose, a different BBH setting). The §6.4 prose describes the 8B result
as "non-trivial improvements … improving all three of MATH, GSM8k, and IFEval" and does not discuss the
TruthfulQA and Safety rows.

At 70B the pattern differs: GSM8K 93.5 → 93.5 with §6.4 attributing the absence of movement to saturation.

## Development and unseen suites — §2.2, §7.2–7.4

> Crucially, we did not examine scores on our unseen set when developing our models, allowing us to
> observe how much we may have overfit to particular evaluations in our decisions around data mixtures,
> algorithms, and hyperparameters. (§2.2)

Pairing (Table 3): MMLU → MMLU-Pro and GPQA; BigBenchHard → AGIEval English; MATH and GSM8K → DeepMind
Mathematics; HumanEval and HumanEval+ → BigCodeBench; IFEval → IFEval-OOD; AlpacaEval 2 → HREF. There is
no unseen safety evaluation (§2.2).

**Table 31 (8B), development (Dev.) and unseen (Uns.) per skill:**

| Skill | SFT Dev / Uns | DPO Dev / Uns | Final Dev / Uns |
|---|---|---|---|
| Avg. | 64.9 / 29.9 | 68.3 / 31.9 | 68.8 / 32.4 |
| Knowledge Recall (MMLU → GPQA) | 65.9 / 31.9 | 68.7 / 31.2 | 68.2 / 35.7 |
| Reasoning (BBH → AGIEval) | 67.9 / 56.2 | 65.8 / 61.8 | 66.0 / 59.3 |
| Math (MATH → DM Mathematics) | 31.5 / 32.3 | 42.0 / 33.0 | 43.7 / 35.4 |
| Coding (HumanEval → BigCodeBench) | 86.2 / 11.5 | 83.9 / 9.5 | 83.9 / 7.4 |
| Inst. Following (IFEval → IFEval-OOD) | 72.8 / 17.6 | 81.1 / 23.9 | 82.4 / 24.3 |

The authors' own conclusions from §7.4.1: the final checkpoints obtain the best average on both splits;
"our choices overfit to the development evaluations in Precise Instruction Following, and to some extent in
Knowledge Recall and Reasoning"; and the DPO data-scaling curves (Fig. 24) indicate "our development
process overfit to MATH to some extent", attributed to LaTeX formatting differences between MATH and
DeepMind Mathematics.

§7.4.2, discussing Table 33, reports a general result about the IFEval pair: "there is a significant difference between
performance on IFEval and IFEval-OOD of all the models, even though we created the latter to be structured
very similar to the original dataset, just with a disjoint set of constraints."

## Checkpoint selection — §6.4

> We evaluated our models every 100 training steps (40 for 70B), and picked as our final 8B model the
> checkpoints with best overall performance on MATH and IFEval.

The same paragraph reports 8B runs reaching GSM8K 89.4 and IFEval 84.8 that "tended to perform worse in
other metrics, dragging down their overall average". The selection rule is itself a slicing decision: two
slices decide which checkpoint is released.

## Used by

ch-50 §1 (per-stage table), §2 (development/unseen gap), §8 (decision table), Recipe (evaluation-split and
checkpoint-selection rows), Generalization lens (a) and (b).
