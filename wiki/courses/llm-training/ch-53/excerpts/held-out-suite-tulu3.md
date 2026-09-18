---
chapter: ch-53
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/tulu-3.md
source_url: https://arxiv.org/abs/2411.15124
primary_text_checked: arXiv:2411.15124 (Tülu 3 report PDF), 2026-09-17
---

# Excerpt: Tülu 3's development / unseen split and the stage-by-stage tables

Used by ch-53 §2 (held-out suite) and §6 (forgetting report). Every number below was read at the
stated locus in the report PDF on 2026-09-17.

## The split and the inspection rule

> "Both splits cover all identified skills, except we have no unseen safety evaluation. Crucially, we
> did not examine scores on our unseen set when developing our models, allowing us to observe how much
> we may have overfit to particular evaluations in our decisions around data mixtures, algorithms, and
> hyperparameters." (§2.2)

The unseen tasks were designed through an independent process from the development tasks, with three
stated principles: formulate tasks the way humans interact with models, give clear instructions that
specify the answer format, and use tolerant answer extraction so that models are not penalized for
small syntax deviations (§7.3).

## Table 3 — skills, development task, unseen task

| Core skill | Development | Unseen |
|---|---|---|
| Knowledge | MMLU (EM), PopQA (EM), TruthfulQA (MC2) | MMLU-Pro (EM), GPQA (EM) |
| Reasoning | BigBenchHard (EM), DROP (F1) | AGIEval English (EM) |
| Math | MATH (flex EM), GSM8K (EM) | DeepMind Mathematics (EM, sympy) |
| Coding | HumanEval, HumanEval+ (pass@10) | BigCodeBench (pass@10) |
| Instruction following | IFEval (EM), AlpacaEval 2 (win rate) | IFEval-OOD (pass@1), HREF (win rate) |
| Safety | Tülu 3 Safety (average over sub-evals) | — none — |

Evaluation settings are in Table 24: the unseen multiple-choice tasks are 0-shot with chain of thought
and a chat template; IFEval and IFEval-OOD use prompt-level loose accuracy.

## Table 31 — one development and one unseen task per skill

| Skill (dev → unseen) | 8B SFT dev / uns | 8B DPO dev / uns | 8B Final dev / uns |
|---|---|---|---|
| Average | 64.9 / 29.9 | 68.3 / 31.9 | 68.8 / 32.4 |
| Knowledge (MMLU → GPQA) | 65.9 / 31.9 | 68.7 / 31.2 | 68.2 / 35.7 |
| Reasoning (BBH → AGIEval) | 67.9 / 56.2 | 65.8 / 61.8 | 66.0 / 59.3 |
| Math (MATH → DM Mathematics) | 31.5 / 32.3 | 42.0 / 33.0 | 43.7 / 35.4 |
| Coding (HumanEval → BigCodeBench) | 86.2 / 11.5 | 83.9 / 9.5 | 83.9 / 7.4 |
| Instruction following (IFEval → IFEval-OOD) | 72.8 / 17.6 | 81.1 / 23.9 | 82.4 / 24.3 |

The 70B columns of the same table: average 78.1 / 41.0 (SFT), 80.5 / 44.4 (DPO), 80.7 / 44.4 (Final).

## Table 6 — 8B stage-by-stage development scores

| Benchmark | 8B SFT | 8B DPO | 8B Final |
|---|---|---|---|
| MMLU (0-shot CoT) | 65.9 | 68.7 | 68.2 |
| PopQA (15-shot) | 29.3 | 29.3 | 29.1 |
| TruthfulQA (6-shot) | 46.8 | 56.1 | 55.0 |
| BigBenchHard (3-shot CoT) | 69.7 | 68.7 | 69.0 |
| DROP (3-shot) | 61.3 | 62.5 | 62.6 |
| MATH (4-shot CoT, flex) | 31.5 | 42.0 | 43.7 |
| GSM8K (8-shot CoT) | 76.2 | 84.3 | 87.6 |
| HumanEval (pass@10) | 86.2 | 83.9 | 83.9 |
| HumanEval+ (pass@10) | 81.4 | 78.6 | 79.2 |
| IFEval (prompt loose) | 72.8 | 81.1 | 82.4 |
| AlpacaEval 2 (LC % win) | 12.4 | 33.5 | 34.5 |
| Safety (6-task average) | 93.1 | 87.2 | 85.5 |

Stage attribution: of the +11.4 GSM8K points between SFT and Final, +8.1 come from DPO and +3.3 from
RLVR; of the +9.6 IFEval points, +8.3 come from DPO and +1.3 from RLVR. No confidence intervals are
reported for any cell.

## Decontamination (§3.2, Tables 8 and 37)

Matching is computed on prompts only, using 8-grams. A test token counts as matched if the two
instances share an 8-gram containing it; the test instance overlaps a training instance when more than
50% of its tokens match that same training instance. A training set counts as contaminated with an
evaluation when its instances overlap more than 2% of that evaluation's instances.

Measured overlaps (Table 37, "% eval overlap" = share of evaluation instances overlapping the dataset):
Evol CodeAlpaca / HumanEval 70.7; DaringAnteater / MATH 30.7; NuminaMath-TIR / MATH 18.2; ShareGPT /
AlpacaEval 19.2; LMSys Chat 1M / AlpacaEval 46.5, HumanEval 17.7, AGIEval English 18.7, MMLU 10.3,
BBH 10.6, GSM8K 8.9; WildJailbreak / WildGuardTest 8.2, HarmBench 6.3.

Removal rates after decontamination (Table 8): Evol CodeAlpaca 3.5%, WildChat GPT-4 5.4%,
WildJailbreak 0.7%, WildGuardmix 1.1%, NuminaMath-TIR 11.3%.

## Limits recorded by the authors

- No unseen safety evaluation exists in the suite (§2.2); ch-53 §8 supplies one.
- For cross-model comparisons the authors cannot verify whether other models trained on GPQA,
  MMLU-Pro, AGIEval, DeepMind Mathematics, or BigCodeBench (§7.4.2).
- Data choices overfit the development evaluations in precise instruction following, and to some
  extent in knowledge recall and reasoning (§7.4.1, Table 32).
- The DPO data-scaling trend generalized on average, but the development/unseen math pair diverged;
  the authors attribute this to LaTeX formatting habits interfering with answer extraction on
  DeepMind Mathematics (§7.4.1, Fig. 24).

Related: [[tulu-3]], [[read]], [[paired-statistics-and-power]], [[minhash-contamination-gate]].
