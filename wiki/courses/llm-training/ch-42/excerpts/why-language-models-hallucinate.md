---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: arXiv:2509.04664v1 (Why Language Models Hallucinate); no library card exists for this slug yet
source_url: https://arxiv.org/abs/2509.04664
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because the library has no card for this source)"
---

# Excerpt: binary grading, abstention, and confidence targets (Kalai, Nachum, Vempala, Zhang)

Used by [[read]] §3 and the negatives section. Checked against arXiv v1 (2025-09-04, "arXiv:2509.04664v1 [cs.CL] 4 Sep 2025") on 2026-09-15.

## Definition of a binary grader (§4.1)
For a prompt `c`, let `R_c` be the plausible responses and `A_c ⊂ R_c` the abstention responses ("I don't know", IDK).
A grader `g_c : R_c → R` is **binary** if `{g_c(r)} = {0, 1}` and `g_c(r) = 0` for every `r ∈ A_c`.

> "Observation 1. Let c be a prompt. For any distribution ρ_c over binary graders, the optimal response(s) are not abstentions, i.e. A_c ∩ arg max_{r∈R_c} E_{g_c∼ρ_c}[g_c(r)] = ∅."

The paper states the consequence directly: "Under binary grading, abstaining is strictly sub-optimal. IDK-type responses are maximally penalized while an overconfident 'best guess' is optimal" (§4.1).

## Confidence targets (§4.2)
Proposed instruction text to append to each question:

> "Answer only if you are > t confident, since mistakes are penalized t/(1 − t) points, while correct answers receive 1 point, and an answer of 'I don't know' receives 0 points."

- "There are several natural values of t including t = 0.5 (penalty 1), t = 0.75 (penalty 2), and t = 0.9 (penalty 9)." The printed penalty for t = 0.75 does not match the stated formula `t/(1−t) = 3`; t = 0.5 and t = 0.9 do match (inconsistency inside the source).
- "A threshold of t = 0 corresponds to binary grading."
- "A simple calculation shows that the expected score of offering an answer beats IDK (score 0) iff its confidence (i.e., probability of being correct) is > t."
- **Behavioral calibration** (§4.2): rather than emitting a probability, the model "must formulate the most useful response in which it is at least t confident", and this "can be audited by comparing accuracy and error rates across thresholds".

## Meta-evaluation of mainstream benchmarks (§4.1, Table 2)
Binary grading / IDK credit, as printed: GPQA Yes/None; MMLU-Pro Yes/None; IFEval Yes (composite of binary sub-scores)/None; Omni-MATH Yes/None; WildBench No/Partial; BBH Yes/None; MATH (L5 split) Yes/None; MuSR Yes/None; SWE-bench Yes/None; HLE Yes/None. Footnote b notes that WildBench's 1-10 rubric "suggests that IDK may score lower than 'fair' responses with hallucination, reinforcing hallucination".

The authors argue that adding hallucination-specific evaluations is not enough: "even the ideal hallucination evaluation and ideal post-training methodology, yielding honest reports of uncertainty, may still be drowned out because of inferior performance on the vast majority of the existing evaluations" (§4.1).

## Pretraining part (§3, used only as background in the chapter)
The paper reduces generative error to a binary classification problem ("Is-It-Valid") and states `(generative error rate) ≳ 2 · (IIV misclassification rate)` (§1).

## Not in this source
- No experiment that trains a policy with a `t/(1−t)` penalty; the proposal is for evaluations.
- No abstention rates for named deployed models.
