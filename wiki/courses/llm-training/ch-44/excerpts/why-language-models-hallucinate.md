---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: arXiv:2509.04664v1 (Why Language Models Hallucinate)
source_url: https://arxiv.org/abs/2509.04664
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because no library card exists yet)"
---

# Excerpt: Binary grading, wrong-answer penalties, and the abstention threshold

Used by [[read]] in the negatives section. Authors: Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, Edwin Zhang (OpenAI, Georgia Tech). arXiv v1 2025-09-04; checked on 2026-09-15.

## The claim about scoring (§4.1)
- A grader `g_c: R_c → R` is binary when its range is {0, 1} and every abstention response ("I don't know") receives 0.
- Observation 1: under any belief distribution over binary graders, no abstention is among the optimal responses. Under binary grading a guess weakly dominates an honest expression of uncertainty.
- Table 2 classifies ten widely used benchmarks; GPQA, MMLU-Pro, IFEval, Omni-MATH, BBH, MATH (L5 split), MuSR, SWE-bench, and HLE are listed as binary grading with no credit for abstention; WildBench is not binary and gives partial credit.

## The proposed instruction (§4.2)
The paper proposes stating the scoring rule inside the prompt, for example:

> Answer only if you are > t confident, since mistakes are penalized t/(1 − t) points, while correct answers receive 1 point, and an answer of "I don't know" receives 0 points.

- Named values, as printed: `t = 0.5` gives penalty 1, `t = 0.75` gives penalty 2, `t = 0.9` gives penalty 9. `t = 0` is ordinary binary grading. Note that `0.75/(1 − 0.75) = 3`, so the middle value as printed does not follow the paper's own `t/(1 − t)` formula.
- The expected score of answering beats the score of abstaining (0) exactly when the probability of being correct exceeds `t`.
- The paper calls the behaviour that is optimal at every threshold **behavioral calibration**: answer when the correctness probability exceeds the stated threshold, otherwise abstain. It can be audited by comparing accuracy and error rate across thresholds.
- The paper's proposal is about evaluation scoring; it also names SWE-bench as a target for the change. Applying the same arithmetic to an RL reward function is an extension made by this course, not a claim of the paper.

## Arithmetic used in the chapter
With reward `+1` for a correct answer, `−w` for a wrong answer, and `0` for an abstention, the expected reward of answering with correctness probability `p` is `p − w(1 − p)`, which exceeds 0 when `p > w/(1 + w)`. Setting `w = t/(1 − t)` returns the paper's threshold `t`.
