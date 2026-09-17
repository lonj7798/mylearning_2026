---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/why-language-models-hallucinate.md on 2026-09-15)
source_url: https://arxiv.org/abs/2509.04664
source_version: arXiv v1 (2025-09-04)
created_at: "2026-09-15"
---

# Excerpt: Why Language Models Hallucinate (Kalai, Nachum, Vempala, Zhang; OpenAI and Georgia Tech)

Facts used by [[read]], read in the arXiv v1 PDF on 2026-09-15.

## Claim used by this chapter (§4.1)
- A grader g_c is binary when its range is {0, 1} and it gives 0 to every abstention. Observation 1: for any distribution over binary graders, no abstention is among the optimal responses, so "under binary grading, abstaining is strictly sub-optimal. IDK-type responses are maximally penalized while an overconfident 'best guess' is optimal".
- Table 2 classifies ten widely used benchmarks (GPQA, MMLU-Pro, IFEval, Omni-MATH, WildBench, BBH, MATH level-5 split, MuSR, SWE-bench, HLE); all except WildBench use binary grading, and only WildBench gives partial credit to an abstention.
- The authors' conclusion is socio-technical: adding hallucination-specific evaluations does not fix the incentive while the primary evaluations keep scoring abstention as a failure (§4.1).

## Explicit confidence targets (§4.2)
- Proposed instruction text: "Answer only if you are > t confident, since mistakes are penalized t/(1 − t) points, while correct answers receive 1 point, and an answer of 'I don't know' receives 0 points." The paper names t = 0.5 (penalty 1) and t = 0.9 (penalty 9) as natural values, and t = 0 as ordinary binary grading.
- "A simple calculation shows that the expected score of offering an answer beats IDK (score 0) iff its confidence (i.e., probability of being correct) is > t."
- Behavioral calibration: with an explicit target, one behaviour is optimal at every threshold — answer when the model's correctness probability exceeds the target and abstain otherwise — and it can be audited by comparing accuracy and error rates across thresholds.

## Calibration and post-training (§3.1, Figure 2)
- Figure 2 reprints the GPT-4 multiple-choice calibration histograms before and after reinforcement learning (OpenAI 2023a, Figure 8): "The pretrained model is well calibrated." The text states that base models are often found to be calibrated "in contrast to post-trained models which may deviate from cross-entropy in favor of reinforcement learning".
