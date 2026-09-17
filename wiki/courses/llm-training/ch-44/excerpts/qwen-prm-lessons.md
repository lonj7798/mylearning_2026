---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: arXiv:2501.07301v2 (The Lessons of Developing Process Reward Models in Mathematical Reasoning)
source_url: https://arxiv.org/abs/2501.07301
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because no library card exists yet)"
---

# Excerpt: Monte-Carlo step labels, consensus filtering, and the bias of Best-of-N evaluation

Used by [[read]] §3 and the negatives section. Authors: Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, et al. (Qwen Team, Alibaba Group). arXiv v1 2025-01-13; this excerpt follows v2 (2025-06-05), checked on 2026-09-15. Released models: Qwen2.5-Math-PRM-7B and -72B.

## Setup (§2.1)
- About 500,000 queries with golden answers; 6-8 responses per query from Qwen2-Math-Instruct and Qwen2.5-Math-Instruct at 7B and 72B; steps split on "\n\n".
- Step label from 8 independent completions with Qwen2.5-Math-Instruct of the same size. Hard label: correct if any of the 8 completions reaches the correct final answer. Soft label: fraction of completions that do.
- All steps after a step labelled 0 are removed.
- PRMs initialized from Qwen2.5-Math-7B/72B-Instruct with the LM head replaced by a two-layer scalar head; cross-entropy on the last token of each step for hard labels, MSE for soft labels.

## The two claims about MC estimation (§3.1)
1. **A PRM and a value model are different objects.** A PRM is meant to judge whether the current step is correct; an MC estimate measures the probability of reaching the correct final answer from that step, which depends on the completion model. A completion model can reach a correct answer from an incorrect step, or fail from a correct step (§3.1.1, §3.1.2).
2. **Data-source comparison, identical 860k queries and responses** (Tables 3 and 4). Best-of-8 average over 7 math benchmarks with a Qwen2.5-Math-7B-Instruct policy, and ProcessBench average F1:

| Step labels | # samples | Best-of-8 avg | ProcessBench avg F1 |
|---|---|---|---|
| MC estimation (Math-Shepherd data) | 440k | 64.3 | 28.9 |
| MC estimation (this paper's data) | 860k | 65.9 | 40.1 |
| LLM-as-a-judge (Qwen2.5-72B-Instruct, same data) | 860k | 65.3 | 46.5 |
| Human annotation (PRM800K) | 264k | 64.9 | 56.5 |
| maj@8 baseline / pass@8 upper bound | — | 66.2 / 74.7 | — |

The two evaluations order the same models in opposite directions (§3.1.2, §3.2).

## Consensus filtering (§3.1.3, §4.1)
- Keep an instance only when the LLM judge and the MC estimate agree on the location of the first error. About 40% of the 860k set is retained (Fig. 2).
- The filtered 350k set reaches ProcessBench 46.3 F1, close to LLM-as-a-judge on 860k, with Best-of-8 differences described as marginal (Fig. 2).
- Threshold study on a 3M set, hard-label threshold from 1/8 to 7/8 (Fig. 5): performance on both Best-of-8 and ProcessBench falls as the threshold rises. The paper recommends threshold 0, that is, a step is negative only when none of the 8 completions reaches the correct answer.
- Soft labels: a correct step can receive a target below 1, which the authors state reduces the model's ability to separate positive from negative; 8 completions give a high-variance estimate (§3.1.4).

## Why Best-of-N alone misleads (§3.2)
- Policy models produce responses with correct answers and flawed processes. Manual annotation of correct-answer samples from Qwen2.5-Math-7B-Instruct (Fig. 6) gives process-error rates of 5.1% (GSM8K), 11.9% (MATH), 27.4% (OlympiadBench), and 43.4% (Omni-MATH); values read from the figure labels.
- On ProcessBench cases that have a correct answer and an erroneous step (Table 5), every open-source PRM other than the authors' scores below 50% detection accuracy (for example Math-Shepherd-PRM-7B 13.9, Qwen2.5-Math-7B-PRM800K 38.2, Qwen2.5-Math-PRM-7B 53.9).
- Process-to-outcome shift (Fig. 8): the share of responses whose minimum step score falls on the final step exceeds 40% for EurusPRM-Stage1/2, Math-Shepherd-PRM-7B, and Skywork-PRM-7B; it is 17.5% for Qwen2.5-Math-PRM-7B.
- Scoring strategy (Fig. 9): for MC-trained PRMs the last-step score beats product and minimum; for human-annotated and LLM-judge PRMs the order is reversed.

## Final models (Tables 6, 7)
- Qwen2.5-Math-PRM-7B: Best-of-8 average 67.6 (maj@8 66.2, pass@8 74.7), ProcessBench average F1 73.5. Qwen2.5-Math-PRM-72B: 69.3 and 78.3.
- Reference points on ProcessBench: o1-mini 87.9, GPT-4-0806 61.9, QwQ-32B-Preview 71.5, the ORM Qwen2.5-Math-RM-72B 38.9.
- Stated limitation: the gap to pass@8 remains, and the best way to use these PRMs inside RL is not studied (Limitation).
