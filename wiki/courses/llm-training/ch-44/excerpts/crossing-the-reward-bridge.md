---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: arXiv:2503.23829v2 (Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains)
source_url: https://arxiv.org/abs/2503.23829
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because no library card exists yet)"
---

# Excerpt: Model-based verifiers for domains without structured answers

Used by [[read]] §7 and the generalization lens. Authors: Yi Su, Dian Yu, Linfeng Song, Juntao Li, Haitao Mi, Zhaopeng Tu, Min Zhang, Dong Yu (Tencent AI Lab, Soochow University). arXiv v1 2025-03-31; this excerpt follows v2 (2025-04-01), checked on 2026-09-15.

## Problem (§1)
- In the paper's own exam data, 60.3% of mathematics problems have a single-term numerical answer that a rule-based verifier can check; for multi-domain queries the share is 45.4%.
- Reference answers in the study are free-form: average length 33.7, 36.3, and 53.9 words for elementary, middle, and high-school math test sets, against 1 word for GSM8K and 1.3 for MATH (§4.1).

## Method (§3)
- Binary model-based reward: a generative verifier is instructed to emit only 0 or 1 given the prompt, the reference answer, and the final step of the response (Eq. 3).
- Soft reward: the probability of the judgment token, `π_φ(1 | x, a, y_T)` when the judgment is 1 and `1 − π_φ(0 | ...)` when it is 0 (Eq. 4); the value stays in [0, 1].
- Rewards are z-normalized within the batch; when the standard deviation is zero all normalized rewards are set to zero (Eq. 5).
- A KL penalty with `β = 0.01` against the base model is subtracted from the normalized reward (Eq. 6).
- The 7B reward model is trained by supervised fine-tuning of Qwen2.5-7B-Instruct on 160k `(x, a, y, c)` judgments collected from RL exploration and labelled by Qwen2.5-72B-Instruct (§3.3, §4.4).

## Results (§4.5, Table 1; base model Qwen2.5-7B, 30k training prompts)
| Setting | Math avg | Multi-subject avg |
|---|---|---|
| Base (no RL) | 43.4 | 15.0 |
| SFT | 45.7 | 23.1 |
| RLOO, rule-based binary | 58.5 | 26.3 |
| RLOO, RM-7B binary | 63.0 | 28.1 |
| REINFORCE, RM-7B soft | 62.2 | 31.2 |
| REINFORCE, rule-based binary | 57.2 | 24.2 |

(The §4.5 text reports the rule-based RLOO math score as 58.8 where Table 1 prints 58.5.)

## Scaling and out-of-distribution (§4.6-§4.7, Tables 2-3)
- RLOO with binary rewards, data scaled from 20k to 100k: rule-based reward moves 58.2 → 51.7 on math and 26.2 → 16.9 on multi-subject; RM-7B moves 63.4 → 65.0 and 30.8 → 35.0.
- Out-of-distribution training sets of 30k examples: NaturalReasoning 29.4 (rule-based) against 39.8 (RM-7B); WebInstruct 33.9 against 44.0.
- Agreement check: majority-voted judgments from Qwen2.5-72B-Instruct against single GPT-4o judgments give Cohen's κ above 0.86 for mathematics and above 0.88 for multi-subject questions, which the authors use to justify one evaluation sample per response (§4.3).
