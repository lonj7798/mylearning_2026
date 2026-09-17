---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/transferability-of-llm-reasoning.md
source_url: https://arxiv.org/abs/2507.00432
created_at: "2026-09-15"
---

# Excerpt: Does Math Reasoning Improve General LLM Capabilities? (Huan et al.)

**Checked on 2026-09-15 against arXiv:2507.00432v2 (2025-10-20).** The library card had no verification section and no numbers at that date; this excerpt is the checked extract used by ch-24 §7, the Recipe, and the Generalization lens.

## Controlled study (§2.2, App. A.3)

- Base: Qwen3-14B-Base. Data: 47K math problems (DeepScaleR low difficulty, SimpleRL levels 3–5).
- SFT: Qwen3-32B traces kept by rejection sampling on correct answers (thinking and non-thinking modes); LLaMA-Factory; LR 5×10⁻⁵; batch 512; 1.5 epochs.
- RL: GRPO in verl with answer correctness as reward; LR 1×10⁻⁶; batch 512; 16 rollouts per prompt; sequences up to 16k tokens; clipping "between 0.22 and 0.28"; KL and entropy coefficients 0; 140 steps.

## Table 1

| Model | Math avg | Other reasoning avg | Non-reasoning avg | IFEval | TI other | TI non |
|---|---|---|---|---|---|---|
| Qwen3-14B-Base | 27.7 | 30.2 | 45.7 | 69.2 | – | – |
| SFT (think) | 49.8 | 45.3 | 21.1 | 42.3 | +52.2 | −104.1 |
| SFT (no-think) | 32.3 | 45.2 | 29.0 | 41.4 | +165.4 | −278.9 |
| RL | 53.8 | 60.0 | 53.2 | 70.0 | +82.3 | +52.2 |

Math: AIME24, AIME25, MATH500, OlympiadBench. Other reasoning: GPQA, LiveCodeBench, ACPBench, HeadQA. Non-reasoning: CoQA, MC-TACO, IFEval, HaluEval.
The transferability index (TI) is the z-normalized, square-root-compressed, difficulty-weighted gain of a domain group divided by that of math, × 100 (§2.1).

## Ablation (§5.2, Table 4; Qwen3-8B-Base, same math queries)

| Setting | Math | Other | Non-reasoning | TI other | TI non |
|---|---|---|---|---|---|
| Base | 27.6 | 23.6 | 33.6 | | |
| Off-policy SFT (Qwen3-32B traces) | 41.9 | 34.4 | 26.6 | 18.3 | −40.5 |
| On-policy SFT (rejection sampling from the policy) | 33.7 | 35.7 | 35.0 | 68.6 | 30.2 |
| Off-policy RL | 45.5 | 35.9 | 31.7 | 36.4 | 4.5 |
| On-policy RL, no KL | 37.1 | 38.2 | 35.8 | 65.6 | 39.3 |
| On-policy RL | 38.6 | 39.9 | 35.0 | 63.7 | 32.4 |

- Authors' reading: the sampling distribution is the most important factor; credit assignment and negative gradients help and are ablated together; KL has a small effect (§5.2, §6).
- Audit of more than 20 released models: SFT-trained models often have negative non-reasoning TI; RL-trained models have higher TI on both groups (§2.1, Fig. 2).
- Not matched in the controlled study: learning rate (5×10⁻⁵ SFT vs 1×10⁻⁶ RL).
