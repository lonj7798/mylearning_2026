---
chapter: ch-45a
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2503.18892
primary_version: arXiv:2503.18892v3 (6 Aug 2025; v1 Mar 2025)
created_at: "2026-09-15"
---

# Excerpt: SimpleRL-Zoo — one hyper-parameter set across ten base models

Zeng, Huang, Liu, Liu, He, Ma, He (HKUST, TikTok, Meituan), 2025. ch-45a uses this source as the row where a single RL configuration is held fixed
and the base model is varied, which is the cleanest available evidence that a recipe row does not transfer
across base models on its own.

## Settings (App. B.5, verbatim)
> "We train our models using the verl (Sheng et al., 2024) framework. And we typically use the same set of
> hyperparameters to train and evaluate all models in the SimpleRL-Zoo series in default main experiment
> setting. We use a prompt batch size of 1,024 and generate 8 rollouts per prompt, with a maximum rollout length
> of 8,192 tokens. Training is performed using a mini-batch size of 256. The default sampling temperature is set
> to 1.0, and the clip ratio is 0.2. For models ranging from 0.5B to 14B parameters, we use a KL loss
> coefficient of 1e-4. For models larger than 14B, the KL loss coefficient is set to 1e-3."

The KL term is in the loss, not in the reward (App. A, the token-level length-rectified GRPO objective). Prompt
batch 1,024 and mini-batch 256 are in different units: 1,024 prompts × 8 rollouts = 8,192 samples, so the
mini-batch of 256 samples implies 32 gradient updates per rollout step if the mini-batch unit is samples.
The paper does not state the mini-batch unit.

Evaluation: temperature 1.0, maximum generation length 16K, same prompt template as training; pass@1 for most
benchmarks and avg@32 additionally for AIME 2024.

## Data difficulty must match the base model (§3.2, Fig. 7)
> "as data difficulty increases, Mistral-7B's performance progressively deteriorates. When faced with
> high-difficulty data (Hard: MATH levels 3-5), the model struggles to generate responses that receive positive
> feedback from the reward system. This failure results in a significant increase in response length without any
> corresponding improvement in accuracy, signaling a breakdown in the training process—often referred to as
> training collapse."

> "Qwen-2.5-7B exhibits a pattern entirely opposite to Mistral-7B-v0.1. Specifically, as dataset difficulty
> decreases, both the model's average accuracy and response length decline."

The paper's conclusion: "zero RL training data must align with the base model's inherent reasoning
capabilities" — one of its four stated key factors (§1 item 4).

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2503.18892 (scratchpad `sources/simplerl-zoo.txt`),
  §1, §3.2, App. A, App. B.5, Fig. 7.
- Not reported: the RL learning rate (checked §2, §3, App. A, App. B.5); optimizer; number of RL steps per model.
