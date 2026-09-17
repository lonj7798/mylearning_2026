---
chapter: ch-31a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/chain-of-hindsight.md on 2026-09-15)
source_url: https://arxiv.org/abs/2302.02676
source_version: arXiv v8 (2023-10-18); v1 2023-02
created_at: "2026-09-15"
---

# Excerpt: Chain of Hindsight Aligns Language Models with Feedback (Liu, Sferrazza, Abbeel; UC Berkeley)

Facts used by [[read]], read in the arXiv v8 PDF on 2026-09-15. Earlier arXiv versions differ in content; loci refer to v8.

## Method (§2; Algorithm 1; App. B-C)
- Model outputs with human ratings are turned into one training sequence with templated feedback, for example "How to explain neural networks to a child? Bad: {a subpar answer} Good: {an excellent answer}" (§2, Fig. 2). Natural-language templates include "A good summary: {positive}, a worse summary: {negative}" (§2).
- Loss masking (§2): "Loss is not applied on other tokens because it hinders model generation at inference time." Feedback tokens are masked; model-generated tokens are predicted. The training paragraph also states "We average the loss over each timestep in the last model output sequence."
- 0% to 5% of past tokens are randomly masked during training so the model does not copy the paired example (§2 Training).
- A pretraining-data log-likelihood term is added to retain general language modeling; its weight λ = 1.5 and it is computed on the Pile (§2 Training; App. C).
- At inference the model is prompted with positive feedback ("Good"); "During inference time, we only employ simple positive tokens" (§2; App. B).
- Hyperparameters (App. C): Adam β1 = 0.9, β2 = 0.95, ε = 1.0e−8; batch size 512 for human-feedback data and 2048 for pretraining data; no dropout; the three feedback datasets are sampled in proportion to size.

## Data and models (§3)
- WebGPT comparisons (19,578), Anthropic HH, and the summarize-from-feedback comparisons. Base models: GPT-J 6B and OPT. Baselines: SFT, SFT with unlikelihood on negatively rated data (SFT-U), conditional SFT (C-SFT), RLHF with PPO.

## Results
- Summarization human evaluation, average win rate (Table 1; 75 labelers, pairwise): CoH 45.3 vs RLHF 30.8 (tie 24.0); CoH 44.0 vs SFT 28.2 (tie 27.9); CoH 42.3 vs C-SFT 29.6; CoH 61.7 vs SFT-U 21.4.
- Dialogue human evaluation, average (Table 2): CoH 36.9 vs RLHF 23.4 (tie 39.8); CoH 39.4 vs SFT 19.1 (tie 41.5); CoH 56.0 vs SFT-U 13.9.
- Dialogue automatic evaluation (Fig. 4; metric: accuracy of classifying which response of an HH dialogue pair is preferred): the text states that "adding unlikelihood degrades performance which indicates unlikelihood hurts model generation ability" and that conditional SFT improves over SFT (§4).
- Model scale (Fig. 5): "for smaller model sizes, CoH exhibits a marginal decrement in performance compared to SFT baselines"; at larger sizes it surpasses SFT and RLHF (§4). Exact sizes and values are shown only in the figure.
- Language feedback ablation (Table 3): CoH vs CoH without language feedback, 15.1 vs 10.6 (tie 74.3).
- Alignment tax (App. E.2, Table 7; lm-evaluation-harness, 5 seeds): average zero-shot GPT-J 40.60, SFT 40.54, CoH 40.95; one-shot 42.53, 42.53, 43.14; few-shot 39.38, 39.59, 39.98. GPT-J numbers are taken from its original paper.

## Limits stated by the source (§6)
- CoH sequences can be long, raising training compute; evaluation relies on hired human labelers.
