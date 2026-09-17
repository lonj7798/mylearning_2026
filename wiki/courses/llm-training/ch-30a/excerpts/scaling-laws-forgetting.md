---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/scaling-laws-forgetting.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2401.05605
created_at: "2026-09-15"
---

# Excerpt: Scaling Laws for Forgetting When Fine-Tuning Large Language Models

**Author:** Damjan Kalajdzievski (Tenyx)
**Version read:** arXiv:2401.05605v1 (11 Jan 2024), the only version read.
**Status:** no library card existed for this slug on 2026-09-15; quotes checked against the v1 PDF text.

## Forgetting metric (§3.1)
The metric L_f is "the usual next token prediction loss ... except where the target next token is provided by the pre-trained base model's prediction instead of the data." The author gives three reasons to prefer it to ground-truth metrics, including: "If there is some transfer learning between some subsets of D′ and D, some subset of D may be forgotten while improving the evaluations on another subset, thus obfuscating forgetting."

Empirical example (§3.1): Llama 2 chat 7B fine-tuned on 6400 OpenOrca examples (200 gradient steps), evaluated on ARC-Challenge: "the pre-trained Llama 2 chat 7B model M achieved 54.1% accuracy, and the fine-tuned model M′ achieved 52.0% accuracy, while the accuracy of M′ evaluated against the answer predicted by M was only 67.2%."

## Setup (§3.2)
LoRA with rank-stabilized scaling γ_r = 1/√r on all linear attention and MLP modules of Llama 2 chat 7B; datasets OpenOrca and "News" (100 web articles from Sept. 2023); forgetting evaluated with L_f on WikiText-103 test; "260 update steps", first 50 steps excluded, Adafactor, context 512, batch size 32, ranks r ∈ {8, 16, 32, 64, 128, 256}.

## Laws (§4.1)
- Eq. 4: L_f(L_ft) = −c_{f,ft} L_ft + s_{f,ft}, "with constants c_{f,ft} ≈ 1.7334, s_{f,ft} ≈ 2.0481 for OpenOrca, and c_{f,ft} ≈ 1.0615, s_{f,ft} ≈ 3.1285 for News."
- Fig. 2 caption: "coefficients of determination .9450 and .9736 for OpenOrca and News respectively."
- Eq. 5-6: fine-tuning loss and forgetting are shifted power laws in the number of fine-tuned parameters P and update steps N; the joint fit raises R² to .9598 and .9769.
- Interpretation stated by the author: "forgetting is unavoidable by early stopping or by tuning a fewer (or greater) number of parameters", and "the higher forgetting of larger models, is predominantly due to larger models being able to achieve better fine-tuning loss."
- App. B (OpenOrca; IA3, top-3-layer tuning, rank-64 attention-only LoRA, LoRA ranks 1000 and 2500, full fine-tuning): "all fits predict the fine-tuning loss and forgetting for large ranks, rank 64 adapters to attention modules only, and full model tuning, but L_f(L_ft) underestimates the forgetting for IA3 and tuning the top 3 layers ... The line L_f(L_ft) has an R2 of only .1851, while L_f(P, N) improves the generalization to .8714 R2."

## Safety observation (§4.2; the example generations in Figs. 5 and 7 are from rank-8 models)
"the base pre-trained model rejected the instructions 32 out of 50 times, whereas the News model only rejected 24 out of 50, and the OpenOrca model rejected just 16 out of 50" (AdvBench, hand-inspected). "the News and OpenOrca models forgot to reject a previously recognized harmful instruction 25% and 50% of the time respectively."

## How ch-30a uses it
§1 (agreement with base predictions as a forgetting signal), §2 (Eq. 4 worked example), §4 (safety), Recipe row.
