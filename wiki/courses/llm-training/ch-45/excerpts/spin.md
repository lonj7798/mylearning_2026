---
chapter: ch-45
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/spin.md (card not re-verified; see Corrections)
source_url: https://arxiv.org/abs/2401.01335
version: arXiv v3, 2024-06-14 (ICML 2024)
verified: 2026-09-15 (rewritten from the primary text; replaces the 2026-04 excerpt)
---

# Excerpt: SPIN — Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models

**Used by:** [[read]] §1, §3, §6, Recipe

## Objective (§4.1, Eq. 4.7)

```
L_SPIN(θ, θ_t) = E[ ℓ( λ·log( p_θ(y|x) / p_{θ_t}(y|x) ) − λ·log( p_θ(y'|x) / p_{θ_t}(y'|x) ) ) ]
```

with x ~ q (prompts), y ~ p_data(·|x) (the human SFT response), y′ ~ p_{θ_t}(·|x) (a sample from the previous iterate). ℓ must be convex and monotonically decreasing (Assumption 5.1); with the logistic loss the objective coincides in form with DPO, and λ plays the role of DPO's β. The opponent at iteration t+1 is a copy of θ_{t+1} (Algorithm 1).

Theorem 5.2: p_{θ_t} = p_data is a global minimum for any λ ≥ 0, and if p_{θ_t} ≠ p_data there is a λ for which θ_t is not a global minimum.
Theorem 5.4 (logistic loss): p_{θ_{t+1}}(y|x) ∝ p_{θ_t}(y|x)·[p_data(y|x)/p_{θ_t}(y|x)]^{1/λ}. Remark 5.5: a smaller λ produces a larger change of the opponent player.

## Setup (§6.1, App. B.1)

- Base: `zephyr-7b-sft-full` (Mistral-7B fine-tuned by HuggingFace on UltraChat200k).
- 50k prompts sampled from UltraChat200k; only the first round of each conversation is used as prompt and ground-truth completion.
- Synthetic set: 50k at iteration 0, 100k at iterations 1–3 (the current iteration's samples plus the previous iteration's).
- 2 epochs per iteration; RMSProp, no weight decay; global batch 64; 10% warmup; bfloat16; max sequence length 2,048.
- Peak learning rate 5e−7 for iterations 0 and 1, decayed to 1e−7 for iterations 2 and 3. λ (written β in the appendix) = 0.1, raised to 5.0 at iteration 3.
- Prompt template: `### Instruction: {prompt}\n\n### Response:`.
- Cost: generation 1.45 h per iteration and training 4.32 h (iteration 0) or 8.64 h (iterations 1–3) on 8× A100 80G (App. B.2, Table 2).

## Results as printed (Tables 4–6)

| Model | Arc | TruthfulQA | Winogrande | GSM8k | HellaSwag | MMLU | Average | MT-Bench |
|---|---|---|---|---|---|---|---|---|
| zephyr-7b-sft-full | 60.41 | 43.73 | 74.19 | 26.76 | 82.85 | 60.92 | 58.14 | 5.94 |
| SPIN iteration 0 | 63.40 | 49.18 | 72.69 | 35.10 | 84.38 | 60.03 | 60.80 | 6.46 |
| SPIN iteration 1 | 65.19 | 55.17 | 72.30 | 35.78 | 84.96 | 59.34 | 62.12 | 6.65 |
| SPIN iteration 2 | 65.96 | 54.91 | 73.56 | 38.06 | 85.41 | 59.93 | 62.97 | 6.78 |
| SPIN iteration 3 | 65.87 | 54.90 | 73.72 | 38.97 | 85.54 | 59.99 | 63.16 | not reported |

Other rows: `zephyr-7b-dpo-full` (DPO on 62k UltraFeedback pairs) averages 61.31; one more SFT epoch on UltraChat200k lowers the average to 57.23 (Table 5); SPIN iteration 3 followed by DPO on the 62k pairs reaches 64.05 (Table 3). Big-Bench-Hard and OpenBookQA (Table 6) move within about 1–3 points with no large degradation.

Limitation stated by the authors (§7): the target data distribution is fixed and human-generated, which "inherently imposes a ceiling on the performance of fine-tuned LLM".

## Corrections to the library card `papers/spin.md`

1. "MT-Bench goes 6.39 → 7.12 across 3 iters" → 5.94 (base) → 6.46 → 6.65 → 6.78 (Table 6); no iteration-3 MT-Bench value is printed.
2. "β=0.1, lr=5e-7, 3 epochs, batch 64" → 2 epochs per iteration; learning rate 5e−7 only for iterations 0–1, then 1e−7; λ raised to 5.0 at iteration 3 (App. B.1).
3. "Sample 50K pairs from π_{t−1} at T=1.0" → the sampling temperature is not stated; the synthetic set is 50k at iteration 0 and 100k afterwards because the previous iteration's samples are kept (§6.1).
4. "Reset reference to π_{t−1}" → the previous iterate is both opponent and reference inside the same loss; there is no separate reset step (Eq. 4.7, Alg. 1).
5. "Zephyr-7B-SFT + 3 SPIN iters matches Zephyr-7B-DPO" → SPIN iteration 1 onward is above zephyr-7b-dpo-full on the leaderboard average (62.12 vs 61.31, Table 3).
