---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/topr-tapered-off-policy-reinforce.md on 2026-09-15)
source_url: https://arxiv.org/abs/2503.14286
source_version: arXiv v2 (2025-03-19); v1 2025-03-18
created_at: "2026-09-15"
---

# Excerpt: Tapered Off-Policy REINFORCE — Stable and efficient reinforcement learning for LLMs (Le Roux, Bellemare, Lebensold, Bergeron, Greaves, Fréchette, et al.; Mila and Reliant AI)

Facts used by [[read]], read in the arXiv v2 PDF on 2026-09-15.

## Why naive off-policy REINFORCE breaks (§2.1)
- With `∇Ĵ_µ(π) = R(τ) ∇ log π(τ)` and data from µ, the expected objective is `Σ_{τ∈T⁺} µ(τ) log π(τ) − Σ_{τ∈T⁻} µ(τ) log π(τ)` (Eq. 4). The negative term "is unbounded above (in terms of π) and can be made arbitrarily large by driving the probability of any single trajectory supported by µ to zero", which drives logits to −∞ and eventually produces degenerate behaviour. The paper notes this does not arise on-policy, because a trajectory whose π(τ) is small is unlikely to be sampled.
- Early stopping, a reward baseline, and KL regularization all mitigate it by "fully or partially ignoring negative trajectories", which limits how much off-policy learning is possible (§2.1, §3.1).

## TOPR (§3, Eqs. 7-8; Table 1)
- General form with truncation limits `[a⁺, b⁺]` for positives and `[a⁻, b⁻]` for negatives applied to the importance ratio π(τ)/µ(τ).
- Canonical TOPR: `a⁻ = 0`, all other limits 1, so the gradient is the SFT update on positives plus a truncated-importance-sampling update on negatives: `Σ_{T⁺} µ(τ)R(τ)∇log π(τ) + Σ_{T⁻} µ(τ)[π(τ)/µ(τ)]₀¹ R(τ)∇log π(τ)`.
- Stated reasons: `a⁻ = 0` lets the contribution of a negative trajectory decay as the policy moves away from it ("any a⁻ > 0 must eventually lead to model degeneracy as with naive REINFORCE"); `a⁺ = 1` keeps a minimum learning rate on positives; the upper limits control variance. Table 1 places SFT, naive REINFORCE, off-policy REINFORCE and truncated importance sampling in the same parameterisation.
- The paper notes PPO's clipping "is not incentivized to reduce the relative probability of negative trajectories below 1 − ε" and that PPO's gradient is zero outside [1−ε, 1+ε] (§2.4).

## Setup (§4.1-4.2)
- Llama 3 8B Instruct (also a 70B model for data generation, and DeepSeek-R1 8B in §4.3); GSM8K (16 candidate solutions per training question) and MATH (32 per question, evaluated on MATH-500); single H100 node, constant learning rate 5e-7, Adafactor, no weight decay and no KL regularization, gradient clipping 1.0, loss divided by sequence length, reward +1 correct / −1 incorrect with no baseline, generation at temperature 1, top-p 1, top-k 500, maximum 512 tokens; one epoch over each generated dataset.

## Results used here
- Figure 1 and Figure 3 (GSM8K, off-policy fine-tuning): naive REINFORCE improves first and then collapses, and its generations are "overwhelmingly degenerate by the end of training"; PPO "makes little progress" because most data points fall outside [1−ε, 1+ε] with ε = 0.2; DPO performs well off-policy; TOPR is the only one of the four that substantially reduces invalid generations (no "The answer is" string).
- Positives-only ablation (Figure 4): removing negative examples is stable but reaches lower pass@1 and maj@16 on GSM8K; the gain from negatives comes from "reducing the number of questions for which no or few solutions are found".
- Dataset composition (§4.3, Figure 5): with the baseline c varied, the effective positive proportion is `p̃ = p(1−c)/(1 + c(1−2p))`; performance peaks around 10-20% effective positives for both a 10%-positive and a 50%-positive training set and "markedly decreases as the proportion of positive examples goes above 50%"; discarding negatives is equivalent to a baseline of c = −1.
- Ratio truncation (§4.3, Figure 7): with a negatively skewed dataset (60% negatives) and gradient clipping raised to 100.0, standard (untruncated) importance sampling "harms the model's performance - producing 31% of bad reasonings by the end of training against 12% for the base model", while TOPR still improves on the base model.
- Generative verifier trained with TOPR (§4.4, Table 2): verifier accuracy 32.6% (Llama 3 8B) → 70.9% (8B TOPR); invalid rate 34.2% → 0.90%; weighted self-consistency with 32 generations 55.5% (no verifier), 56.7%, 61.5%. The authors attribute the drop in invalid outputs to their being "negatively rewarded".
