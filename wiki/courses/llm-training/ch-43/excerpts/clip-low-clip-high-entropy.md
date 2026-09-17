---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2509.26114
primary_version: arXiv:2509.26114v1 (30 Sep 2025)
created_at: "2026-09-15"
---

# Excerpt: Clip-Low Increases Entropy and Clip-High Decreases Entropy in Reinforcement Learning of Large Language Models

Park, Kim, Kim, Jo, Choi, Cho, Ryu (UCLA; Seoul National University; KRAFTON; Stanford; Santa Clara
University), 2025.

## Claim
The PPO/GRPO clipping rule biases entropy independently of the reward: the lower clip (which applies to
negative-advantage tokens) increases entropy and the upper clip (which applies to positive-advantage tokens)
decreases it; with the symmetric default `ε_low = ε_high = 0.2` the clip-high effect dominates and entropy
falls even when the rewards are random (Abstract, §2.3).

## Theory (§2)
- Setting: advantage independent of prompt and response with `E[A] = 0`, `P(A > 0) = P(A < 0) = ν`,
  `E[A | A > 0] = µ`; tabular softmax policy; full-batch policy gradient (Eq. 5) or natural policy gradient
  (Eq. 6) on the GRPO subproblem.
- Events: `X_k(s)` is "clip-low happens" (`π_k(a|s)/π_old(a|s) < 1 − ε_low`), `Y_k(s)` is "clip-high happens"
  (`π_k(a|s)/π_old(a|s) > 1 + ε_high`).
- Theorem 1 (policy gradient): `H(θ_{k+1}|s) − H(θ_k|s) = µνη d_{π_old} [ p_k (E[Q] − E[Q|X_k]) − q_k (E[Q] −
  E[Q|Y_k]) ] + O(η²)`, with `Q = π_k(a|s)(log π_k(a|s) + H(θ_k|s))`, `p_k = P(X_k)`, `q_k = P(Y_k)`.
  Lowering `ε_low` raises `p_k` and amplifies the first (entropy-increasing) term; raising `ε_high` lowers
  `q_k` and shrinks the second (entropy-decreasing) term.
- The sign claim holds when `E[Q] − E[Q|X_k] ≥ 0` and `E[Q] − E[Q|Y_k] ≥ 0` (Ineq. 7). The authors state these
  are "not guaranteed to hold universally" and that counterexamples exist, but that the estimated values are
  positive throughout their runs (Figs. 1-2, Qwen2.5-1.5B-Instruct and Llama3.2-1B-Instruct).
- Theorem 2 gives the same separation for the natural policy gradient (§2.2, proof App. B).

## Experiments
- Random-reward runs (§2.3): GRPO without standard-deviation normalization, Qwen2.5-3B-Instruct and
  Llama3-8B-Instruct; GRPO batch 512, optimizer batch 256, 8 samples per prompt, temperature 1.0, AdamW at
  constant 5e-7, no KL loss and no entropy loss. Decreasing `ε_low` raises entropy, decreasing `ε_high` lowers
  it (Fig. 3); with `ε_low = ε_high = 0.2` entropy falls (§2.3). The §2.3 setup names Qwen2.5-3B-Instruct and
  Llama3-8B-Instruct as the base models, while the figures of §2 are captioned Qwen2.5-1.5B-Instruct
  (Figs. 1-3) and Llama3.2-1B-Instruct (Figs. 1-2, 8a); the paper does not reconcile the two lists.
- Fig. 4: training with random rewards lowers entropy for Qwen, Llama and Olmo base models; for Olmo2 the
  authors used `ε_high = ε_low = 0.1` "due to slow convergence" (App. C.2).
- True-reward RLVR (§3, GSM8K with Qwen2.5-3B-Instruct; DAPO-Math-17k with Qwen2.5-7B-Instruct): disabling
  clip-high (`ε_high = ∞`) raises entropy, disabling clip-low (`ε_low = 1.0`) lowers it (Fig. 5 left). With
  true rewards the overall pull is downward: the configuration `ε_high = ∞, ε_low = 0.2` raised entropy under
  random rewards but lowered it under true rewards (§3.2).
- Entropy control: `ε_high = ∞` with `ε_low = 0.15` "achieves a balance, preventing both entropy collapse and
  entropy explosion" for Qwen2.5-3B-Instruct on GSM8K (§3.2, Fig. 5 right). Smaller `ε_low` can reach entropy
  explosion.
- Exploration: with the symmetric default `ε_low = ε_high = 0.2`, pass@8 declines over training while mean@8
  rises; with entropy controlled through clipping, pass@8 is preserved at comparable mean@8 (Figs. 6-7,
  Qwen2.5-3B-Instruct and Llama3-8B-Instruct on GSM8K; Qwen2.5-7B-Instruct on AMC and MATH-500 at
  mean@32 / pass@32). The figures print no table values.
- App. C.1 Table 1 (GSM8K / DAPO-Math-17k): AdamW; LR 5e-7 / 1e-6; GRPO batch 512; optimizer batch 256; 16
  policy updates per rollout; group size 8; max response length 4,096; train temperature 1.0; top-p 1.0;
  "For all experiments, KL divergence loss or entropy regularization loss were not deployed." The body text of
  §2.3 states validation temperature 0.6 while Table 1 prints validation temperature 1.0 and top-p 0.95.

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2509.26114v1 (scratchpad `sources/clip-low-clip-high-entropy.txt`).
- Not reported: numeric pass@k tables; results at model sizes above 8B; any interaction with a KL penalty.
