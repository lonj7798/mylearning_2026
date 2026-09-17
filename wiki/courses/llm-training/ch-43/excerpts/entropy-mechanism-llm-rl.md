---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/entropy-mechanism-llm-rl.md
source_url: https://arxiv.org/abs/2505.22617
primary_version: arXiv:2505.22617v1 (the only version)
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten to match the card verified on 2026-09-14)"
---

# Excerpt: Cui et al. 2025 — the entropy mechanism of RL for reasoning LLMs

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, et al. (17 authors).

## Scope of the study (§2.2)
11 base models from four families (Qwen2.5 0.5B/1.5B/3B/7B/32B, Mistral-7B-v0.3, Mistral-Nemo-Base-2407,
Mistral-Small-3.1-24B-Base-2501, LLaMA3.2-3B, LLaMA3.1-8B, DeepSeek-Math-7B-Base), math and code tasks, and
four algorithms (GRPO, REINFORCE++, PRIME; Fig. 6 adds RLOO), all in verl, starting from base models, with
reference-KL coefficient 0 and clip ε = 0.2.

## The entropy-performance relation
- Eq. 5: `H(π_θ, D) = −E[log π_θ(y_t | y_<t)]`, averaged over the tokens of the responses to a prompt batch.
- Eq. 6: `R = −a·exp(H) + b`, where `R` is validation accuracy and `a, b` are fitted per run. `dR/dH =
  −a·exp(H)`; the value at `H = 0` is `−a + b` (§2.5). The §2 takeaway box prints a different expression,
  `R = −a exp(H + b)`, which does not match Eq. 6 or the abstract.
- 73% of the entropy consumption and 76% of the performance gain happen in the first 200 of 2,400 gradient
  steps (§2.3, Fig. 2). Fitting on the first 36 steps predicts the next 200 with RMSE 0.9% (math) and 1.2%
  (code); at the final step the error is 0.5% (math) and 1.9% (code) (§2.4).
- The authors state the predictability "is not arguably universal": other policy models and off-policy data
  showed different entropy patterns (§2.6). No pass@k measurement is reported (checked §2.2, §4.3, Table 2).

## Mechanism (§3)
- Lemma 1 (tabular softmax, first order): `H(π^{k+1}|s) − H(π^k|s) ≈ −Cov_{a~π^k}(log π^k(a|s), z^{k+1}_{s,a}
  − z^k_{s,a})`, with `z` the logits.
- Theorem 1 (vanilla policy gradient): `z^{k+1} − z^k = η·π(a|s)·A(s,a)`, so
  `ΔH ≈ −η·Cov(log π(a|s), π(a|s)·A(s,a))`.
- Theorem 2 (natural policy gradient): `ΔH ≈ −η·Cov(log π(a|s), A(s,a))`.
- Empirical check on Qwen2.5-7B: `−dH` tracks the measured covariance and the covariance stays positive
  during training (§3.3, Fig. 8).
- After Theorem 1 the paper states that an action with "high/low probability and high/low advantage would
  lower the entropy, and vice versa" (§3.2). No result split by advantage sign is reported (checked §3-§4).

## Interventions (§4)
- Token covariance (Eq. 10): `Cov(y_i) = (log π_θ(y_i) − mean_j log π_θ(y_j))·(A(y_i) − mean_j A(y_j))` over
  the `N` rollout tokens of a batch. Both factors are centered, so a below-average-probability token with a
  below-average (for example negative) advantage has a positive product.
- Clip-Cov (Eqs. 11-12): sample `⌊r·N⌋ ` indices uniformly among the tokens whose `Cov(y_i)` lies in
  `[ω_low, ω_high]` and detach their policy gradient. Values used: `r = 2×10⁻⁴`, `ω_low = 1`, `ω_high = 5`
  (§4.3); both bounds are more than 500× the average covariance.
- KL-Cov (Eqs. 13-14): the tokens with `rank(Cov) ≤ k·N` receive the loss term `−β·D_KL(π_θold ‖ π_θ)`;
  Listing 1 implements the penalty as `|log_prob − old_log_prob|`. Values: `k = 2×10⁻³` (7B) or `2×10⁻⁴`
  (32B), `β = 1`.
- Table 1 (Qwen2.5-7B, step 1), mean token covariance: top 0.02% 5.654; top 0.2% 3.112; top 2% 1.385; top 20%
  0.351; top 50% 0.152; all tokens 0.003.
- Table 2, average of 7 math benchmarks: Qwen2.5-7B — GRPO 38.6, Clip-higher 38.8, Clip-Cov 40.4, KL-Cov 40.6;
  Qwen2.5-32B — GRPO 45.8, Clip-higher 47.2, Clip-Cov 50.3, KL-Cov 52.2. On 32B, KL-Cov gains 15.0 points on
  AIME24 and 14.6 on AIME25 over GRPO (§4.3). Runs use DAPO-MATH with 256 prompts × 8 responses at
  temperature 1 and 8 policy updates per rollout step, maximum generation 8,192 tokens.
- §4.1: entropy-loss coefficients 0.0001 and 0.001 had minor influence, 0.01 caused entropy explosion, and
  0.005 stabilized entropy without outperforming the other baselines (Fig. 9). Reference-KL coefficients
  0.001-0.1 stabilized entropy but lowered accuracy (Fig. 10).
- §4.5: clip-higher affects only positive-advantage tokens; the low-probability positive-advantage tokens it
  admits have an average covariance of about −0.03. The authors report no relationship between the controlled
  entropy level and final performance, and call the optimal entropy an open question.

## Not in this source
The "H < 0.1 nats" collapse threshold, a ranking of tokens by `p·A`, a "top 2%" selection rule, and any
comparison with SAC temperature tuning are not in the paper; they were removed from the library card on
2026-09-14 and from this excerpt in the 2026-09 revision.
