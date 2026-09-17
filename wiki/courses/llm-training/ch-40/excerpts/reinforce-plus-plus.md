---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: arXiv:2501.03262 v1 and v9 (REINFORCE++); library card [[reinforce-plus-plus]] (verified 2026-09-14)
source_url: https://arxiv.org/abs/2501.03262
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; aligned with the verified card)"
---

# Excerpt: REINFORCE++ — normalize over the batch, not the group

Used by [[read]] §3, §4, §6, and the Recipe. Authors (v9): Jian Hu, Jason Klein Liu, Haotian Xu, Wei Shen; v1 (4 Jan 2025) was by Jian Hu alone. Implementation in OpenRLHF.

## Two variants (§3.1–3.2)
```
REINFORCE++ (k ≥ 1)
  A_{q,o_t} = r(o_1:T, q) − β Σ_{i=t..T} KL(i),  KL(t) = log[π_θold(o_t|q,o_<t) / π_ref(o_t|q,o_<t)]   (Eq. 4)
  A^norm = (A − mean_batch(A)) / (std_batch(A) + ε)                                                     (Eq. 5)

REINFORCE++ w/ Baseline (k > 1)
  A′ = R − mean_group(R)                                                                                (Eq. 6)
  A^norm = (A′ − mean_batch(A′)) / (std_batch(A′) + ε)                                                  (Eq. 7)
  L = L_PPO(A^norm) − λ · E[½ (log π_θ/π_ref)²]                                                         (Eq. 8)
```
The surrogate is PPO-clip (Eq. 1). w/ Baseline equals PPO with the critic removed, GAE `λ = γ = 1`, and two-stage global normalization (§3.3). The k2 loss coefficient `λ` is not reported.

## Arguments against group-level normalization (§2.2, App. A)
1. Theorem 1: `(r_i − mean)/std` is biased for any group size `N ≥ 2` because the denominator depends on `r_i`.
2. With `k = 4` or `8`, near-equal rewards drive the local std toward zero and the advantage "explodes". (Exactly equal rewards give zero, not a large value — see [[read]] §5.)
3. Rewarding a response for beating other samples of the same prompt is linked by the authors to overfitting on easy prompts.
App. B.1 argues that k2 is the correct separate-loss estimator for reverse KL and that k3 (used in GRPO) estimates forward KL.

## Reported results (v9)
- Chat-Arena-Hard, Llama-3-8B-SFT with a Bradley–Terry RM (~700K pairs, 20,000 prompts): REINFORCE++ k=1 46.7 (mean length 832); GRPO k=4 46.8 (860); RLOO 44.6; ReMax 45.1 (Table 1).
- 30 AIME-24 training questions (model not stated): GRPO train pass@1 95.0 with AIME-25 pass@1 0.0 and pass@16 0.4; REINFORCE++ 71.0 / 2.5 / 40.0 (Table 2).
- Knights and Knaves: average 62.1 vs GRPO 55.7, with the gap at 4+ people (§4.2, Fig. 4).
- Qwen2.5-Math-Base on MATH splits: AIME-24 pass@8 21.04 vs 18.96; MATH-500 pass@1 72.00 vs 73.00 (Table 3).
- Tool use, average@32: w/ Baseline 24.10, GRPO 22.58, PPO 21.85 (Table 4).
Each is a single run without reported seeds.

## v1 settings (§4.2 Table 1)
KL coefficient β 0.01 (general) and 0.001 (mathematics); clip ε 0.2; 4 samples per prompt; rollout batch 256, training batch 128 (units not stated); actor LR 5e-7; γ 1.0; maximum 25,000 samples. Training time on 70k samples, H100: PPO 60 h, REINFORCE++ 42 h (§5.2 Table 2).

## Recommendations as stated (§5.1)
Plain REINFORCE++ "performs best with symmetric rewards, such as −1/1" and is recommended when only one response per prompt can be scored; the w/ Baseline variant is recommended with `k > 1` and 0/1 rewards. The authors cite third-party reports (ScaleRL, LitePPO, DLER) that batch-level normalization is more stable (§5.2, not verified here).
