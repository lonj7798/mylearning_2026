---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: arXiv:2507.18071v2 (Group Sequence Policy Optimization)
source_url: https://arxiv.org/abs/2507.18071
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source — the library has no GSPO card)"
---

# Excerpt: GSPO — sequence-level importance ratios

Used by [[read]] §8, the Recipe, and the Common-mistakes table. Authors: Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, et al. (Qwen Team, Alibaba). Dated 2025-07-29; text read from arXiv v2 on 2026-09-15.

## The argument (§3)
Large rollout batches are split into mini-batches for several gradient updates, which makes training off-policy. GRPO applies the importance weight `π_θ(y_{i,t} | x, y_{i,<t}) / π_θold(y_{i,t} | x, y_{i,<t})` at each token position. Importance sampling corrects a distribution mismatch only when averaged over many samples from the behaviour distribution; at one token position there is a single sample, so the weight "fails to perform the intended distribution-correction role" and adds variance that accumulates over long sequences and is amplified by clipping. The authors report model collapse that resumed training could not repair, even after reverting to an earlier checkpoint and retuning clipping ranges or generation length.

## Objective (Eqs. 5–7)
```
J_GSPO(θ) = E[x~D, {y_i}~π_θold] (1/G) Σ_i min( s_i(θ) Â_i, clip(s_i(θ), 1−ε, 1+ε) Â_i )
Â_i = (r(x, y_i) − mean{r(x, y_i)}) / std{r(x, y_i)}
s_i(θ) = ( π_θ(y_i | x) / π_θold(y_i | x) )^(1/|y_i|) = exp( (1/|y_i|) Σ_t log[π_θ(y_{i,t}|x,y_{i,<t}) / π_θold(y_{i,t}|x,y_{i,<t})] )
```
Length normalization in `s_i` is stated to reduce variance and keep the ratio in one numerical range; without it, a few tokens change the sequence ratio a great deal and responses of different lengths would need different clipping ranges. The clipping ranges of GSPO and GRPO "differ in order of magnitude" because the ratios are defined differently.

## Gradient comparison (§4.2, Eqs. 10 and 12)
GSPO weights every token of a response equally by `s_i(θ) Â_i`; GRPO weights token `t` by its own ratio, which can range over `(0, 1+ε]` for `Â_i > 0` and `[1−ε, +∞)` for `Â_i < 0`.

## GSPO-token (§4.3, Eqs. 13–14)
`s_{i,t}(θ) = sg[s_i(θ)] · π_θ(y_{i,t}|·) / sg[π_θ(y_{i,t}|·)]`, numerically equal to `s_i(θ)`; identical to GSPO when all token advantages in a response are equal, and used when per-token advantage adjustment is wanted (multi-turn RL).

## Empirical results (§5.1–5.3)
- Setting: a cold-start model fine-tuned from Qwen3-30B-A3B-Base; each rollout batch split into four mini-batches; GSPO clip range 3e-4 (left) and 4e-4 (right); GRPO baseline tuned to 0.2 and 0.27. Curves for training reward, AIME'24 (avg Pass@1 over 32 samples), LiveCodeBench (avg Pass@1 over 8 samples), and CodeForces Elo (Fig. 1). No final-score table.
- Clipping fractions (Fig. 2): GSPO about 0.15 of tokens, GRPO 0.0013 — two orders of magnitude more tokens excluded, with higher training efficiency.
- MoE stability (§5.3): on the 48-layer Qwen3-30B-A3B-Base, about 10% of activated experts change after one gradient update for the same rollout, which invalidates token-level ratios. GRPO needed Routing Replay (caching and replaying π_θold's expert routes) to converge (Fig. 3); GSPO does not, because the sequence likelihood is not sensitive to individual token likelihoods.
- Infrastructure (§5.4): sequence-level likelihoods are stated to tolerate training-versus-inference precision differences well enough to use the inference engine's likelihoods directly.

## Limits
Single study; no seeds, no ablation of length normalization, and no final benchmark table. The stated contribution to "the latest Qwen3 models" is an assertion, not a measured comparison in this paper.
