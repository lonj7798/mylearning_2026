---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/lazy-likelihood-displacement-grpo.md on 2026-09-15)
source_url: https://arxiv.org/abs/2505.18830
source_version: arXiv v1 (2025-05-24)
created_at: "2026-09-15"
---

# Excerpt: On the Effect of Negative Gradient in Group Relative Deep Reinforcement Optimization (Deng, Ren, Li, Sutherland, Li, Thrampoulidis; UBC and Vector Institute)

Facts used by [[read]], read in the arXiv v1 PDF on 2026-09-15.

## Definition and measurement
- Lazy Likelihood Displacement (LLD, Definition 4.1): after optimizing the objective, `ln π_fin(y⁺|x) < ln π_init(y⁺|x) + ε` for a small ε ≥ 0, that is, the likelihood of a correct response decreases or increases only marginally.
- Measurement protocol (§3): Qwen2.5-0.5B-Instruct, Qwen2.5-Math-1.5B with MATH, DeepSeek-R1-Distill-Qwen-1.5B with AIME; 8 rollouts per question; prompts whose rollouts are all correct or all incorrect are dropped; parameters are reset for each question, one GRPO update is applied, and `Δ(x)` is the mean change of ln π(y⁺|x) over the correct responses (Eq. 2).
- Findings (Figure 1): under GRPO many questions show small or negative Δ(x); a "Pos Only" variant that sets negative advantages to 0 gives substantially higher Δ(x), "particularly on the left side of the plots, where many samples show several-fold improvements". The authors also state negative gradients "are not always harmful": some samples have smaller Δ(x) under Pos Only, and Pos Only scores 1.3% below GRPO on average (Table 2, Qwen2.5-Math-1.5B: GRPO 41.14, Pos Only 39.84).
- Inspection of the LLD questions (§3, Figure 2): the incorrect responses in those groups are "nearly correct" or give the correct answer in the wrong output format, so "penalizing entire partially correct responses is suboptimal".

## Theory (§4.1)
- Lemma 4.2: with a binary reward, GRPO is preference optimization between the group of correct and the group of incorrect responses, with weights `p⁺ = (1−p)/√(p(1−p))` and `p⁻ = p/√(p(1−p))`, where p is the correctness rate.
- Theorem 4.4 (unconstrained-features model, gradient flow): the likelihood change of a correct response becomes lazier as the Group Weighted Hidden Embedding Score grows, whose negative-gradient term is `p⁻ Σ_k Σ_j Σ_{k'} α⁻_{k,k'} ⟨h_{x,y⁺_{<k}}, h_{x,y⁻_{<k'}}⟩`; Corollary 4.5 isolates the contribution of one negative token k′.
- Identification check (Table 1): ranking questions by GWHES overlaps with the ranking by actual likelihood change far more than random ranking: top-10 overlap 50% vs 17.5% (DeepSeek-1.5B, AIME) and 60% vs 21.3% (Qwen2.5-Math-1.5B, MATH); top-15 overlap 75% vs 26.3% and 75% vs 31.9%.

## NTHR (§5)
- Negative token hidden reward: `s⁻_{j,<k'} = Σ_i Σ_k α⁻_{k,k'} ⟨h_{x,y⁺_{i,<k}}, h_{x,y⁻_{j,<k'}}⟩` (Eq. 7), the influence of one token of an incorrect response on the likelihood of the group's correct responses.
- Selection: tokens with `s⁻_{j,<k'} > τ` are down-weighted, with `τ = β · min_i s̄⁺_i` where s̄⁺_i is the average influence of correct response i on the other correct responses; the retained tokens get advantage `η · Â⁻` with η < 1. Experiments use β = 1 and η = 0, so the selected tokens receive no penalty at all (§5.3).
- Tokens with high NTHR are often "logically or stepwise correct terms" inside the wrong response (§5.1, Figure 3).
- Results (Table 2, greedy decoding, average of AIME24, AMC, MATH500, Minerva, Olympiad; trained on MATH levels 3-5): Qwen2.5-Math-1.5B base 19.10, GRPO 41.14, Pos Only 39.84, NTHR 41.94; Qwen2.5-0.5B-Instruct base 9.46, GRPO 11.72, NTHR 12.66; Qwen2.5-1.5B-Instruct base 23.06, GRPO 26.96, NTHR 28.48; Qwen2.5-3B base 31.36, GRPO 33.88, NTHR 36.30; Qwen2.5-Math-1.5B on DeepScaleR: GRPO 37.80, NTHR 39.60. A random-token baseline (GRPO+Random) gives "only modest gains" on Δ(x) (Figure 4).
- Limitation stated: DeepSeek-1.5B was trained with a 4k context window, which shortened responses over training.
