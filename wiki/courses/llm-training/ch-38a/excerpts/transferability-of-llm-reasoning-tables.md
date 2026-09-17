---
chapter: ch-38a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/transferability-of-llm-reasoning.md (card exists but carries no numbers and no Verification section as of 2026-09-15)
source_url: https://arxiv.org/abs/2507.00432
created_at: "2026-09-15"
---

# Excerpt: tables and formulas from "Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning"

**Authors:** Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, Xiang Yue.
**Version read:** arXiv:2507.00432v2 (20 Oct 2025).
**Status:** the library card [[transferability-of-llm-reasoning]] describes the paper but states no numbers and has not been verified to the 2026-09 card standard. Everything below was read at the stated locus in the v2 PDF.

## Transferability Index (§2.1)
For group g ∈ {math, other, non} and benchmark b in group g:

```
ΔR_b = R_b^model − R_b^base
σ_g   = Std{ΔR_b : b ∈ B_g}
δ_b   = ΔR_b / σ_g
s_b   = sign(δ_b) · |δ_b|^(1/2)
w_b   = 100 − R_b^base,      ŵ_b = w_b / Σ_{u ∈ B_g} w_u
DI_g  = Σ_b ŵ_b · s_b
TI_g(%) = 100 · DI_g / DI_math
```

The paper does not state whether Std is the sample or the population standard deviation.

## Table 1 — controlled study on Qwen3-14B-Base
Math reasoning (AIME24 / AIME25 / MATH500 / Olympiad / average):
- Qwen3-14B-Base 13.0 / 9.3 / 60.4 / 27.9 / 27.7
- UniReason-Qwen3-14B-think (SFT) 52.0 / 37.0 / 85.0 / 25.0 / 49.8
- UniReason-Qwen3-14B-no-think (SFT) 16.0 / 13.0 / 77.2 / 22.7 / 32.3
- UniReason-Qwen3-14B (RL) 55.7 / 38.0 / 87.8 / 33.8 / 53.8

Other reasoning (GPQA / LiveCodeBench2 / ACPBench / HeadQA / average / TI_other):
- Base 42.6 / 29.7 / 10.7 / 37.6 / 30.2 / –
- SFT think 55.9 / 21.9 / 68.6 / 34.8 / 45.3 / +52.2
- SFT no-think 48.7 / 23.5 / 69.3 / 35.0 / 45.2 / +165.4
- RL 57.7 / 40.6 / 65.4 / 40.2 / 60.0 / +82.3

Non-reasoning (CoQA / MC-TACO / IFEval / HaluEval / average / TI_non):
- Base 10.0 / 67.7 / 69.2 / 35.7 / 45.7 / –
- SFT think 1.7 / 38.2 / 42.3 / 2.3 / 21.1 / −104.1
- SFT no-think 5.3 / 66.1 / 41.4 / 3.3 / 29.0 / −278.9
- RL 28.2 / 74.0 / 70.0 / 40.7 / 53.2 / +52.2

## Table 4 — component ablation on Qwen3-8B-Base
| Setting | Math avg. | Other reasoning avg. | Non-reasoning avg. | TI_other | TI_non |
|---|---|---|---|---|---|
| Qwen3-8B-Base | 27.6 | 23.6 | 33.6 | – | – |
| Off-policy SFT | 41.9 | 34.4 | 26.6 | 18.3 | −40.5 |
| On-policy SFT | 33.7 | 35.7 | 35.0 | 68.6 | 30.2 |
| Off-policy RL | 45.5 | 35.9 | 31.7 | 36.4 | 4.5 |
| On-policy RL (no KL) | 37.1 | 38.2 | 35.8 | 65.6 | 39.3 |
| On-policy RL | 38.6 | 39.9 | 35.0 | 63.7 | 32.4 |

Settings (Table 3): sampling q ∈ {δ_{y=y*}, π_θ}, weights w ∈ {1, rejection-sampled 1, advantage A_t}, KL coefficient β ∈ {0, >0}. Conclusions stated in §5.2: "on-policy methods consistently transfer better than off-policy methods"; credit assignment and negative examples "not only improve transferability but also increase response length"; "On-policy RL performance remains largely unchanged with or without KL regularization."

## Distribution-shift diagnostics (§3, §4)
KL divergence to the backbone (Fig. 4): UniReason-Qwen3-14B-SFT-no-think 0.372 on MATH-500 and 0.283 on IFEval; UniReason-Qwen3-14B (RL) 0.084 and 0.019 on the same two. Average token-rank shift (§4.2, Fig. 17): RL 0.98 positions, SFT no-think 10.6. Shifted-token counts in the case study (Table 8): SFT shifts 390 tokens for a reasoning prompt and 158 for a non-reasoning prompt, against a small task-relevant set for RL. PCA-based latent shift is lowest for RL in all three task groups (§3.2, Table 2).

## Settings (App. A.3.1–A.3.2)
RL: verl, GRPO on Qwen3-14B-Base, answer correctness as reward, LR 1e−6, train batch 512, clipping thresholds 0.22–0.28, sequences up to 16k tokens, 16 rollouts per prompt, mini-batches of 128, KL and entropy coefficients 0, 140 steps. SFT: LLaMA-Factory, LR 5e−5, batch 512, 1.5 epochs, targets from Qwen3-32B in think mode with rejection sampling. Training data: 47K curated math problems (DeepScaler plus SimpleRL levels 3–5).

## How ch-38a uses it
§2 (Table 1), §5 (Table 4 ablation), §4 (KL and rank-shift diagnostics), §9 (measurement protocol), Recipe.
