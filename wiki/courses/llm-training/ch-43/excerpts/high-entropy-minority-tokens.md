---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2506.01939
primary_version: arXiv:2506.01939v2 (13 Nov 2025); NeurIPS 2025
created_at: "2026-09-15"
---

# Excerpt: Beyond the 80/20 Rule — High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

Wang, Yu, Gao, Zheng, Liu, Lu, et al. (Qwen Team, Alibaba; LeapLab, Tsinghua University), 2025.

## Token-entropy statistics of a chain of thought
- Setting: Qwen3-8B in thinking mode, AIME'24 and AIME'25 questions, decoding temperature 1.0, over 10^6
  response tokens collected; per-token entropy computed from the full categorical distribution (§3).
- "the entropy of over half the tokens (approximately 50.64%) is below 10^-2, while only 20% of tokens have
  entropy greater than 0.672" (§3, Entropy Pattern 1). The 0.672-nat value is the 80th percentile of that
  token sample and is used as the threshold `h_threshold` in the decoding experiment of §3 (Eq. 5).
- Entropy Pattern 2: the highest-entropy tokens bridge two reasoning steps ("forking tokens"); the
  lowest-entropy tokens complete an ongoing linguistic or mathematical structure (§3).

## Decoding experiment (§3, Eq. 5, Fig. 3)
Two temperatures are used at decoding time: `T_high` for tokens with `H_t > 0.672`, `T_low` for the rest.
Lowering `T_high` lowers the AIME average more than lowering `T_low`; raising `T_high` raises it more than
raising `T_low`, up to the point where very large temperatures produce unusable output (Fig. 3; the plotted
values include 68.33 and 68.96 at the low end and 71.67 / 72.22 near the top of the red curve, with 62.71,
20.14 and 0.14 at extreme temperatures).

## RLVR changes mainly the high-entropy tokens
- Table 1: the overlap of the top-20% high-entropy token positions between intermediate RLVR checkpoints and
  the base model falls from 100% to 86.67%, while the overlap with the final RLVR model rises from 86.67% to
  100%. The base-model overlap stays above 86% throughout (§4).
- Fig. 4: average entropy change after RLVR, per 5% entropy percentile of the base model, is concentrated in
  the high-entropy percentiles (§4).

## Training on forking tokens only (§5)
- Objective (Eq. 6): DAPO's token-level loss multiplied by the indicator `1[H_t^i ≥ τ_ρ^B]`, where `τ_ρ^B` is
  the entropy threshold that selects the top-ρ fraction of tokens in the (micro-)batch, and the token
  normalizer counts only the selected tokens.
- Settings (§5.2): DAPO with clip-higher `ε_low = 0.2`, `ε_high = 0.28`; overlong reward shaping with maximum
  response length 20,480 and 4,096-token cache; verl training batch 512, mini-batch 32 (16 gradient steps per
  batch), learning rate 1e-6, no warmup or schedule; "the training process excludes both KL divergence loss
  and entropy loss"; data DAPO-Math-17K; ρ = 20%.
- Evaluation: 16 responses per question at temperature 1.0, average accuracy reported as Acc@16 (§5.2).
- Table 2, average over AIME'24, AIME'25, AMC'23, MATH500, Minerva, OlympiadBench, all-tokens DAPO →
  forking-token DAPO: Qwen3-32B base 66.59 → 70.69 (AIME'24 55.83 → 63.54, AIME'25 45.63 → 56.67);
  Qwen3-14B base 61.40 → 64.39; Qwen3-8B base 53.71 → 54.23 (AMC'23 falls 77.81 → 77.19).
- ρ ablation (§5.3, Fig. 7): 10%, 20%, 50% give similar results at 8B; at 14B and 32B, ρ = 10% is slightly
  lower and ρ = 100% is "a notable decline"; every setting other than 20% shows lower overall entropy.
  Training on the bottom 80% (lowest-entropy tokens) gives "a substantial decline in performance".
- Out-of-domain check (§5.4, Fig. 9): trained on DAPO-Math-17K, evaluated on LiveCodeBench (avg@16), top-10%
  and top-20% runs stay above the all-token run over 1,400 gradient steps. The figure prints no table values.

## Statements about entropy bonus versus clip-higher (§6, Discussion 3, Interpretation)
The authors argue that an entropy bonus raises the entropy of the low-entropy majority, which "can degrade
performance by disrupting the low-entropy majority", while clip-higher raises `ε_high` and therefore admits
more high-ratio tokens, which they observe tend to be higher-entropy. No controlled comparison of the two is
reported in this paper.

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2506.01939v2 (scratchpad `sources/high-entropy-minority-tokens.txt`).
- Not reported: pass@k at any k; KL to a reference policy; any non-Qwen3 base model.
