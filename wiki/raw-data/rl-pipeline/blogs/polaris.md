<!-- scope: POLARIS (HKU NLP / ByteDance Seed / Fudan blog, 2025) — RL recipe for already-strong reasoning models: mirrored-J difficulty distribution, removal of solved prompts, diversity-based temperature choice and stage-wise increase, rollout rescue, length extrapolation with YaRN
     deps: [[dapo]], [[grpo]]
     see-also: [[skywork-or1]], [[sampling-temperature-schedule]], [[entropy-mechanism-llm-rl]], [[scalerl]], [[deepmath-103k]], [[rl-design-guide]]
-->

# POLARIS: A Post-Training Recipe for Scaling Reinforcement Learning on Advanced Reasoning Models
- **Core Insight:** For a strong starting model, the rollout temperature and the prompt-difficulty distribution must be set per model: Qwen3-4B kept stable AIME24 accuracy from T = 0.6 to 1.4 and collapsed above 1.55, so the authors started RL at T = 1.4 (and 0.7 for DeepSeek-R1-Distill-Qwen-7B) and raised it each stage, filtered prompts to a mirrored-J pass-rate shape, and removed prompts above 0.9 accuracy between stages; POLARIS-4B-Preview reached 81.2 / 79.4 on AIME24/25 avg@32 against 73.8 / 65.6 for Qwen3-4B ("Diversity-Based Rollout Sampling"; "Final Results" table).
- **Guideline:** When starting RL from a model that already reasons well, choose the initial rollout temperature by sweeping it and taking the highest value before accuracy drops (measure distinct 4-gram ratio for diversity), not the common 0.6 or 1.0 defaults which the authors report degraded performance, and raise it between stages (by about 0.05 if entropy fell slightly, more if it fell sharply); remove prompts the model solves 8/8 before training and those above 0.9 accuracy after each stage, because keeping only pass rates ≤ 4/8 hurt training as much as no filtering did ("Data Difficulty Distribution").
- **Authors:** Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun Zhang, Shansan Gong, Ming Zhong, Jingjing Xu, Xipeng Qiu, Mingxuan Wang, Lingpeng Kong (The University of Hong Kong, ByteDance Seed, Fudan University)
- **Year:** 2025 (citation year on the page; no publication date shown)
- **URL:** https://hkunlp.github.io/blog/2025/Polaris/
- **Source type:** blog (recipe with ablations)
- **Relevant topics:** rollout temperature, entropy, difficulty distribution, dynamic prompt removal, zero-advantage groups, multi-stage length, length extrapolation

## Summary
POLARIS trains DeepSeek-R1-Distill-Qwen-7B and Qwen3-4B (and a 1.7B model) with a DAPO-style objective (Clip-High, no KL loss, no entropy loss, binary outcome reward checked by LaTeX/SymPy). Its additions concern data and rollout: difficulty shaping, per-model temperature, rescue of zero-advantage groups, staged context length, and YaRN at inference.

## Key Contributions
- Mirrored-J difficulty distribution from offline pass rates (8 rollouts, T = 0.6) and its ablation.
- Dynamic removal of samples with accuracy > 0.9 at the end of each phase.
- Temperature choice by diversity zones and stage-wise temperature increase.
- Rollout Rescue and Intra-Batch Informative Substitution for 0/8 groups.
- Training-free length extrapolation with YaRN factor 1.5.

## Key Figures/Tables to Study
- Temperature-zone plot (Robust Generation / Controlled Exploration / Performance Collapse), difficulty-distribution ablation, stage temperature table, accuracy-by-length plot, final results table.

## Technical Details
**Difficulty.** Remove 8/8-solved prompts. For the 7B model, DeepScaleR-40K + AReaL-boba-106k became a 53K set (26K + 27K); a further pass gave 30K for Qwen3-4B. Ablation on DeepScaleR-40K with the 7B model: full 40K vs 26K (8/8 removed) vs 19K (only pass ≤ 4/8); the 19K and unfiltered sets both hurt training.

**Temperature.** Qwen3-4B zones: stable 0.6–1.4, controlled exploration 1.4–1.55, collapse from 1.55. Stage temperatures: POLARIS-7B 0.7 → 1.0 → 1.1; POLARIS-4B 1.4 → 1.45 → 1.5.

**Zero-advantage groups.** Rollout Rescue: with n = 8, keep a per-prompt store of a correct response from an earlier epoch; when a group is 0/8, replace one failed rollout with it. Intra-Batch Informative Substitution: duplicate mixed-outcome groups to replace zero-advantage groups; the authors state this matches DAPO's benefit without extra rollouts.

**Length.** Qwen3-4B: training at 24K response length caused a sharp irreversible drop; starting at 40K improved steadily, so the authors recommend starting at the maximum decoding length when affordable. For the 7B model, short initial lengths worked. At inference, responses over 32K had 26% accuracy; YaRN (factor 1.5, attn_factor 1.0) raised this to over 50%; test with at least 64K.

**Results (avg@32 on AIME).** POLARIS-4B 81.2 / 79.4 (Qwen3-4B 73.8 / 65.6); POLARIS-7B 72.6 / 52.6 (R1-Distill-Qwen-7B 55.0 / 39.7; Skywork-OR1-7B-Math 69.8 / 52.3).

## Verification
- Read the blog page on 2026-11-24. Section names are the blog's headings; there are no numbered tables.
- Ablations are single runs; no seeds or variance are reported. The claims against Claude-4-Opus, Grok-3-Beta and o3-mini-high are not backed by the table read and are omitted here.
