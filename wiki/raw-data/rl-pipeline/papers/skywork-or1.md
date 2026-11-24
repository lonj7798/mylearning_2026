<!-- scope: Skywork Open Reasoner 1 technical report (arXiv:2505.22312) — MAGIC recipe: offline/online pass-rate filtering, multi-stage context, no advantage mask for truncation, rollout temperature 1.0, adaptive entropy control, KL removal; entropy-collapse ablations
     deps: [[dapo]], [[grpo]], [[entropy-mechanism-llm-rl]]
     see-also: [[polaris]], [[scalerl]], [[sampling-temperature-schedule]], [[deepmath-103k]], [[rl-design-guide]]
-->

# Skywork Open Reasoner 1 Technical Report (Skywork-OR1)
- **Core Insight:** In long-CoT RL from DeepSeek-R1-Distill models, the speed of entropy collapse predicts test performance: rollout temperature 0.6 drove entropy to near zero within about 100 steps and stalled learning, more SGD steps per rollout (more off-policy updates) accelerated collapse and lowered test scores, and the final recipe uses T = 1.0, an entropy loss switched on only below target entropy 0.2, and no KL loss (§3.2.4, §3.2.5, §3.2.6, §4).
- **Guideline:** When RL training starts from a distilled reasoning model, sample rollouts at T = 1.0 rather than 0.6, filter prompts offline (drop base-model correctness 0 or 1) and online at each stage (drop prompts solved with correctness 1 in the previous stage), and do not mask the advantage of truncated responses, because masking raised accuracy on non-truncated responses but lowered overall accuracy and clip ratio and gave no test-time scaling benefit (Ablation Exp. 3, §3.2.3).
- **Authors:** Jujie He, Jiacai Liu (equal contribution), and others, Skywork AI, Kunlun Inc.
- **Year:** 2025 (arXiv:2505.22312; v2 read, 29 May 2025)
- **URL:** https://arxiv.org/abs/2505.22312
- **Source type:** paper (technical report with ablations)
- **Relevant topics:** rollout temperature, entropy collapse, adaptive entropy control, pass-rate filtering, multi-stage context length, truncated-response handling, KL penalty

## Summary
Skywork-OR1 applies RL to DeepSeek-R1-Distill-Qwen 7B and 32B on math and code. Average accuracy across AIME24, AIME25 and LiveCodeBench rose from 57.8% to 72.8% for 32B and from 43.6% to 57.5% for 7B (Abstract). Skywork-OR1-32B scores 82.2 / 73.3 / 63.0 on AIME24 / AIME25 / LiveCodeBench (2024-08 to 2025-02). The report is organized as a sequence of numbered ablation experiments.

## Key Contributions
- Data filtering at two times: offline before training, online at the start of each stage, plus rejection of zero-advantage groups (§3.1, §3.2.1).
- Multi-stage training with increasing context length (8K → 16K) (§3.2.2).
- Ablation of advantage masking for truncated responses (§3.2.3).
- Temperature and entropy-control ablations; study of what accelerates entropy collapse (§3.2.4–3.2.5, §4).
- KL-loss ablation (§3.2.6).

## Key Figures/Tables to Study
- Ablation Exp. 2 (multi-stage), Exp. 3 (adv mask), Exp. 4 / Fig. 9 (temperature), Exp. 5 (KL), Exp. 8 (entropy coefficients), §4 entropy-collapse experiments.

## Technical Details
**Filtering.** Offline: remove prompts with base-model correctness 1 or 0. Online: at each stage, remove prompts the actor solved with correctness 1 in the previous stage. Rejection sampling keeps only groups with non-zero advantage. Hard NuminaMath-1.5 problems were added (counts not in the portion read).

**Multi-stage (Exp. 2).** Start at 8K context, switch to 16K at step 540; saves about 100 training hours over 1,000 steps and reaches the same AIME24 accuracy (about 60) as training at the long context from the start.

**Truncated responses (Exp. 3).** About 40% of responses were truncated at the start of Stage I. Variants: No-Adv-Mask, Adv-Mask-Before, Adv-Mask-After. Masking raised non-truncated accuracy but caused reward-hacking-like behavior; the final recipe uses no mask.

**Temperature (Exp. 4).** τ = 1.0 chosen over 0.6; at 0.6 entropy starts very low or collapses within about 100 steps. AIME25 values only in Fig. 9.

**Entropy control.** Adaptive entropy loss: target entropy 0.2, coefficient start c₀ = 0, step Δ = 0.005; active only when entropy is below target. Fixed coefficients from 1e-4 to 1e-2 were sensitive to coefficient and data (Exp. 8).

**KL (Exp. 5).** With β = 1e-3, KL to the reference fell rapidly toward zero in Stage 2 and AIME24 gains stalled; β = 0 for all released models.

**Baseline (§4).** R1-Distill-Qwen-7B, 2,700 steps, 32 H800 GPUs: 69.2% avg@8 AIME24, 53.3% avg@8 AIME25, 50.5% pass@1 LiveCodeBench.

## Verification
- Read arXiv HTML v2, first 100K of 153K characters, on 2026-11-24. Not read: group-size results (§5.2), full results table (Table 13), data-source counts (§6). This card does not state a group-size finding.
- Figure-only values (Fig. 9) are not transcribed.
