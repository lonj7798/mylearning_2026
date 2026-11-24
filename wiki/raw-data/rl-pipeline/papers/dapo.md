<!-- scope: DAPO (arXiv:2503.14476, ByteDance Seed / Tsinghua AIR / HKU) — Clip-Higher, Dynamic Sampling, token-level loss, overlong filtering + soft overlong punishment, integer-answer DAPO-Math-17K; cumulative AIME24 ablation on Qwen2.5-32B
     deps: [[grpo]], [[ppo]]
     see-also: [[dr-grpo]], [[scalerl]], [[polaris]], [[skywork-or1]], [[slime]], [[verl-grpo]], [[rl-design-guide]]
-->

# DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- **Core Insight:** Four changes to GRPO, added one at a time on Qwen2.5-32B, raise AIME24 avg@32 from 30 to 50: overlong filtering (36), Clip-Higher ε_high = 0.28 (38), soft overlong punishment (41), token-level loss (42), and dynamic sampling that discards groups whose responses are all correct or all wrong (50) (§4.2, Table 1); the prompt set was rewritten so every answer is an integer, because varied answer formats made rule-based checking error-prone (§3.5).
- **Guideline:** When a group-relative RL batch contains prompts whose G responses all receive the same reward, oversample and drop those groups until the batch is full of mixed-outcome groups, and penalize length linearly inside a buffer before the hard limit instead of giving truncated responses the full failure reward, because these two steps contributed +8 and +3 AIME24 points in the cumulative ablation (Table 1); when writing reference answers, transform them to a single canonical type (integers in DAPO-Math-17K) so a rule verifier can match them (§3.5).
- **Authors:** Qiying Yu (project lead) et al., ByteDance Seed, Institute for AI Industry Research (AIR) Tsinghua University, The University of Hong Kong
- **Year:** 2025 (arXiv v1 17 Mar 2025; v2 20 May 2025 read)
- **URL:** https://arxiv.org/abs/2503.14476
- **Source type:** paper (algorithm + open system + dataset)
- **Relevant topics:** dynamic sampling, zero-variance groups, clip range, loss aggregation, overlong reward shaping, answer-format normalization

## Summary
DAPO starts from naive GRPO on Qwen2.5-32B base in verl and adds four techniques. It drops the KL term, arguing that long-CoT training should move far from the initial model (§2.3; no ablation). The final system reaches 50 on AIME 2024, above DeepSeek-R1-Zero-Qwen-32B's 47, with 50% of that model's training steps (Fig. 1, §4.2).

## Key Contributions
- Clip-Higher: decoupled ε_low = 0.2, ε_high = 0.28 (§3.1).
- Dynamic Sampling: filter accuracy-0 and accuracy-1 groups, sample until the batch is full (§3.2).
- Token-level policy-gradient loss: average over all tokens in the batch (§3.3).
- Overlong reward shaping: overlong filtering (mask truncated samples) and soft overlong punishment (§3.4, Eq. 13).
- DAPO-Math-17K with integer answers (§3.5).

## Key Figures/Tables to Study
- Table 1 (cumulative ablation), Eq. 13 (soft overlong punishment), Fig. 2 (entropy under Clip-Higher).

## Technical Details
**Soft overlong punishment (Eq. 13).** R_length(y) = 0 if |y| ≤ L_max − L_cache; ((L_max − L_cache) − |y|) / L_cache if L_max − L_cache < |y| ≤ L_max; −1 if |y| > L_max. Added to the correctness reward.

**Setup (§4.1).** AdamW, constant LR 1×10⁻⁶ with 20-step warm-up; prompt batch 512; 16 responses per prompt; mini-batch 512 (16 gradient updates per rollout step); max length 16,384 expected + 4,096 soft-punish cache = 20,480. AIME evaluation avg@32 at T = 1.0, top-p 0.7.

**Ablation (Table 1, AIME24 avg@32).**

| Configuration | AIME24 |
|---|---|
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (DAPO) | 50 |
| DeepSeek-R1-Zero-Qwen-32B | 47 |

**Answer normalization (§3.5).** Example: an answer of the form (a + √b)/c is reworded so the expected answer is a + b + c.

## Verification
- Read arXiv HTML v2 on 2026-11-24. Course excerpts of this paper also exist under `wiki/courses/llm-training/ch-16|40|43|44a|45a|46/excerpts/dapo.md`; this is the first library card.
- The ablation is cumulative, single-seed; each delta depends on the order in which techniques were added.
- [[scalerl]] Fig. 5a found GSPO and CISPO losses reach a higher asymptote than the DAPO loss.
