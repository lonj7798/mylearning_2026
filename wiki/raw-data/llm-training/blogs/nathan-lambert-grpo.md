<!-- scope: Nathan Lambert on GRPO tweaks and length-normalization bias
     deps: [[README]]
     see-also: [[grpo]], [[dr-grpo]], [[deepseekmath]], [[deepseek-r1]]
-->

# Interconnects — GRPO Tweaks, Base-Model RL, and Data Curation
- **Core Insight:** DeepSeek's original GRPO has a subtle length-normalization bug that makes the model prefer shorter correct outputs and underpenalize repetition; a family of 2025 patches ("Dr. GRPO", Kimi k1.5 variant, REINFORCE++) fix this in different ways.
- **Guideline:** When implementing GRPO from scratch, default to token-level aggregation (no per-sequence length normalization) unless you explicitly want short-response bias.
- **Author:** Nathan Lambert
- **Year:** 2025 (March 2025 — "Recent reasoning research")
- **URL:** https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo
- **Relevant topics:** GRPO length normalization, Dr. GRPO, base-model RL, reasoning data curation, reward hacking in reasoning RL

## Summary
This Interconnects post surveys 2025 papers on GRPO improvements, base-model RL, and reasoning data curation. The most-cited section argues that DeepSeek's original GRPO formulation normalizes per-output by 1/|o_i| (token count), which causes two biases: (1) correct shorter responses get systematically rewarded more than correct longer responses of equivalent quality, and (2) repetitive patterns within correct responses are not penalized as sharply as they would be under PPO. Multiple follow-up papers propose corrections.

## Key Contributions
- Concise explanation of GRPO's length-normalization issue with worked example.
- Review of fixes: Dr. GRPO, Kimi k1.5 modifications, REINFORCE++ (2025) approach, simple "sum not mean" token aggregation.
- Analysis of how these fixes interact with response-length dynamics during reasoning RL (models typically want to grow CoT length during training).
- Discussion of base-model RL trade-offs: no SFT cold-start means emergent behavior but less readable outputs.

## Key Figures/Tables to Study
- **Original GRPO loss equation** with length normalization term highlighted.
- **Dr. GRPO bias-correction equation** side-by-side.
- **Response-length-over-training plots** showing whether each variant allows CoT growth.

## Technical Details

### The original GRPO loss (DeepSeekMath)
```
L = (1/G) Σ_i (1/|o_i|) Σ_t min(ratio_{i,t} * A_i, clipped) - beta * KL
```
The `(1/|o_i|)` term averages token contributions within each rollout.

### The issue
For a fixed (correct) answer:
- A 50-token correct response and a 500-token correct response have the same A_i (both at top of their group).
- But the 500-token response's ratio-advantage product is divided by a 10x larger denominator.
- Net: the shorter response accumulates more gradient toward being reinforced.

For a partially-repetitive response: repetition lengthens the response, shrinks the per-token gradient, so the repetition pattern is not squeezed out.

### The fixes (per Lambert's tracking)
1. **Dr. GRPO** — drops length normalization entirely in the policy-gradient term while keeping KL normalization per-token.
2. **Kimi k1.5 / K2 variant** — rescales the loss to a per-prompt (rather than per-token) level and adds length-independent advantage.
3. **Sum-not-mean** — a trivial fix: aggregate over tokens with sum instead of mean. Used in some community replications.
4. **REINFORCE++** — adds variance reduction that is less length-sensitive.

### Base-model RL tradeoff
- SFT cold-start (R1 approach): loses some emergent behavior, gains readability.
- No cold-start (R1-Zero approach): mixed-language output, unreadable CoT, but higher observed creativity of reasoning traces.
- Lambert argues base-model RL is "cleaner science" for studying what RL unlocks, but likely not the best production recipe.

### Data curation reminders
The post also tracks data-curation trends: which prompts are worth RL-training on (high reward variance per Qwen 2.5's observation), how much SFT cold-start data is enough (R1's 800K is upper bound; many community replications work with <50K), and why careful RL-prompt filtering beats adding more prompts.

## Connections
- [[grpo]], [[deepseekmath]] — the original algorithm and paper.
- [[dr-grpo]] — the bias-correction Lambert advocates.
- [[deepseek-r1]] — R1 and R1-Zero are the primary case studies.
- [[kimi-k2]] — alternative GRPO-family formulation.
- [[nathan-lambert-rl-overview]] — parent overview post.
- [[nathan-lambert-interconnects]] — lab-index page.
