<!-- scope: MiniMax-01 series — lightning attention MoE + M1 CISPO RL
     deps: []
     see-also: [[deepseek-v3]], [[kimi-k1-5]]
-->

# MiniMax-01 / MiniMax-M1
- **Core Insight:** Clip the importance-sampling weights, not the token updates — CISPO preserves all tokens for gradient computation while bounding off-policy drift, giving 2× speedup over DAPO at 32B scale.
- **Guideline:** When rollouts go long and off-policy (linear-attention rollouts are cheap), switch from token-clipping to IS-weight-clipping.

- **Authors / Lab:** MiniMax AI
- **Year:** 2025 (MiniMax-01 Jan 2025; MiniMax-M1 Jun 2025)
- **URL:** https://arxiv.org/abs/2501.08313 (01) — https://arxiv.org/abs/2506.13585 (M1)
- **Relevant topics:** lightning attention (linear attention), hybrid attention, CISPO, importance-sampling clipping, reasoning RL on hybrid-attention MoE

## Abstract
MiniMax-01 is a 456B-parameter / 45.9B-active hybrid-attention MoE: 7 out of every 8 layers use **lightning attention** (I/O-aware linear attention variant) and 1 layer uses SoftMax. Context up to 1M tokens at training, 4M at inference. MiniMax-M1 (Jun 2025) is the reasoning-RL evolution: full RL on 512 H800 GPUs completing in 3 weeks for ~$534,700, using the new **CISPO** algorithm that clips importance-sampling weights rather than token-probability ratios. Released in 40K and 80K thinking-budget variants.

## Key Contributions
- **Lightning attention hybrid** — 7:1 linear-to-softmax layer ratio in the MoE.
- **CISPO** — clips importance-sampling weights `r̂(θ) = clip(r(θ), 1−ε_low, 1+ε_high)` while keeping all tokens in the gradient. 2× speedup vs DAPO on controlled Qwen2.5-32B.
- **Full RL at $534,700** — 512 H800s × 3 weeks is a new open cost-efficiency point for reasoning RL.
- **40K / 80K thinking-budget variants** released; 40K is an intermediate checkpoint on the 80K trajectory.
- **Rule-based + GenRM hybrid rewards** with continuous online monitoring for length-bias mitigation.

## Post-training pipeline
- **SFT data:** high-quality CoT across math, coding, STEM, writing, QA; ~60% math+coding weight. Size not publicly itemized.
- **Preference / RL algorithm:** **CISPO** (novel GRPO-family variant). Group-relative advantage normalization retained; clipping moved from token-probability ratio to IS weight.
- **Reward model:** generative reward models (GenRM) with continuous monitoring to catch length bias; combined with verifiable rewards in math/code/logic domains.
- **Reward datasets:** math ~50K curated, logic ~53K via SynLogic synthesis, competitive programming 30K, SWE-bench thousands, general 25K RM-verified.
- **KL / entropy handling:** CISPO's design replaces token-clipping's entropy effects; explicit KL β not surfaced.
- **Rollout scale:** 512 H800 GPUs × 3 weeks; supports up to 1M-token training context via lightning attention.
- **Hyperparameters:**
  - AdamW β1 = 0.9, β2 = 0.95, ε = 1e-15 (pretraining; RL reuses).
  - Pretraining LR 8e-5 → 8e-6.
  - Pretraining 7.5T tokens.
  - Group size G not explicitly stated; IS-weight clipping ε_high tuned, ε_low disabled.
  - Context window 1M train / 4M inference (01); 80K thinking budget (M1).
- **Verifiable rewards:** yes — math verification, code execution, SynLogic logic-puzzle verification, SWE-bench task success.
- **Self-improvement / iterative:** 40K → 80K budget extension is an iterative curriculum; SynLogic provides synthetic self-generated logic problems.

## Innovations vs predecessors
Changes from **industry 2024 baselines → MiniMax-M1**:
- **CISPO** as a new RL algorithm family — distinct from PPO (token-clip), GRPO (token-clip), DAPO (dynamic sampling); clips a different quantity.
- Lightning-attention hybrid architecture makes long-context rollouts affordable — the RL cost story is conditional on the architecture.
- 1M-token training context not seen elsewhere at this scale in 2025.
- SynLogic-generated logic-reasoning dataset introduced.

## Key Figures/Tables to Study
- CISPO vs GRPO vs DAPO training-curve comparison (from M1 paper) — the 2×-speedup claim.
- Cost table: $534,700 / 512 H800 / 3 weeks — a rare full public RL cost disclosure.
- Lightning-attention 7:1 layer-pattern diagram — architectural base that makes CISPO-long-context RL feasible.

## Connections
- [[deepseek-v3]] — MoE + long-context peer; V3.2 DSA vs lightning attention are different sparse/linear attacks on the same problem.
- [[kimi-k1-5]] — shared long-CoT RL territory; different algorithms (mirror descent vs CISPO).
- [[grpo]] / DAPO — direct baselines CISPO is measured against.

## Gaps / what the report does NOT disclose
CISPO's ε_high exact value + schedule. KL β (if any). SFT dataset size. Group size G. Number of RL steps. Per-batch rollout count. GenRM architecture / training data. Reward-weight mixing across math/logic/code/SWE/general. Lightning-attention implementation details at RL-rollout time. 40K→80K extension schedule.
