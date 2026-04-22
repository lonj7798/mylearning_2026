<!-- scope: Qwen 2.5 technical report — SFT + DPO + GRPO post-training
     deps: [[README]]
     see-also: [[qwen-3]], [[grpo]], [[deepseekmath]]
-->

# Qwen2.5 Technical Report
- **Core Insight:** At ~1M SFT examples and a two-stage context curriculum (short then mixed long), SFT + DPO + GRPO stacked in order produces a state-of-the-art open Instruct model.
- **Guideline:** Use GRPO as the final RL stage after DPO; order RL prompts by reward-model score variance to focus learning on informative examples.
- **Authors:** Qwen Team (Alibaba)
- **Year:** 2024 (arXiv Dec 2024)
- **URL:** https://arxiv.org/abs/2412.15115
- **Relevant topics:** Long-context SFT curriculum, DPO with Online Merging Optimizer, variance-prioritized GRPO, 18T pretraining

## Abstract
Qwen2.5 is Alibaba's open LLM family (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B + Qwen2.5-MoE). Pretrained on 18T tokens. Post-training covers three stages — SFT (1M examples), DPO (150K preference pairs), and GRPO — across the full size range. The report discloses the two-stage SFT context curriculum, specific DPO learning rate and optimizer, and the variance-based prompt ordering used for GRPO. The 72B model beats Llama 3.1 70B Instruct on most benchmarks; Qwen2.5-Math uses a specialized math-focused post-training.

## Key Contributions
- 1M SFT examples spanning short + long context in two phases.
- Online Merging Optimizer for DPO stability.
- Variance-ordered GRPO prompts: queries with high response-score variance are trained first.
- Released 0.5B–72B dense + MoE variants with full Instruct checkpoints.
- Long-context extension to 128K (1M with YARN extrapolation).

## Key Figures/Tables to Study
- **SFT curriculum table:** Phase 1 (short, 32K tokens max) vs Phase 2 (mixed, up to 262K).
- **DPO preference data construction pipeline:** SFT-resampled chosen/rejected via quality filter.
- **Benchmark comparison against Llama 3.1, Mistral Large, etc.**
- **Reward-variance prioritization ablation** (if included).

## Technical Details — Post-Training Pipeline

### SFT
- **Total examples:** 1,000,000 across SFT + DPO + GRPO stages.
- **Two-stage context curriculum:**
  - **Phase 1:** short instructions, max 32,768 tokens. Strong performance on typical tasks.
  - **Phase 2:** mix of short (<=32K) and long (up to 262,144) instructions. Maintains short-task quality while teaching long-context instruction following.
- Standard completion-masked loss.

### DPO
- **Preference data:** ~150,000 pairs. The SFT model resamples responses for new queries; quality-check-pass responses = chosen, quality-check-fail = rejected.
- **Optimizer:** Online Merging Optimizer (keeps a running merged checkpoint to stabilize DPO).
- **Learning rate:** 7e-7.
- **Epochs:** 1.
- **Beta:** not explicitly reported in the public tech report text (standard DPO beta ~0.1 assumed).

### GRPO
- **Algorithm:** GRPO (same family as DeepSeekMath).
- **Prompt ordering (novel):** the sequence in which queries are processed during training is determined by the variance of their response scores, as evaluated by the reward model. Queries with higher variance — where the RM can discriminate strongly between good/bad samples — are prioritized.
- **Reward model:** Qwen reward model trained on preference data (architecture matches policy, linear head).
- Exact KL coefficient, group size G, and LR not individually disclosed in the public arxiv text; Qwen2.5-Math technical report (arXiv 2409.12122) discloses math-specific RL details.

### Pretraining
- **Data:** 18T tokens (significant expansion from Qwen 2's 7T).
- **Context:** native 4K during pretraining, extended to 128K.
- **Architecture:** dense Transformer with SwiGLU, RoPE, RMSNorm; tied input/output embedding for small sizes.

### Benchmark highlights (Qwen2.5-72B-Instruct)
- MMLU: 86.1%
- HumanEval: 85.4%
- MATH: 83.1%
- IFEval: 86.1%

## Connections
- [[qwen-3]] — successor with reasoning/non-reasoning unified; see qwen-3 page.
- [[grpo]] + [[deepseekmath]] — GRPO lineage Qwen adopts.
- [[dpo]] — stage 2; Online Merging Optimizer is a Qwen-specific stabilizer.
- [[tulu-3]] — similar three-stage philosophy (SFT -> DPO -> RL), but Qwen uses GRPO vs Tulu's PPO-RLVR.
- [[llama-3]] — main closed-weight competitor Qwen 2.5-72B targets.
