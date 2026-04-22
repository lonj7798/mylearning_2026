<!-- scope: Gemma 2 technical report — distillation-heavy open models from Google DeepMind
     deps: [[README]]
     see-also: [[gemma-2]]
-->

# Gemma 2: Improving Open Language Models at a Practical Size
- **Core Insight:** On-policy knowledge distillation from a larger teacher, combined with model merging (WARP), lets a 9B/27B student reach quality usually demanding 70B+ SFT.
- **Guideline:** For sub-30B open models, invest in distillation-SFT with on-policy student completions before spending RLHF cycles.
- **Authors:** Gemma Team, Google DeepMind
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2408.00118
- **Relevant topics:** On-policy distillation, WARP model merging, RLHF with oversized RM, knowledge distillation in post-training

## Abstract
Gemma 2 is Google DeepMind's 2B/9B/27B open model family. Pretraining uses knowledge distillation from a larger teacher model (unspecified, presumed Gemini Ultra). Post-training applies SFT with on-policy distillation, followed by RLHF with a reward model that is an order of magnitude larger than the policy (inverted from the usual smaller-RM setup). Multiple RLHF-trained checkpoints are then merged with WARP (Weight Averaged Rewarded Policies). The 27B model competes with Llama 3 70B on open benchmarks.

## Key Contributions
- On-policy distillation for SFT: student generates completions from SFT prompts; teacher scores; KL divergence minimized between student and teacher distributions on student-sampled text.
- Oversized RM: the reward model is larger than the policy, a reversal of the usual "policy >> RM" configuration.
- WARP (Weight Averaged Rewarded Policies) merging: EMA during RL, SLERP after RL, repeated — produces a single final checkpoint with superior robustness.
- Architectural detail: interleaved local-attention (sliding window) and global-attention layers, logit softcapping.

## Key Figures/Tables to Study
- **On-policy distillation diagram** vs classic teacher-outputs-SFT distillation.
- **WARP merging schematic:** EMA -> SLERP -> repeat.
- **RM-larger-than-policy ablation** if included.
- **Benchmark table vs Llama 3 8B/70B.**

## Technical Details — Post-Training Pipeline

### SFT with on-policy distillation
- The Gemma 2 team generated completions from a teacher model for SFT prompts, then trained the student on this synthetic data.
- **On-policy variant:** to address distribution mismatch, the student generates completions from the SFT prompts at training time; the teacher distribution is evaluated on the student's own outputs, and KL(student || teacher) is minimized across the completion tokens.
- This is essentially distillation against student-sampled rollouts rather than teacher-sampled rollouts.

### RLHF
- **Reward model:** ~an order of magnitude larger than the policy (novel — contrast with the usual RM ≤ policy size).
- **Orientation:** RM trained with emphasis on multi-turn conversational capability.
- **RL algorithm:** not fully disclosed in the public report — report states "similar to Gemma 1.1" but with the new RM.
- Specific LR, KL coefficient, batch size for RLHF not publicly itemized.

### WARP model merging
Three stages:
1. **EMA during RL fine-tuning:** maintain an exponential moving average of policy weights during RL training for intra-run stability.
2. **SLERP after RL:** spherical linear interpolation between checkpoints of multiple RL runs (different seeds/hyperparameters).
3. **Repeat:** apply the SLERP output as the initialization for another RL round.

WARP is the reported mechanism by which Gemma 2 combines multiple RLHF-tuned policies into a final release checkpoint.

### Pretraining context
- **Architecture:** interleaved local-attention (sliding window) and full attention layers; logit softcapping to stabilize large-vocab logits; grouped-query attention.
- **Sizes:** 2B, 9B, 27B dense.
- **Pretraining data:** 2T tokens (2B), 8T (9B), 13T (27B); distilled from a larger teacher during pretraining as well.

### What is NOT disclosed
The public Gemma 2 report does not itemize SFT dataset size, preference data volume, RLHF hyperparameters, or specific data sources — a notable contrast with Llama 3 and Tulu 3. Gemma 3's report (2025) also remains sparse on post-training numbers.

## Connections
- Gemma 3 — successor continuing distillation-heavy approach.
- [[llama-3]] — contrast: Llama 3 does iterative RLHF with a same-size RM and full hyperparameter disclosure.
- [[orca]] + [[orca-2]] — parallel line on teacher-explanation distillation for SFT.
- [[constitutional-ai]] — Gemma 2's safety fine-tuning borrows the self-critique idea.
- [[lima]] — Gemma 2's curated-SFT philosophy echoes the "quality over quantity" finding.
