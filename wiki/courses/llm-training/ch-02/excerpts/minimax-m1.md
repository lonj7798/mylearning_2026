---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "MiniMax — MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention (arXiv:2506.13585), §3.2"
source_url: https://arxiv.org/abs/2506.13585
created_at: "2026-09-15"
revised: "2026-09-15 (created from arXiv:2506.13585v1 §3.2; generality revision)"
---

# Excerpt: MiniMax-M1 RL numerics (arXiv 2025-06, §3.2)

Created on 2026-09-15 from arXiv:2506.13585v1 (2025-06-16). The library has a verified card for the earlier
report [[minimax-01]], which states that no MiniMax-M1 card exists; this excerpt covers only §3.2.

## Model
- MiniMax-M1 is developed from MiniMax-Text-01, which has 456 billion parameters in total and 45.9 billion
  activated per token; the architecture is a hybrid Mixture-of-Experts model with lightning attention
  (Abstract, §1). Full RL training ran on 512 H800 GPUs for three weeks (Abstract).

## Computational precision mismatch between generation and training (§3.2)
- During RL training, the probabilities of rolled-out tokens differed between training-mode and inference-mode
  code; the authors attribute this to "a precision mismatch between the training and inference kernels" and
  report that it "prevented reward growth in our experiments".
- The issue "did not appear in smaller, dense models with softmax attention".
- A layer-by-layer analysis identified "high-magnitude activations in the LM head at the output layer" as the
  primary source of error.
- Fix: the LM output head was computed in FP32. The text states that the correlation between training and
  inference probabilities improved "from approximately 0.9x to 0.99x"; Figure 3 prints Pearson correlations of
  0.987319 before and 0.997135 after the fix. The correlation "remained stable throughout training, enabling
  successful reward increase." No reward curve or accuracy numbers for the unfixed run are given in §3.2.

## Optimizer settings for RL (§3.2)
- The VeRL default AdamW configuration, betas (0.9, 0.999) and eps 1e-8, "can result in" non-convergence.
- Gradient magnitudes in M1 training span 1e-18 to 1e-5, with most below 1e-14, and gradients of adjacent
  iterations are weakly correlated. The chosen settings are β1 = 0.9, β2 = 0.95, eps = 1e-15. No ablation table
  is given.

## Repetition truncation (§3.2)
- Generation is stopped if 3,000 consecutive tokens each have probability above 0.99, to avoid long repetitive
  responses whose large gradients threatened stability.

## Used in ch-02
- §7 (LM-head precision as a source of sampler–learner mismatch), §8 (ε relative to gradient magnitude),
  Recipe rows.
