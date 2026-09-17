---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: primary source, arXiv:2310.04415 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2310.04415
created_at: "2026-09-15"
---

# Excerpt: Why Do We Need Weight Decay in Modern Deep Learning?

**Authors:** Francesco D'Angelo, Maksym Andriushchenko (equal contribution), Aditya Varre, Nicolas Flammarion (EPFL)
**Version read:** arXiv:2310.04415v2 (2024-11-04); v1 2023-10. NeurIPS 2024. Source type: paper.

## Claim (Abstract)
"for large language models trained with nearly one-epoch training, we describe how weight decay balances the bias-variance tradeoff in stochastic optimization leading to lower training loss and improved training stability." "weight decay is never useful as an explicit regularizer but instead changes the training dynamics in a desirable way."

## LLM setting (§3, App. B)
GPT-2-124M (NanoGPT) on OpenWebText, 50,000 iterations, batch 256, context 256 (1,024 for Fig. 8), AdamW with LR 6e-4, β1 = 0.9, β2 = 0.95, 400-iteration warmup, 10× cosine decay, gradient clipping at ℓ2 threshold 1.0, initialization std 0.02, bfloat16-capable A100 GPUs.

## Results (§3)
- Fig. 6: final training loss is lower for λ = 0.1 and 0.3 than for λ = 0. The paper cites Hoffmann et al. (2022, Fig. A7) for "≈ 0.02 lower" training loss with weight decay in AdamW.
- Fig. 19 (appendix, cited in §3): the generalization gap is close to zero even without weight decay.
- Fig. 21: an ℓ2 penalty added to the loss gives the same improvement, so decoupling is "not necessary to achieve this effect". Fig. 22: SGD with momentum shows a similar improvement.
- Effective LR: for sign SGD with weight decay the direction w/‖w‖₂ evolves with effective LR η_t/‖w_t‖₂. Fig. 7: a run without weight decay whose LR schedule matches the effective LR of the λ = 0.1 or 0.3 run reproduces the whole training loss curve in float32; in bfloat16 these runs diverge mid-training.
- Fig. 8 (context 1,024): at LR 0.001 without weight decay, all three bfloat16 seeds diverge late in training and do not recover; float32 is stable; weight decay prevents the divergence. Lowering LR to 0.0006 also prevents it but trains slower (Fig. 24).

## Over-training regime (§2), for contrast
For ResNets trained for many epochs on vision data, weight decay with large LR keeps SGD noise non-vanishing and improves test error through implicit regularization.

## Limits stated by the authors (§4)
"given our limited computational resources, we do not conduct truly large-scale experiments."

## Verification
- Checked on 2026-09-15 against arXiv:2310.04415v2 (Abstract, §1, §3, §4, App. B).
