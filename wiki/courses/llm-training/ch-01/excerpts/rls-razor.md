---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rls-razor.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2509.04259
created_at: "2026-09-15"
---

# Excerpt: RL's Razor: Why Online Reinforcement Learning Forgets Less

**Authors:** Idan Shenfeld, Jyothish Pari (equal contribution), Pulkit Agrawal (Improbable AI Lab, MIT)
**Version read:** arXiv:2509.04259v1 (2025-09-04). Source type: paper (preprint).
**Status:** no library card existed for this slug on 2026-09-15. ch-30a and ch-38a treat the SFT-versus-RL comparison in full.

## Claim (Abstract, §1)
"the degree of forgetting is determined by the distributional shift, measured as the KL-divergence between the fine-tuned and base policy evaluated on the new task." The predictor is E_{x∼τ}[KL(π_0 ‖ π)] over inputs x from the new task τ, with π_0 the base policy and π the fine-tuned policy (the paper calls it forward KL).

## LLM settings (§3; App. B.1, Table 2)
- Qwen 2.5 3B-Instruct trained on Open-Reasoner-Zero math questions, SciKnowEval Chemistry L-3, and ToolAlpaca. Prior-capability suite: HellaSwag, TruthfulQA, MMLU, IFEval, WinoGrande, HumanEval.
- Optimizer settings: AdamW; max grad norm 1; weight decay 0; bfloat16; warmup 50 steps. SFT LR sweep {1e-5, 3e-5, 5e-5, 7e-5, 9e-5}, 1-2 epochs, batch {16, 32, 64, 128}. GRPO LR sweep {1e-5 … 5e-5}, KL regularization 0, group size 64, Dr. GRPO loss. The table note states that weight decay and max gradient norm were "manually ablated; since they showed no significant effect on results, they were not included in the final sweep."

## Fit quality (§4)
- ParityMNIST toy setting: "A quadratic fit achieves R2 = 0.96."
- LLM experiments: "a quadratic fit achieving R2 = 0.71 (Figure 11)."

## Alternative predictors (§6, Table 1, MNIST task, R² of a 2nd-degree polynomial)
| Variable | R² |
|---|---|
| KL, forward | 0.96 ± 0.01 |
| KL, reverse | 0.93 ± 0.01 |
| Total variation | 0.80 ± 0.01 |
| Weight change, L1 | 0.34 ± 0.02 |
| Weight change, Fisher-weighted L2 | 0.58 ± 0.02 |
| Weight change, spectral norm | 0.58 ± 0.02 |
| Activation change, L2 | 0.55 ± 0.02 |

"large parameter shifts could occur without forgetting, and conversely, forgetting sometimes occurred despite small parameter movement" (§6).

## bfloat16 and update sparsity (§6)
"the reason for the observed sparse updates was the use of bfloat16 for model training. Since bfloat16 has a limited mantissa, small parameter updates (such as those produced by RL) can fail to cross the representational threshold, effectively causing no update at all. Performing the same training with float32 resulted in models with identical performance but without any sparsity in their weight updates." All algorithms gave full-rank weight updates.

## Verification
- Checked on 2026-09-15 against arXiv:2509.04259v1 (Abstract, §1, §3, §4, §6, Table 1, App. B.1 Table 2).
