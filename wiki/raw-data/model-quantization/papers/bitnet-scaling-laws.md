<!-- scope: scaling-law studies for 1-bit / 1.58-bit LLM pretraining
     deps: [[bitnet-b158]], [[chinchilla]]
     see-also: [[bitnet-b158-2b]], [[fp8-vs-bf16-scaling]], [[bitnet-models]]
-->

# Scaling Laws for 1-bit / Sub-2-bit LLM Pretraining
- **Core Insight:** Across the BitNet b1.58 line, the parameter-vs-loss scaling exponent matches the FP16 scaling exponent over the 700M–4B range — effective bits per parameter stay near 1.58, with a roughly constant ~20-30 % effective-parameter penalty that can be paid down by training more tokens or scaling the model wider rather than deeper.
- **Guideline:** Treat 1.58-bit pretraining as roughly equivalent to FP16 pretraining of a model with ~ 70-80 % of the effective parameter count; the right way to recover quality is more tokens, not more bits.
- **Authors:** Microsoft Research BitNet team (Wang, Ma, Wei et al.); related work from Ouyang et al. ("Low-Bit Quantization Favors Undertrained LLMs")
- **Year:** 2024-2025
- **URL:** BitNet b1.58 scaling: https://arxiv.org/abs/2402.17764 + follow-ups; companion analysis https://arxiv.org/abs/2411.17691
- **Relevant topics:** scaling laws, 1-bit pretraining, effective parameter count, bit budget

## Abstract
The BitNet line is now backed by a small but growing literature on how 1-bit / 1.58-bit training scales with parameters and tokens. The original BitNet b1.58 paper (Ma et al. 2024) reported that, beyond ~3B parameters, the loss curves of 1.58-bit and FP16 transformers converged to within model-to-model noise at the same parameter count. The BitNet-b1.58-2B-4T release confirms the trend at 2B / 4T tokens. The Ouyang et al. companion finding ("Low-Bit Quantization Favors Undertrained LLMs") shows the *opposite* asymmetry for post-training quantization: PTQ degradation grows monotonically with pretraining token count, while *native* low-bit training is least painful for *small* models trained on *many* tokens — the opposite regime. Together these two lines yield a practical rule: at any given deployment bit-width, native low-bit training beats PTQ once the token budget exceeds a critical threshold.

## Key Contributions
- **Loss matching at scale**: BitNet b1.58 700M / 1.3B / 3B runs match FP16 loss curves on the same data, with a small gap that narrows as parameters grow.
- **Effective-parameter penalty**: 1.58-bit training is ≈ equivalent to FP16 training of a model with ~70-80 % the parameters; the penalty can be paid down by widening the model rather than dropping bits.
- **Latency / energy scaling**: BitNet's CPU inference scales sublinearly with model size because the W·x matmul has no multiplies; for memory-bound serving the 1-bit model is roughly bandwidth-bound at its absolute weight size.
- **Token-budget asymmetry** (Ouyang et al. 2024): a Chinchilla-optimal model that will be post-training-quantized at W4 actually wants *fewer* pretraining tokens than the dense Chinchilla recipe suggests — extra training pulls the weights into sharper, more-PTQ-fragile configurations.
- **Crossover prediction**: as deployments push to 100 T training tokens, *PTQ* methods will lose vs *native low-bit* methods; the BitNet line is positioned for that crossover.

## Key Figures/Tables to Study
- Loss-vs-FLOPs curves at 700M / 1.3B / 3B for BitNet b1.58 vs FP16 baseline — the foundational scaling claim.
- Eval-vs-token-budget plot from Ouyang et al. showing PTQ-W4 degradation growing with training tokens.
- BitNet-b1.58-2B-4T eval table cross-referenced to FP16 1B / 2B models at similar token budgets.

## Technical Details

### Bit-budget arithmetic
- Ternary weights = log₂ 3 ≈ 1.585 bits per weight.
- Equivalent storage: at the same memory footprint, 1.58-bit can hold ~ 10× the parameters of FP16 — but the effective parameter capacity is only ~ 7-8× because of the ~20-30 % effective-parameter penalty.
- Inference: 1-bit W × INT8 A is integer add/subtract only on the W·x reduction, so the energy is dominated by the activation load + the INT8 accumulate.

### Native vs PTQ asymmetry (Ouyang et al. 2024)
- For PTQ: degradation grows with training token count, because more pretraining produces sharper weights with stronger outliers.
- For native low-bit training (BitNet): the quantization constraint is part of the optimization target, so weights stay quantization-friendly throughout.
- Combined: there is a token threshold D* beyond which native low-bit training beats PTQ of a dense model at the same deployment bit-width. The threshold drops as deployment precision drops (W4 has a higher D* than W2 or W1.58).

### Scaling exponent
- BitNet b1.58 reports α (the exponent in N^-α scaling) within ~ 0.02 of the FP16 value over its empirical range.
- This is the key claim: the loss-vs-parameters slope is the same as FP16, only the intercept is shifted (the effective-parameter penalty).

### Open questions (flagged by these papers)
- Whether the loss-matching trend extends to > 10B parameter BitNet models — no public training run there yet.
- Whether the FP4-class native training (NVFP4, MXFP4) sits between FP16 and BitNet on the same scaling curve.
- How the scaling law for the LR / batch / data mix interacts with quantization granularity.

## Connections
- [[bitnet-b158]] — the parent 2024 paper.
- [[bitnet-b158-2b]] — the 2B / 4T-token release that validates the scaling claim at production scale.
- [[fp8-vs-bf16-scaling]] — Kumar et al.'s precision-aware scaling law from the FP8 side; together with the BitNet analysis, they cover both ends of the precision axis.
- [[chinchilla]] — the FP16 baseline scaling law that BitNet runs are compared to.
- [[bitnet-models]] — index of official BitNet releases.
- [[era-of-1bit-llms]] — Microsoft's framing essay tying the scaling claims to the hardware design implications.
