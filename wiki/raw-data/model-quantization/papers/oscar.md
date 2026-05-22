<!-- scope: Outlier Suppression — channel-wise gamma migration for BERT activation quant
     deps: percentile-clipping, data-free-quantization
     see-also: outlier-channel-splitting, smoothquant, awq, llm-int8
-->

# Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models
- **Core Insight:** Activation outliers in transformer models concentrate in a small subset of channels and are largely caused by LayerNorm's per-channel γ scaling; migrating the γ amplification into the subsequent linear layer's weights (Gamma Migration) and then clipping the residual outliers (Token-Wise Clipping) recovers most of the loss for 8-bit BERT/BART without QAT — the direct conceptual ancestor of SmoothQuant.
- **Guideline:** Before PTQ, fold each LayerNorm's γ into the downstream linear weight (W ← W·diag(γ); γ ← 1) to remove the outlier-amplification structural source; then apply per-token percentile clipping on the residual to handle the few remaining outlier activations; only then fit per-tensor activation scales.
- **Authors:** Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, Xianglong Liu
- **Year:** 2022 (NeurIPS)
- **URL:** https://arxiv.org/abs/2209.13325
- **Relevant topics:** outlier suppression, LayerNorm gamma migration, per-token clipping, BERT PTQ

## Abstract
Outlier Suppression identifies LayerNorm's per-channel γ scaling as the structural cause of activation outliers in transformer PTQ — γ is concentrated in a few channels (often >10× the median), and downstream activations inherit that scaling. The paper proposes two fixes applied in sequence: (1) Gamma Migration, which folds γ into the next linear layer's weights, eliminating the LayerNorm-injected channel-wise dilation; (2) Token-Wise Clipping, which then sets a per-token clip threshold (rather than per-tensor) to handle the few residual outlier tokens. Together they enable 8-bit PTQ for BERT-Base / BART without QAT, with <0.5 GLUE drop. This is the direct conceptual precursor to SmoothQuant.

## Key Contributions
- Identifies LayerNorm's γ as the structural source of activation outliers in transformers.
- Gamma Migration: closed-form weight re-parameterisation that absorbs γ into the next layer.
- Token-Wise Clipping: per-token percentile clip threshold instead of per-tensor.
- 8-bit BERT PTQ within 0.5 GLUE points of FP, no QAT needed.
- Sets up the activation-to-weight migration philosophy that SmoothQuant generalises.

## Key Figures/Tables to Study
- **Figure 1** — per-channel activation magnitude with vs without γ; visualises the outlier source.
- **Figure 4** — token-wise vs per-tensor outlier distribution: outliers cluster at specific tokens (often [SEP], [CLS]).
- **Table 2** — BERT-Base 6-bit PTQ: Outlier Suppression vs baseline percentile-clipping.

## Technical Details

### LayerNorm gamma migration (the closed-form transformation)
Standard LN + Linear sequence:
`y = LayerNorm(x; γ, β),  z = W·y + b`
LN output is `y_i = γ_i · (x_i − μ)/σ + β_i`. If a single γ_i is large (say 10×), output channel i has 10× larger magnitude than its neighbours — the structural outlier.

Migration: set γ ← 1 and absorb γ into W:
`W' = W · diag(γ),  β' = β · γ`
The forward `W'·y' = W·diag(γ)·y_normalized = W·y` (algebraically identical), but now y' = LN(x; γ=1, β) has uniform per-channel magnitude.

Constraint: works when γ feeds directly into a linear layer. Doesn't work across non-linearities — γ must migrate to the very next op.

### Token-wise clipping
After migration, residual outliers concentrate at specific tokens (typically [SEP], [CLS]). Per-tensor clip wastes scale on these positions. Token-wise clip:
- For each token position t, compute its per-tensor max: m_t = max_c |x_{t,c}|.
- Sort tokens by m_t; identify the top-K outlier-token set.
- Use a separate (higher) clip threshold for outlier tokens.
- Quantize the rest with the standard percentile.

Implementation: per-token scales S_t are negligible memory (one scalar per token) and don't break GEMM if computed before quantization.

### Combined recipe
1. Apply gamma migration to every LayerNorm in the model (one-time weight transformation).
2. Run a calibration sweep with token-wise percentile clipping.
3. Fit per-tensor activation scales on the clipped, migrated activations.
4. PTQ to 8-bit / 6-bit weights and activations.

### Empirical effect (BERT-Base GLUE)
- 8-bit PTQ baseline: FP 84.6 → 82.3 (Δ −2.3)
- + Gamma Migration: 83.7 (Δ −0.9)
- + Gamma Migration + Token Clipping: **84.2** (Δ −0.4)

### Limits
- LayerNorm-followed-by-non-linearity (GELU before next linear) blocks γ migration.
- 4-bit regime still requires QAT — gamma migration alone insufficient.

## Connections
- [[percentile-clipping]] — token-wise clipping is the per-position generalisation.
- [[data-free-quantization]] — same equivalent-transformation philosophy (DFQ uses per-channel rescale across ReLU; Outlier Suppression uses LN γ migration).
- [[outlier-channel-splitting]] — alternative outlier-handling strategy (split high-magnitude channels into two).
- [[smoothquant]] — LLM-era generalisation: migrate activation outliers into weights via arbitrary per-channel scaling, no longer restricted to LN γ.
- [[awq]] — orthogonal: per-channel scaling driven by activation magnitude during weight-only PTQ.
- [[llm-int8]] — extreme-outlier-handling alternative: mixed-precision INT8 + FP16 outlier path.
- [[quantization-error-propagation]] — explains why LN amplifies outlier-channel noise post-quantization.
