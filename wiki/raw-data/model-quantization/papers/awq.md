<!-- scope: AWQ — activation-aware weight-only PTQ; per-channel scale grid-searched from activation magnitude
     deps: [[smoothquant]], [[gptq]]
     see-also: [[omniquant]], [[autoawq]], [[spqr]], [[squeezellm]]
-->

# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
- **Core Insight:** Only ~1% of weight channels are "salient" — they multiply with large-magnitude activations and thus dominate the output error — and protecting those channels by per-channel scaling (grid-searched, gradient-free) is enough to make 4-bit weight-only quantization nearly lossless without ever touching the activation precision.
- **Guideline:** As the W4A16 default, run AWQ with auto-grid search over a single per-channel scale `s ∈ [1, 4]` driven by activation absmean; combine with `group_size=128` weight quant; serve with AWQ/Marlin kernels for ~3× FP16 throughput.
- **Authors:** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han
- **Year:** 2023 (MLSys 2024 Best Paper)
- **URL:** https://arxiv.org/abs/2306.00978
- **Relevant topics:** activation-aware weight quant, salient-channel protection, W4A16 PTQ, TinyChat kernel, grid-search calibration

## Abstract
AWQ proposes a weight-only PTQ that achieves near-FP16 quality at 4-bit on LLMs without any gradient-based optimization. The key empirical fact: in a quantized linear layer, the per-output error is dominated by a small fraction (~1%) of weight channels — those multiplying the activation dimensions with largest magnitude. AWQ protects these salient channels via a per-channel scale `s ∈ R^{C_in}` chosen by grid search to minimise the layer output MSE; the scale is then folded into the previous LayerNorm so the runtime graph is unchanged. Combined with group-wise (G=128) INT4 weight quant, AWQ matches GPTQ while generalizing better across domains and modalities (it doesn't overfit to the calibration text because it doesn't backprop into reconstruction). Shipped in TinyChat and `AutoAWQ` with >3× FP16 throughput on desktop and mobile GPUs.

## Key Contributions
- Empirical proof that protecting **1% salient weight channels** (chosen by activation magnitude, not weight magnitude) recovers the bulk of W4 quantization error.
- A **gradient-free, single-parameter** scaling search per layer — no backprop, no reconstruction MSE optimization → no calibration overfit.
- The equivalent transformation `Y = (W · diag(s)⁻¹) · (diag(s) · X)` reused from [[smoothquant]] but applied weight-only (scale is folded backwards into the preceding op).
- Demonstrates generalization across vision-language models and instruction-tuned LLMs where GPTQ overfits to its calibration corpus.
- **TinyChat** inference engine: fused dequant + GEMM with kernel-aware weight packing; >3× speedup over HF FP16 on RTX 4090 and Jetson Orin.
- Won **MLSys 2024 Best Paper Award**.

## Key Figures/Tables to Study
- **Figure 2 (motivation):** error contribution of top-1% activation-magnitude channels vs random 1% — the empirical justification.
- **Figure 5 (method):** scale-search curve — output MSE as a function of α, single concave bowl.
- **Table 4–6:** OPT/LLaMA/LLaMA-2 W4 results — AWQ matches or beats GPTQ on every model.
- **Table 8:** vision-language (LLaVA) and instruction-tuned generalization — AWQ wins where GPTQ regresses.

## Technical Details

### Salient channel observation
For weight `W ∈ R^{C_out × C_in}` and activation `X ∈ R^{C_in}`, the per-output error of RTN quant is
```
err_o = Σ_j ΔW_{o,j} · X_j
```
Empirically, the per-output error is dominated by the j's with largest `mean(|X_j|)`. Top-1% by activation magnitude carries the majority of MSE.

### Per-channel scaling
Mirror SmoothQuant's equivalent transformation but applied weight-only:
```
Y = W · X = (W · diag(s)⁻¹) · (diag(s) · X)
```
- `diag(s) · X` is *not* quantized (weight-only setting, A16).
- `W · diag(s)⁻¹` is quantized to INT4: dividing the salient weight columns by `s_j > 1` reduces their dynamic range so RTN preserves them.
- The `diag(s) · X` factor is absorbed into the preceding LayerNorm γ/β so no extra runtime op.

### Grid search for `s` (no backprop)
Parameterize a single scalar `α ∈ [0, 1]` per layer and set
```
s_j = (mean(|X_j|))^α    (per input channel)
```
Search α on a small grid (paper uses 20 points in [0, 1]) and pick the value that minimises layer output MSE on calibration data:
```
α* = argmin_α  || W · X  −  Q(W · diag(s)⁻¹) · diag(s) · X ||²
```
- `Q(·)` is INT4 group-wise (G=128) RTN.
- One forward pass per α; no gradients.

### Folding `diag(s)` backwards
For the common pattern `LayerNorm → Linear`, multiply LayerNorm's affine `γ_j ← s_j γ_j`, `β_j ← s_j β_j`. For `Linear → Linear` (e.g. attention out → FFN up), absorb `s` into the upstream Linear's weight.

### Weight quantization after scaling
- Per-row asymmetric INT4 with `group_size = 128` (8 groups per 1024-dim row).
- Each group stores INT4 weights + FP16 scale + (optional) zero point.
- Effective bits/weight ≈ 4 + 16/128 = 4.125.

### Hyperparameters
| Knob | Value |
|------|-------|
| Bits | 4 (also 3) |
| Group size | 128 |
| α grid | 20 points in [0, 1] per layer |
| Calibration | 128 sequences × 512 tokens (Pile / C4) |
| Activation precision | FP16 (weight-only) |
| Inference kernel | TinyChat / AWQ / Marlin |

### Why AWQ generalizes better than GPTQ
GPTQ minimises `||W X − Ŵ X||²` against a calibration X — the rounding is overfit to that corpus's covariance. AWQ only uses calibration to estimate `mean(|X_j|)` and pick one scalar α per layer — the per-channel scale is data-cheap and shifts gracefully across domains (instruction-tuned, multilingual, multimodal).

## Connections
- The equivalent-transformation predecessor (W8A8 instead of W4A16): [[smoothquant]].
- The Hessian-based rival (W4A16, often comparable accuracy but worse OOD): [[gptq]].
- Learnable extension (LWC + LET trained by gradient): [[omniquant]].
- Salient-weight-keeping cousins: [[squeezellm]] (dense-and-sparse), [[spqr]], [[owq]].
- Inference framework: [[autoawq]], [[tinychat-and-tensorrt-llm-quant]], [[marlin-kernel]] (compatible).
- Same lab's KV-cache work: [[kivi]], [[atom]].
