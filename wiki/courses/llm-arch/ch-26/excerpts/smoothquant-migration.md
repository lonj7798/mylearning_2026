# Excerpt: SmoothQuant — Migrating Quantization Difficulty

<!-- source: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (2023) -->

## The Activation-Weight Asymmetry

Before SmoothQuant, the quantization community treated weight quantization and activation quantization as independent problems. Weight quantization to INT8 was considered "solved" — weight distributions are smooth and well-behaved. Activation quantization was the hard part: LLM activations contain outlier features (specific hidden dimensions with magnitudes 10-100x the mean) that make uniform INT8 quantization destructive.

SmoothQuant's insight is that these are not independent problems. In a matrix multiplication $Y = XW$, you can redistribute the quantization burden between $X$ (activations) and $W$ (weights) without changing the result.

## The Mathematical Trick

For any per-channel diagonal matrix $\text{diag}(s)$ with $s_j > 0$:

$$Y = XW = (X \, \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \, W) = \hat{X} \hat{W}$$

This identity holds exactly. Choosing $s_j$ appropriately can equalize the quantization difficulty between activations and weights:

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

where $X_j$ is the $j$-th channel of the activation tensor across the calibration set, $W_j$ is the $j$-th input channel of the weight matrix, and $\alpha \in [0, 1]$ controls the migration strength.

### What alpha controls

| $\alpha$ | Effect |
|----------|--------|
| 0 | No smoothing — all difficulty stays in activations |
| 0.5 | Geometric mean — difficulty split evenly (default) |
| 0.75 | More difficulty shifted to weights |
| 1.0 | All difficulty shifted to weights |

In practice, $\alpha = 0.5$ works for most layers. Some layers with extreme outliers benefit from $\alpha = 0.75$.

## Worked Example

Consider a single matmul where activations have one outlier channel:

**Before smoothing:**
- Activation channel magnitudes: $[0.5, 0.4, 0.6, 50.0, 0.3]$
- Weight channel magnitudes: $[1.2, 0.8, 1.0, 0.5, 0.9]$

The activation range is dominated by channel 3 (magnitude 50.0). Quantizing to INT8 with $\Delta_X = 50.0/127 \approx 0.394$, channels 0-2, 4 map to only $\pm 1$ — nearly all precision is wasted.

**Smoothing factors** ($\alpha = 0.5$):

$$s_3 = \frac{50.0^{0.5}}{0.5^{0.5}} = \frac{7.07}{0.71} = 10.0$$

**After smoothing:**
- Smoothed activation channel 3: $50.0 / 10.0 = 5.0$
- Smoothed weight channel 3: $0.5 \times 10.0 = 5.0$
- Activation range is now $[0.5, 0.4, 0.6, 5.0, 0.3]$ — much more uniform
- Weight range is now $[1.2, 0.8, 1.0, 5.0, 0.9]$ — slightly less uniform, but weights tolerate this

Now $\Delta_X = 5.0/127 \approx 0.039$ — channels 0-2, 4 get $\pm 10$ to $\pm 15$ integer levels instead of $\pm 1$. The information content is preserved.

## Why Weights Are More Tolerant

Weights are static — they do not change per input. This means:
1. Their distribution is fixed and can be characterized exactly from the model checkpoint
2. Per-channel quantization is trivially applicable (each output channel gets its own scale)
3. They have no outlier features (the outlier phenomenon is specific to activations in transformers)

Shifting outlier magnitude from activations to weights makes the weight distribution slightly less uniform, but this is a minor perturbation that per-channel INT8 quantization handles easily.

## Enabling W8A8 Inference

SmoothQuant's contribution is enabling **W8A8** quantization — both weights and activations in INT8. Previous INT8 schemes were either:
- **W8A16**: weights in INT8, activations in FP16 (GPTQ, RTN) — requires dequantization before matmul, cannot use INT8 tensor cores
- **W8A8 with mixed precision**: LLM.int8() keeps outlier channels in FP16 — requires custom kernels, splits the matmul

With SmoothQuant, the entire matmul runs in INT8, utilizing the full throughput of INT8 tensor cores (~2x FP16 throughput on A100/H100). The smoothing factors are precomputed offline and folded into the preceding layer norm's affine parameters, adding zero runtime overhead.

## Results from the Paper

On OPT-175B with W8A8 quantization:

| Method | WikiText-2 PPL | Latency (ms) |
|--------|---------------|--------------|
| FP16 baseline | 8.34 | 1.00x |
| LLM.int8() (mixed) | 8.41 | 0.95x (overhead from splitting) |
| Naive W8A8 | >1000 (broken) | — |
| SmoothQuant W8A8 | 8.42 | 0.56x (1.8x speedup) |

Naive W8A8 completely fails on 175B models because outlier activations destroy the quantization. SmoothQuant matches LLM.int8() quality while being nearly 2x faster because it avoids the mixed-precision split.

## Scope and Limitations

- **Granularity:** SmoothQuant operates per-channel, not per-tensor. Per-tensor smoothing would be too coarse.
- **Calibration:** Requires a small calibration set (typically 512 examples) to estimate activation magnitudes. The smoothing factors are fixed after calibration.
- **Not for sub-8-bit:** SmoothQuant targets W8A8 specifically. For INT4 weight quantization, GPTQ and AWQ are more appropriate (they quantize only weights, not activations).
- **Layer-specific alpha:** While $\alpha = 0.5$ works for most layers, some architectures benefit from per-layer $\alpha$ tuning. The paper provides a simple heuristic for this.
