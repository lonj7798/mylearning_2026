# Excerpt: FP8 Mixed-Precision Training Design

<!-- source: [[deepseek-v3|report]], Section 3.3 -->

## The Precision Landscape

Modern GPUs (H100/H800) have native FP8 tensor cores that deliver roughly 2x the throughput of BF16 for matrix multiplications. But naively quantizing all operations to FP8 causes training divergence — the dynamic range and precision of 8-bit floats is insufficient for many operations.

DeepSeek-V3's solution: **stratified precision**, where each component uses the minimum precision it requires.

```
FP8  (bulk compute):   attention QKV projections, attention output projection,
                        FFN up/down/gate projections, expert FFNs
                        -> These are ALL linear layers (GEMMs)

FP16 (moderate sensitivity): attention score computation (softmax)

FP32 (high sensitivity):     embeddings, output head (vocabulary projection),
                              gating/routing network, RMSNorm,
                              master weights, optimizer states (Adam moments)
```

The rationale: **GEMMs dominate training FLOPs** (>90% of total compute in a Transformer). Everything else — normalization, softmax, routing — is negligible in FLOP count but critical for numerical stability. By running >90% of FLOPs in FP8 and <10% in FP32, DeepSeek-V3 gets most of the throughput benefit with none of the stability cost.

## Fine-Grained Quantization

### The Problem with Per-Tensor Scaling

Standard FP8 quantization uses one scaling factor per tensor:

$$x_\text{FP8} = \text{round}\left(\frac{x}{\text{scale}}\right), \qquad \text{scale} = \frac{\max(|x|)}{448}$$

where 448 is the maximum representable value in E4M3 FP8 format. The problem: if one element in a large tensor has a much larger magnitude than the rest, the scaling factor is set to accommodate that outlier, and all other elements lose precision.

### Tile-Wise and Block-Wise Scaling

DeepSeek-V3 uses **fine-grained scaling**:

| Tensor Type | Quantization Granularity | Scaling Factor Count |
|-------------|--------------------------|---------------------|
| Activations | Tile: 1 x 128 | One per 128 elements along inner dim |
| Weights | Block: 128 x 128 | One per 16,384 elements |

For a GEMM computing $C = A \times B$ where $A$ is activations (shape $M \times K$) and $B$ is weights (shape $K \times N$):

- $A$ is quantized in tiles of size $1 \times 128$ along the $K$ (inner) dimension
- $B$ is quantized in blocks of size $128 \times 128$

The key insight: **scaling along the inner (contraction) dimension ensures that elements being multiplied together share compatible dynamic ranges.** When computing one element of the output, the inner products involve $A$ elements from one tile and $B$ elements from one block — both quantized with their own local scaling factors.

## High-Precision Accumulation

FP8 multiplication ($a \times b$) produces a result that should be accumulated in higher precision:

```
for chunk of 128 elements along inner dimension K:
    partial_sum_fp32 += sum(A_fp8[chunk] * B_fp8[chunk])   # accumulate in FP32

final_result = sum(all partial_sums)                         # combine FP32 partials
```

Without this chunked FP32 accumulation, the inner product would accumulate thousands of FP8 products in lower precision, causing **numerical drift** — small errors that compound across thousands of additions.

The chunk size of 128 matches the quantization tile size, creating a natural alignment: each FP32 accumulation step processes elements that share a single scaling factor.

## Validation

The report establishes a rigorous validation methodology:

1. Train a smaller model (likely ~7B) with both BF16 and FP8 from identical initialization
2. Compare training loss curves — relative error stays **below 0.25%** throughout training
3. At final loss ~2.0, 0.25% relative error = 0.005 nats, well within random seed variance

## What Stays in FP32 and Why

| Component | Why FP32 |
|-----------|----------|
| **Embeddings** | Vocabulary-sized lookup table; quantization errors here affect every subsequent computation. The embedding table is read once per token, so FP8 saves negligible compute. |
| **Output head** | Softmax over vocabulary; small precision errors shift probability mass across 129K tokens. Again, accessed once per token — not a FLOP bottleneck. |
| **Gating network** | Routing decisions are effectively discrete (top-8 selection). Small FP8 errors can flip which experts are selected, causing discontinuous behavior changes. |
| **RMSNorm** | Running statistics (variance) require precision for stability. Normalization is element-wise, not a GEMM — FP8 tensor cores provide no speedup. |
| **Optimizer states** | Adam's first and second moments accumulate over the entire training run. FP8 accumulation here would cause catastrophic drift. |

## Connection to Training Cost

FP8 GEMMs on H800: ~1,979 TFLOPS (FP8) vs ~989 TFLOPS (BF16). The 2x throughput improvement, applied to the >90% of FLOPs that are GEMMs, translates to roughly 40-50% wall-clock reduction for training.

For DeepSeek-V3's 2.788M H800 GPU-hours, this means FP8 saved approximately 1-1.5M GPU-hours — roughly $2-3M at $2/GPU-hour. FP8 alone accounts for a substantial fraction of DeepSeek-V3's cost advantage over BF16-trained competitors.
