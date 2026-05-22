---
chapter: ch-06
course: model-quantization
phase: read
excerpt_of: "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (Jacob et al. 2018)"
source_url: https://arxiv.org/abs/1712.05877
arxiv: 1712.05877
created_at: "2026-05-21"
---

# Excerpt: Jacob 2018 — the integer-only pipeline

**Authors:** Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko
**Year:** 2018 (CVPR)
**Raw-data source:** [[raw-data/classics/integer-only-inference]]

---

## The affine quantization

The (S, Z) convention every later framework adopts:

```math
q \;=\; \text{clamp}\!\left(\, \text{round}(x/S) + Z,\; 0,\; 255 \,\right) \qquad \text{(uint8)}
```

```math
\hat{x} \;=\; S \cdot (q - Z)
```

`S ∈ ℝ₊` (fp32 at compile time), `Z ∈ [0, 255]` integer. Note: the zero-point `Z` is chosen so real `0.0` maps **exactly** to integer `Z` — this is what makes ReLU and zero-padding work correctly under quantization.

---

## Per-layer GEMM (the canonical pipeline)

For weights `W` (int8, scale `S_w`, `Z_w = 0` symmetric), input `X` (uint8, `S_x`, `Z_x`):

```text
1. int32 accumulate:
   acc[i,j] = Σ_k (W[i,k] − Z_w) · (X[k,j] − Z_x)

2. Bias add:
   acc += b                                    # b is int32, S_b = S_w · S_x, Z_b = 0

3. Requantize:
   output = clamp( round(acc · M) + Z_y,  0,  255 )
   where M = S_w · S_x / S_y
```

Three steps. The first is a standard int8 GEMM (gemmlowp / ARM NEON / AVX-512 VNNI / INT8 tensor cores all support it). The third is the load-bearing step.

---

## The fixed-point multiplier (the canonical formula)

`M = S_w · S_x / S_y` is a real number in `(0, 1)` for any sane scale choice. Express it as:

```math
M \;=\; M_0 \cdot 2^{-n}, \qquad M_0 \in [0.5, 1), \quad n \in \mathbb{Z}_+
```

Store `M_0` as int32 (specifically `round(M_0 · 2^{31})`, giving 31-bit fractional precision). `n` is a right-shift count. Then:

```text
int32 · M  ≈  SaturatingRoundingDoublingHighMul(acc, M_0) >> n
```

Two integer ops: a multiply-high (the doubling-high gets one extra bit of precision) and a right-shift. **No floating-point unit ever touched.**

This is what makes int8 inference run on $5 microcontrollers, Edge TPUs, FPGAs, and dedicated NPUs. It's preserved verbatim in every modern integer GEMM kernel — including the fast paths inside CUTLASS and CUDNN's INT8 routines.

---

## Cross-layer chaining

At each layer boundary, `S_y` of layer ℓ is `S_x` of layer ℓ+1, so `M` chains across the whole network. Calibration determines all `(S, Z)` at compile time. The integer pipeline is fully data-independent at runtime — no scale recomputation, no calibration mode, just integer GEMM + integer requantize over and over.

---

## Special-case ops

| Op | How it integrates |
|---|---|
| ReLU | absorbed into the requantize clamp by setting `Q_min = Z_y` instead of 0 |
| Element-wise add | operands rescaled to matching `S` first |
| Concat | needs matching `(S, Z)` across all inputs; calibrated jointly |
| Pooling | average/max preserve `(S, Z)` for the same input |

---

## Simulated quantization (the QAT primitive)

To train a model that will quantize cleanly, insert fake-quant ops in the forward:

```python
def fake_quant(x, S, Z, qmin, qmax):
    # forward: round-trip through int grid in fp space
    return S * (clamp(round(x/S) + Z, qmin, qmax) - Z)
```

Backward is STE — identity gradient within the clip range, zero outside. The model trains in fp32 with fake-quant ops in place; the FP weights are *shadow* weights that get rounded in the forward.

After training: freeze shadows, drop fake-quant ops, integer pipeline runs against quantized weights. Same primitive every later QAT method ([[lsq]], [[lsq-plus]], DoReFa, PACT) uses.

---

## Bias-scale coupling (the underrated detail)

Bias is the small term — keep as int32 with `S_b = S_w · S_x` (no separate calibration). This is critical because:

- Bias values can be up to 100× the weight magnitudes (they absorb layer offsets).
- Quantizing bias to int8 would catastrophically clip; int32 storage solves it.
- The `S_w · S_x` scale aligns bias with the matmul accumulator exactly — no extra rescale needed before adding.

Every later framework (PyTorch FX, TensorRT, TFLite, ONNX) preserves this int32-bias convention.

---

## Empirical effect

| Model | Top-1 FP32 | Top-1 INT8 (Jacob recipe) | CPU speedup (Pixel-2 ARM) |
|---|---|---|---|
| MobileNetV1 | 70.9 | 70.0 (Δ −0.9) | 1.9× |
| MobileNetV2 | 71.7 | 71.2 (Δ −0.5) | 2.1× |
| Inception-V3 | 78.0 | 77.2 (Δ −0.8) | 1.7× |

The MobileNetV2 case is the worst, because depthwise convs have huge per-channel weight range variance — fixed in subsequent work by [[data-free-quantization]] (CLE).

---

## Common pitfalls

- **Per-tensor weight scale on MobileNet-style depthwise.** Per-channel scale is mandatory; per-tensor loses 5–10 top-1 points.
- **Missing the bias int32 storage.** Quantizing bias to int8 destroys accuracy on layers with large offsets.
- **Forgetting `M_0`'s 31-bit precision.** Storing `M_0` as int16 is tempting (fits in a register) but the rounding error in the requantize multiply shows up as accuracy loss in deep networks.
- **Treating `M_0 · 2^{−n}` as a floating-point multiply.** That defeats the entire integer-only point; emit the multiply-high + shift sequence.

---

## What survives into the LLM era

- **(S, Z) convention**: every modern framework.
- **`M = S_w · S_x / S_y`**: every INT8 GEMM kernel in CUTLASS, [[marlin-kernel]], TensorRT.
- **Per-tensor activation scale**: **dies past ~6.7B** ([[llm-int8]]) — superseded by per-token / per-channel.
- **Fake-quant + STE**: still the QAT primitive.

---

## Connections

- [[excerpts/i-bert]] — extends the pipeline to transformer non-linearities (GELU, Softmax, LayerNorm).
- [[excerpts/quantization-mapping]] — the Krishnamoorthi whitepaper sibling to this paper.
- [[ch-06]] — parent synthesis.
- [[ch-07]] — [[llm-int8]] is where the per-tensor activation assumption breaks at LLM scale.
