<!-- chapter: ch-06
     phase: classical-bridge
     title: Integer-Only Inference — Jacob 2018 + I-BERT
     sources: [[integer-only-inference]], [[i-bert]], [[q8bert]], [[zeroq]], [[q-bert]]
     forward: [[llm-int8]] (ch-07), [[gptq]] (ch-08), [[smoothquant]] (ch-09)
-->

# Chapter 6 — Integer-Only Inference: Jacob 2018 + I-BERT

> **Core insight.** Integer-only inference — no floating-point in the hot path — requires solving two problems independently: (1) the **GEMM requantize** problem, which Jacob 2018 solves with the fixed-point multiplier `M = S_w · S_x / S_y ≈ M_0 · 2^{−n}`; (2) the **non-linearity** problem, which I-BERT solves with three polynomial / shift approximations (i-GELU, i-Softmax, i-LayerNorm). Together they make every op in a transformer integer-executable on commodity ARM / Intel CPUs.
>
> **Guideline.** If targeting integer-only deployment (mobile CPU, FPGA, NPU), use the (S, Z) affine scheme from Krishnamoorthi/Jacob, fold scales across layers via `M = S_w·S_x/S_y`, and replace every non-linearity with its I-BERT approximation. For LLM GPU deployment, the integer-only constraint is *not* what you want — GPUs have fast FP8 / FP16 paths; the lessons of this chapter resurface instead in *INT8 GEMM* + *FP16 outlier path* hybrids ([[llm-int8]] in ch-07).

---

## Why this chapter exists

Ch-05 gave you the (S, Z) playbook for *where to put scales*. This chapter asks the next question: how do you actually *execute* a quantized network in pure integer arithmetic on hardware that has no fp32 unit in the inner loop? That's a different problem — one of numerical engineering, not algorithm design.

Three threads:

1. **Jacob et al. 2018** — the canonical integer-only GEMM pipeline for ResNet/MobileNet on Pixel-2 ARM CPUs. Introduces the fixed-point requantization `M_0 · 2^{−n}` trick and the simulated-quantization (fake-quant) primitive for QAT.
2. **Q8BERT 2019 → I-BERT 2021** — port to transformers. Q8BERT shows INT8 BERT works *if* you keep Softmax/GELU/LayerNorm in fp; I-BERT closes that gap with integer polynomial approximations.
3. **Q-BERT + ZeroQ** — Hessian-aware mixed precision + BN-stat synthetic calibration; these set up *why* you might want different bit-widths per layer, and *how* you might calibrate without data.

The lessons of this chapter mostly become moot for LLMs on GPUs — the integer-only constraint is a CPU/mobile artifact — but two ideas survive into the LLM era: per-token activation quantization (Q-BERT's group-wise idea, mature in [[zeroquant]] ch-08) and the per-tensor *equivalent transformation* paradigm that makes scales fold cleanly across layers (the seed of [[smoothquant]] in ch-09).

---

## 1. The Jacob 2018 integer-only pipeline

The TFLite reference paper, [[integer-only-inference]]. Defines the int8 scheme that became the de facto industry standard for mobile / edge inference.

### Per-layer GEMM — the load-bearing pipeline

For weights `W` (int8, scale `S_w`, `Z_w = 0` symmetric), input `X` (uint8, `S_x`, `Z_x`):

```text
1. int32 accumulate:
   acc[i,j] = Σ_k (W[i,k] − Z_w) · (X[k,j] − Z_x)        # in int32

2. Bias add:
   acc += b                                                # b stored int32, S_b = S_w · S_x

3. Requantize to uint8 (the load-bearing step):
   output = clamp( round( acc · M ) + Z_y,  0,  255 )
   where M = S_w · S_x / S_y
```

`M` is a real number in `(0, 1)` for any sane scale choice. The genius of Jacob 2018 is observing that you can store `M` *as an integer* via the fixed-point trick.

### The fixed-point requantization multiplier (the canonical formula)

Decompose:

```math
M \;=\; M_0 \cdot 2^{-n}, \qquad M_0 \in [0.5, 1), \quad n \in \mathbb{Z}_+
```

Store `M_0` as an int32 (specifically `round(M_0 · 2^{31})`, giving 31-bit fractional precision). `n` is just a right-shift count. Then:

```text
int32 · M  ≈  SaturatingRoundingDoublingHighMul(acc, M_0) >> n
```

Two integer ops: one multiply-high, one right-shift. No floating-point unit ever touched. This single trick is what makes int8 inference run on $5 microcontrollers — and is preserved verbatim in every modern integer GEMM kernel, including the fast paths inside CUTLASS and CUDNN's INT8.

**Cross-layer factoring.** At each layer boundary, `S_y` is the next layer's `S_x`, so `M` chains across the whole network. Calibration determines all `(S, Z)` at compile time → the entire integer pipeline is data-independent at runtime.

**Special-case ops:**

- **ReLU**: absorbed into the requantize clamp by setting `Q_min = Z_y` instead of 0.
- **Element-wise add**: requires matching `S` for the two operands; one is rescaled first.
- **Concat**: needs matching `(S, Z)` across all inputs — calibrated jointly.

### Simulated quantization (the QAT primitive)

During training, insert a fake-quant op in the forward:

```python
def fake_quant(x, S, Z, qmin, qmax):
    # forward: quantize then dequantize (rounds to grid in fp)
    return S * (clamp(round(x/S) + Z, qmin, qmax) - Z)
```

Backward is STE (identity within the clip range). The model trains in fp32 with fake-quant ops in place; the FP weights are *shadow* weights that get rounded in the forward. After training, freeze the FP shadows, drop the fake-quant ops, and the integer pipeline runs against the quantized weights.

This primitive — fake-quant forward + STE backward — is the same one every later QAT method ([[lsq]], [[lsq-plus]], DoReFa, PACT) uses. It's covered in [[ch-04]]; here it's just the QAT entry point into Jacob's integer pipeline.

---

## 2. Q8BERT — the first INT8 transformer (and what it left broken)

[[q8bert]] applies Jacob's recipe to BERT-Base. The good news: GLUE-avg drops only 0.2 from FP32 (82.5 → 82.3) after one extra fine-tuning epoch with fake-quant on all GEMMs. The bad news: Softmax, GELU, LayerNorm, residual-add, and embedding lookup are **kept in fp32**.

Module-by-module audit from the paper:

| Module | Quantized? | Why / why not |
|---|---|---|
| Q/K/V projections | INT8 | standard GEMM, Jacob recipe |
| Attention output projection | INT8 | same |
| FFN both linears | INT8 | same |
| Classifier head | INT8 | same |
| **Softmax** | fp32 | needs `exp` — no integer primitive |
| **GELU** | fp32 | needs `erf` — no integer primitive |
| **LayerNorm** | fp32 | needs `1/√(Var+ε)` — no integer primitive |
| Residual add | fp32 | mismatched scales |
| Embedding lookup | fp32 | lookup, not GEMM |

This is *not* integer-only inference. It's INT8-GEMM-with-fp32-non-linearities. To actually delete all fp32 paths from a transformer, you need integer approximations of the three non-linearities. That's the I-BERT contribution.

### Practical Q8BERT lessons that stuck

- **Per-channel weight scales are mandatory.** Per-tensor weights cost 1.5–2 GLUE points on MNLI — observed empirically by Q8BERT, restated by every later transformer-quant paper.
- **Activation calibration on a single batch is unlucky.** Use EMA over the first ~5 batches with momentum 0.99.
- **LayerNorm params (γ, β) stay fp.** Too few parameters to compress, too sensitive to quantize.

---

## 3. I-BERT — the three integer approximations

[[i-bert]]. The first end-to-end integer-only transformer. It plugs into Jacob's GEMM pipeline by replacing the three non-linearities with integer polynomial / shift approximations.

### i-GELU — polynomial fit of erf (the load-bearing formula)

GELU: `GELU(x) = (x/2)(1 + erf(x/√2))`. I-BERT approximates `erf` with a *single squared polynomial*:

```math
L(x) \;=\; \text{sign}(x) \cdot \Bigl[\, a \cdot \bigl(\text{clip}(|x|,\, 0,\, -b) + b\bigr)^2 + 1 \,\Bigr],
\quad a = -0.2888,\; b = -1.769
```

so:

```math
\text{i-erf}(q_x) \;=\; \text{sign}(q_x) \cdot \Bigl[\, a \cdot \bigl(\text{clip}(|q_x|,\, 0,\, -b/S_x) + b/S_x\bigr)^2 \cdot S_x^2 + 1 \,\Bigr]
```

```math
\text{i-GELU}(q_x) \;=\; (q_x / 2) \cdot \bigl(1 + \text{i-erf}(q_x)\bigr)
```

One squared-poly + a few shifts. Max approximation error vs true GELU: **< 1e-3** over the active range.

### i-Softmax — the `2^x` shift trick

Standard: `s_i = exp(x_i − max_x) / Σ_j exp(x_j − max_x)`.

Use the identity `exp(x) = 2^{x · log₂ e} = 2^{x · 1.4427}`. For integer `x' = round(x · 1.4427)`:

```text
1. Split: x' = q · 1  +  r, with r ∈ [−1, 0)        # integer q, fractional r
2. 2^q is a bit shift.
3. 2^r ≈ 0.3585·r² + 0.353·r + 0.344                # tiny polynomial, |r| ≤ 1
4. exp(x) ≈ 2^q · 2^r                                # integer pieces
5. Normalize: 1/Σ via integer reciprocal table or one Newton step.
```

The exponential is now a shift + a 3-term polynomial. No fp `exp` in the kernel.

### i-LayerNorm — Newton-Raphson integer sqrt

LayerNorm needs `1/√(Var + ε)`. Compute `Var` as int32 sum-of-squares (straightforward). Then integer `sqrt(v)` via Newton-Raphson:

```text
x_{n+1} = (x_n + ⌊v / x_n⌋) >> 1
```

Converges in `⌈log₂ bit_width⌉` ≈ 5 steps for int32. Final scale absorbs `γ`, shift absorbs `β`; rescale output to int8.

### Integer pipeline contract

Every module takes `(q ∈ int8, S ∈ fp metadata, Z ∈ int8)` and emits the same triple. Scales fold into the next layer's `M = S_in · S_op / S_out` at compile time (Jacob 2018 requantization). **Runtime: zero fp ops.**

### Empirical effect

| Metric | BERT-Base FP | I-BERT |
|---|---|---|
| GLUE-avg | 83.2 | 83.0 (Δ = −0.2) |
| GLUE-avg, BERT-Large | 85.5 | 85.4 (Δ = −0.1) |
| Latency speedup (BERT-Base, Intel Cascade Lake AVX-512) | 1.0× | **2.4×** |
| Latency speedup (BERT-Large) | 1.0× | **4.0×** |

The non-linearity polynomials cost ~5% of inference time; the GEMM savings dwarf them.

---

## 4. Q-BERT — Hessian-aware mixed precision for transformers

[[q-bert]] is HAWQ (covered in [[ch-05]]) applied to BERT. The novelty for transformers is **per-block** sensitivity measurement and the observation that sensitivity varies 100× across BERT-Base layers.

### Per-block Hessian sensitivity

For block `ℓ` with weights `W_ℓ`:

```math
\Omega_\ell \;=\; \lambda_{\max}(H_\ell) \cdot \|\Delta W_\ell(b)\|^2
```

`λ_max(H_ℓ)` estimated by power iteration with Hessian-vector products via autograd double-backward — no need to form `H`. Typical Q-BERT allocation on BERT-Base:

| Block | Bits (W) | Bits (A) |
|---|---|---|
| Embedding | 8 | 8 |
| Mid-attention | 4 | 8 |
| FFN | 3 | 8 |
| Pooler | 2 | 8 |

### Group-wise activation quantization (the seed of per-token)

Q-BERT also introduced **group-wise activation quantization** at attention-head granularity: split activations into G groups (G = num_heads) along the channel dim; each group has its own `(S_g, Z_g)`. This is the precursor to ZeroQuant's per-token quant and SmoothQuant's per-channel migration.

### Empirical effect

| Setting | BERT-Base GLUE-avg | Model size |
|---|---|---|
| FP32 | 82.5 | 415 MB |
| Q-BERT mixed 2-3 bit | 80.2 | 32 MB (**13× compression**) |
| Q-BERT mixed 4-bit | 82.0 | ~50 MB |

13× compression at 2.3 GLUE drop. Compare with LLM-era: [[gptq]] (ch-08) uses **uniform 4-bit + group_size=128** and beats this on much larger models, because LLM loss landscapes are flatter per-block than BERT's. The mixed-precision allocation idea survives mostly as the "which layers to skip" decision in production stacks (e.g. AutoGPTQ excludes `lm_head` and `embed_tokens`).

---

## 5. ZeroQ — synthetic-data calibration when you have no calibration set

[[zeroq]] is the no-data version of PTQ for CNNs. It tackles the deployment scenario where calibration data is unavailable (privacy, gating, cross-organisation handoff).

### BN distillation loss

For each conv layer with BatchNorm, optimise a synthetic input batch `x` to match the FP model's BN running statistics:

```math
L_{\text{dist}}(x) \;=\; \sum_\ell \bigl\| \mu_\ell(x) - \mu_\ell^{\text{BN}} \bigr\|^2_F
\;+\; \bigl\| \sigma^2_\ell(x) - \sigma^{2,\text{BN}}_\ell \bigr\|^2_F
```

`μ_ℓ(x), σ²_ℓ(x)` are the per-channel mean/variance computed on the forward pass of `x` through the FP model up to layer `ℓ`.

Procedure:

1. Initialise `x ~ N(0, 1)`, shape `(32, 3, 224, 224)`.
2. Adam-optimise `x` for ~500 steps minimising `L_dist(x)`.
3. Use `x` as the synthetic PTQ calibration set.

Total wall clock: ~30 seconds on V100 (synthesis + PTQ + HAWQ allocation). The synthesised images don't look like ImageNet — they look like noise that happens to match the per-layer BN moments — but they suffice because PTQ only cares about the *marginal* per-channel distribution, not realistic semantics.

### Why it doesn't transfer to LLMs

Transformers have no BatchNorm — only LayerNorm with no running statistics to anchor on. The data-free idea survives in **LLM-QAT** (covered later), which replaces BN distillation with *teacher-generated text* as the calibration source. Same recipe, different distributional anchor.

### Combining with HAWQ

ZeroQ + HAWQ gives data-free mixed-precision PTQ:

| Setting | ResNet-50 ImageNet top-1 |
|---|---|
| FP32 | 77.7 |
| Uniform 6-bit PTQ (no data) | crashes |
| **ZeroQ + HAWQ 6-bit mixed (no data)** | **77.2 (Δ = −0.5)** |
| PACT 6-bit QAT (with data) | 77.4 |

Within 0.2% of QAT with real data. The "no data" claim is real.

---

## 6. What integer-only gives you, and what it costs

| Benefit | Concrete artifact |
|---|---|
| 2-4× CPU inference speedup | Pixel-2 MobileNet (Jacob); Cascade Lake BERT-Large (I-BERT) |
| Zero fp32 unit usage | runs on M0-class MCUs, TPU/Edge TPU, FPGA |
| Deterministic numerics | bit-exact across hardware (unlike fp16's NaN edge cases) |
| Power | INT8 multiply is ~4× cheaper than fp32 multiply in silicon |

| Cost | Concrete artifact |
|---|---|
| Engineering: every non-linearity needs an integer approximation | i-GELU, i-Softmax, i-LayerNorm; each is its own bug surface |
| Numerical: int8 has 256 levels; outliers clip hard | Q-BERT needs group-wise activation quant to survive |
| QAT bootstrap: simulated quantization adds ~10% to training time | Q8BERT needs 1 extra epoch |
| Calibration scales are baked in at compile time | model retargeting (new domain) needs recalibration |

### What survives into the LLM era — and what dies

| Idea | Survives? | Where |
|---|---|---|
| (S, Z) affine quantizer | ✓ | every modern framework |
| Fixed-point `M_0 · 2^{−n}` requantize | ✓ | INT8 GEMM kernels in CUTLASS, [[marlin-kernel]] |
| Simulated quantization (fake-quant + STE) | ✓ | every QAT method, [[llm-qat]] |
| Per-tensor activation scale | **dies past 6.7B** | superseded by per-token / per-channel ([[llm-int8]], [[zeroquant]]) |
| Integer Softmax / GELU / LayerNorm | mostly dies on GPU | GPUs have fast fp16 paths; survives on edge (mobile LLM, TFLite) |
| BN-stat synthetic calibration | dies (no BN in transformers) | replaced by teacher-generated text in [[llm-qat]] |
| Per-block Hessian mixed precision | mostly dies | replaced by uniform W4 + group_size=128 in [[gptq]] |

---

## 7. Practitioner's checklist for integer-only deployment

```text
□ Target hardware has integer GEMM unit? (ARM NEON, Intel AVX-512 VNNI, NVIDIA INT8 tensor cores, NPU/TPU)
□ All non-linearities replaced with integer approximation? (i-GELU, i-Softmax, i-LayerNorm — or fall back to dequant→fp→requant)
□ Per-channel weight scales calibrated? (per-tensor weight scale loses 1.5–2 points on transformers)
□ Activation scales calibrated with EMA over ≥5 batches?
□ Bias kept as int32 with S_b = S_w · S_x?
□ Requantize multipliers M = S_w·S_x/S_y precomputed as (M_0 int32, n shift)?
□ ReLU absorbed into requantize clamp? (saves a kernel launch)
□ Residual-add operands rescaled to matching S?
□ Embedding lookup either fp16 or int8 with separate per-row scale?
□ Skip quantization on layers where Δaccuracy > target?  (typically lm_head, embed_tokens)
```

---

## Connections and what's next

- **Forward to [[llm-int8]] (ch-07)** — abandons integer-only, keeps INT8 GEMM + FP16 outlier path. The first crack in the Jacob 2018 framework: per-tensor activation scales fail past 6.7B.
- **Forward to [[zeroquant]] (ch-08)** — per-token activation scale + group-wise weight scale. The "group-wise" idea is Q-BERT's per-head group quant, scaled up.
- **Forward to [[smoothquant]] (ch-09)** — the equivalent-transformation idea from DFQ (ch-05), reapplied to migrate activation outliers into weights so per-tensor or per-channel INT8 activations work again.
- **Back to [[quantization-mapping]] (ch-05)** — Krishnamoorthi's (S, Z) playbook is the foundation Jacob's pipeline operationalises.
- **Back to [[adaround]] / [[brecq]] (ch-04)** — calibration objectives that improve PTQ rounding decisions for the Jacob integer pipeline.

## Further reading

- [[integer-only-inference]] — Jacob et al. 2018, the TFLite reference paper.
- [[i-bert]] — Kim et al. 2021, integer-only transformer.
- [[q8bert]] — Zafrir et al. 2019, INT8 BERT via QAT.
- [[q-bert]] — Shen et al. 2020, Hessian-aware mixed-precision BERT.
- [[zeroq]] — Cai et al. 2020, BN-stat synthetic calibration.
- [[gemmlowp]] — Google's open-source INT8 GEMM library used in TFLite (referenced from Jacob 2018).
