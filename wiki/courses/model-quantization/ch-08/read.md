<!-- chapter: ch-08
     phase: llm-ptq-2022
     title: GPTQ at Scale + ZeroQuant Family
     sources: [[gptq]], [[obc]], [[zeroquant]], [[zeroquant-v2]], [[zeroquant-fp]], [[nuqmm]], [[marlin-kernel]]
     forward: [[awq]] (ch-09), [[omniquant]] (ch-10), [[spqr]] (ch-11), [[machete-kernel]] (ch-19)
-->

# Chapter 8 — GPTQ at Scale + ZeroQuant Family

> **Core insight.** GPTQ is OBC ([[ch-05]]) with two engineering tricks — a Cholesky-factor representation of `H⁻¹` and a *lazy block update* — that drop the cost of a per-layer OBS sweep from intractable to ~4 GPU-hours for OPT-175B. Nothing else changes. The Hassibi 1993 mathematics is unchanged; what's new is making `O(d³)` runnable when `d = 16384`. The ZeroQuant family attacks the same problem from the *granularity* angle (group-wise weights + per-token activations + layer-wise distillation) and adds the empirical case for **FP4/FP8 beating INT at low bits** that motivates the entire 2024 NVFP4 line.
>
> **Guideline.** For weight-only W4 LLM PTQ in 2026: use **GPTQ with `group_size=128`, `actorder=True`, `percdamp=0.01`**, 128 calibration sequences × 2048 tokens of C4. Pair with [[marlin-kernel]] (Ampere) or [[machete-kernel]] (Hopper) for serving. For weight-and-activation W8A8: use **ZeroQuant (group-wise weights + per-token activations + LKD)** for INT; for FP-native hardware (H100/Blackwell), use **ZeroQuant-FP (E2M1 weights + E4M3 activations)** which gives 0.3+ ppl lift over INT at the same bit budget.

---

## Why this chapter exists

By August 2022, three things were true:

1. [[llm-int8]] (ch-07) had proven INT8 PTQ works at 175B if you handle outliers.
2. [[obc]] (ch-05) had proven the OBS framework cleanly unifies pruning + quantization at BERT scale.
3. Nobody had a *weight-only* PTQ that pushed below 8 bits for an LLM. The memory bottleneck for a 175B model is the weights, not the activations; LLM.int8's 50% memory cut wasn't enough.

GPTQ ([[gptq]], October 2022) was the answer. Frantar & Alistarh — the same group that wrote OBC six months earlier — ported their algorithm to LLM scale with two crucial engineering moves: a Cholesky reformulation that turns the per-column OBS update into a triangular solve, and a lazy block update that defers Hessian propagation until you've finished a block of 128 columns. Result: OPT-175B at 4-bit in ~4 GPU-hours with **<1 perplexity loss**. The era of W4 deployment had begun.

In parallel, Microsoft's ZeroQuant line ([[zeroquant]], May 2022; V2 March 2023; FP July 2023) attacked the same problem from the *granularity* side. Group-wise weights + per-token activations was their finding; layer-wise knowledge distillation (LKD) was the cheap data-free recovery step. ZeroQuant-FP added the empirical fact that on H100, **FP8 activations beat INT8 activations** and **FP4 weights match or beat INT4 weights** — the result that justifies the entire NVFP4 / MXFP4 push covered in ch-17.

This chapter covers both threads, then the [[marlin-kernel]] that turns the GPTQ algorithm into production-grade GEMM throughput.

---

## 1. GPTQ — porting OBS to LLM scale

### The same objective, the same setup as OBC

For each linear layer with weights `W ∈ ℝ^{d_out × d_in}` and calibration activations `X ∈ ℝ^{d_in × N}` (N = batch tokens, typically 128 × 2048 = 262144):

```math
\hat{W}^{*} \;=\; \arg\min_{\hat{W} \in \mathcal{Q}} \, \bigl\| W X - \hat{W} X \bigr\|_F^2
```

`𝒬` is the set of representable quantized weights (per-row or per-group scaled INT-k). The Hessian of this layer-local quadratic is:

```math
H \;=\; 2\, X X^\top \;\in\; \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}
```

Shared across all output rows of W → compute once, reuse `d_out` times. Computable from a single forward pass over the calibration batch — **no backward**, no gradient, no training.

### The OBS column update (the load-bearing per-column formula)

For column `q` of weight row `w`, the optimal rounding direction and the compensation for the remaining columns:

```text
w_q   :=  quant(w_q)                                                    # round to int4 grid
δ_F   := −(w_q − quant(w_q)) / [H⁻¹]_{qq}  ·  [H⁻¹]_{q, F}              # F = remaining
w_F   :=  w_F + δ_F                                                     # apply correction
```

This is literally OBC, literally OBS. The novelty is what comes next.

### Cholesky reformulation (engineering trick #1)

Instead of recomputing `H⁻¹` after each column (`O(d³)` per column = `O(d⁴)` total), GPTQ pre-computes the upper-triangular **Cholesky factor `R`** of `H⁻¹` once:

```math
H \;=\; L\, L^\top \quad \Rightarrow \quad H^{-1} \;=\; L^{-\top} L^{-1} \;=\; R^\top R
```

The quantities the OBS update needs — `[H⁻¹]_qq` and `[H⁻¹]_{q, F}` — can be read directly from `R`. After processing column `q`, `R` gets a rank-1 downdate (one triangular solve, `O(d²)`).

Total per-layer cost: `O(d³)` for the initial Cholesky + `d × O(d²)` per-column updates = **`O(d³)` total**. This is the OBC contribution; GPTQ inherits it.

### Lazy block update (engineering trick #2 — the real novelty)

Process columns in **blocks of B = 128**:

```text
For block b in 0, B, 2B, ..., d_in:
    For column q in block b:
        local OBS update: quantize column q against current local sub-problem
        compensate ONLY columns within this block immediately
    At block boundary:
        apply accumulated correction to ALL remaining columns (b+B, ..., d_in)
        as a single dense GEMM
```

Why this matters: per-column updates are *memory-bound* (touch the entire remaining-column tensor). Batching them turns the inner loop into a **compute-bound dense matmul**, which GPUs love. Empirical speedup: ~10× over the per-column OBC implementation.

### Damping (`percdamp`)

H is often near-singular (calibration X has rank ≤ N; for short calibration sets, H is rank-deficient). Damp before Cholesky:

```math
H \;\leftarrow\; H + \lambda I, \qquad \lambda \;=\; \text{percdamp} \cdot \text{mean}(\text{diag}(H))
```

Standard `percdamp = 0.01`. Set higher (0.1) if Cholesky fails (NaN); set lower (0.001) only if you have a large calibration set and want maximal accuracy.

### Activation ordering (`actorder`)

Quantize columns in **descending `diag(H)` order**. The intuition: high-activation-energy columns are the ones whose rounding error propagates most through the layer's output. By processing them first (when the slack from `[H⁻¹]_{qq}` is largest), they get the cleanest rounding; their errors are then compensated by the rounding of low-energy columns.

Empirical effect: ~0.1–0.3 perplexity improvement at 4-bit on Llama models. Free, no compute cost (one sort).

### Grouping (`group_size`)

Per-output-channel scale is too aggressive at low bits — single rows have weights spanning 100× dynamic range. **`group_size = 128`** stores one (scale, zero_point) per 128 consecutive input dims per output row. Storage overhead: ~0.05 bits/element at INT4 (each group needs 16 bits scale + 4 bits zero_point ÷ 128 = 0.156 bits). Accuracy gain: 2–5 perplexity points at INT3/INT4 versus per-channel.

### The standard recipe

| Knob | Value |
|---|---|
| Bits | 4 (also 3 with [[quip]] preprocessing, 2 with [[aqlm]]) |
| `group_size` | 128 |
| `actorder` | True |
| `percdamp` | 0.01 |
| Block size B | 128 |
| Calibration samples | 128 sequences × 2048 tokens |
| Calibration source | C4 / WikiText / domain text |
| Symmetric / asymmetric | asymmetric (zero-point) for INT4 |

### Empirical effect (the paper's headline numbers)

| Model | FP16 WikiText-2 PPL | GPTQ W4 PPL | GPTQ W3 PPL | Wall-clock to quantize |
|---|---|---|---|---|
| OPT-175B | 8.34 | 8.37 (Δ +0.03) | 8.68 (Δ +0.34) | ~4 GPU-hours |
| BLOOM-176B | 8.11 | 8.21 (Δ +0.10) | 8.64 (Δ +0.53) | ~4 GPU-hours |
| OPT-66B | 9.34 | 9.55 (Δ +0.21) | 9.99 (Δ +0.65) | ~2 GPU-hours |
| OPT-13B | 10.13 | 10.31 | 11.61 | ~30 min |
| Llama-2-7B (post-paper, AutoGPTQ) | 5.47 | 5.69 | 6.30 | ~10 min |

W4 GPTQ is within 0.1 perplexity of FP16 at 175B scale, runs as a single-GPU job, requires no retraining. This is the watershed result that made W4 LLM deployment mainstream.

### Why GPTQ won the production stack

- **No training data needed.** Calibration text is enough.
- **No backward pass.** Forward-only Hessian; runs on inference hardware.
- **One-shot.** No iterative refinement, no convergence diagnostics.
- **Format compatible.** Output is packed INT4 + per-group FP16 scales — exactly what [[marlin-kernel]] and [[machete-kernel]] consume.
- **Numerically stable.** Cholesky + percdamp avoids the failure modes of naive H⁻¹.

The cost: GPTQ doesn't see activation distribution shifts caused by quantizing upstream layers (it's per-layer, not per-block — see [[brecq]] in ch-05). [[awq]] (ch-09) addresses this with activation-aware scaling; [[omniquant]] (ch-10) addresses it with block reconstruction.

---

## 2. ZeroQuant — the granularity-focused alternative

[[zeroquant]] takes a different tack. Instead of a sophisticated OBS rounding algorithm, it argues that the *right granularity* + a cheap data-free distillation step is enough for W8A8 LLM PTQ.

### Three pieces

**(1) Group-wise symmetric weight quant.** For weight `W ∈ ℝ^{d_out × d_in}`, partition each row into groups of G consecutive input dims:

```math
s \;=\; \max(|W_g|) / (2^{b-1} - 1), \qquad \hat{W}_g \;=\; \text{round}(W_g / s) \cdot s
```

`b = 8` (or 4), `G = 128–256`. Same as GPTQ's grouping.

**(2) Per-token (dynamic) activation quant.** For activation `X ∈ ℝ^{B × S × d}`, compute one scale per token at runtime:

```math
c_{b, s} \;=\; \max_d |X_{b,s,d}| / 127, \qquad \hat{X}_{b,s,d} \;=\; \text{round}(X_{b,s,d} / c_{b,s})
```

No calibration set needed; the scale is recomputed every forward pass. Dequantization folds into the next op via output scale `c_{b,s} · s_w`. This is the activation analogue of GPTQ's per-channel weight scale.

**(3) Layer-wise Knowledge Distillation (LKD).** A memory-cheap distillation that only requires one transformer block resident at a time:

```text
for each transformer block f_i in order:
    1. Run FP teacher + partially-quantized student up to block i on calibration data.
    2. Quantize block f_i's weights → student block f̂_i.
    3. Optimize block-i quant params to minimize ||f_i(h) − f̂_i(h)||² on calibration samples.
    4. Only block i in full precision in memory → fits on single GPU for 20B models.
```

LKD is decisive at W4: pure RTN W4 loses 5–10 ppl; LKD recovers within 1 ppl of FP. The MSE objective at block grain is literally [[brecq]]'s objective without the Fisher weighting.

### Recipe

| Knob | Value |
|---|---|
| Weight bits | 8 (FC + attn) or W4-FC / W8-attn mixed |
| Weight group size | 128–256, symmetric |
| Activation bits | 8 |
| Activation scale | per-token dynamic |
| LKD samples | 128–1024 |
| LKD optimizer | Adam, lr 5e-7, ≤ 1 epoch |

### Empirical effect

| Setting | GPT-NeoX-20B PPL | Speedup |
|---|---|---|
| FP16 | 9.40 | 1.0× |
| **W8A8 ZeroQuant** | **9.45 (Δ +0.05)** | 4.16× |
| W4A8 ZeroQuant (with LKD) | 9.62 | ~5× |

W8A8 essentially lossless; W4A8 within 0.3 ppl. Compare with LLM.int8 (W8A8 INT8 + FP16 outlier): ZeroQuant is faster (no outlier path overhead) but requires calibration + LKD. The two methods occupy different corners of the design space.

---

## 3. ZeroQuant-V2 — the LoRC residual

[[zeroquant-v2]] is two papers in one. First: a systematic sensitivity benchmark across (model family × size × bit-width × W-only vs A-only). Second: **Low-Rank Compensation (LoRC)** — store the per-layer quantization error as a low-rank residual.

### LoRC (the load-bearing formula)

Given quantized weight `Ŵ` (from RTN or GPTQ) and full-precision `W`, the residual is `E = W − Ŵ`. Store:

```math
E \;\approx\; U V^\top, \quad U \in \mathbb{R}^{d_{\text{out}} \times r}, \; V \in \mathbb{R}^{d_{\text{in}} \times r}
```

via truncated SVD `E = U_1 Σ_1 V_1^⊤` keeping the top-r singular vectors. At inference:

```math
y \;=\; (\hat{W} + U V^\top)\, x \;=\; \hat{W}\, x + U\, (V^\top x)
```

The right-multiply `V^⊤ x` is rank-r and cheap (FP16). Memory cost: `r · (d_out + d_in) · 16` bits — for `r = 8` on a 4096×4096 layer this is ~0.06 bits/weight extra. Tiny.

### The key empirical findings from the sensitivity study

1. **Activation quantization is more sensitive than weight quantization** for low bits. → focus engineering on activations (motivates [[smoothquant]] / [[awq]]).
2. **Smaller models are sometimes *more* robust to activation quant than larger ones.** Outlier emergence again.
3. **No existing PTQ holds quality at W4A4 or W4A8 universally.** Different model families fail at different settings — calibration matters. (This is the gap [[quarot]] eventually closes.)

### LoRC ablation

| Setting | Llama-7B Wiki PPL |
|---|---|
| FP16 | 5.68 |
| GPTQ W4 | 5.85 |
| GPTQ W4 + LoRC r=8 | 5.74 |
| GPTQ W4 + LoRC r=16 | 5.71 |
| GPTQ W4 + LoRC r=32 | 5.70 |

Diminishing returns past r=16 for 7B-class. Useful when GPTQ alone isn't enough but full fine-tuning is too expensive.

---

## 4. ZeroQuant-FP — the FP4/FP8 verdict

[[zeroquant-fp]] is the paper that justifies the entire 2024 NVFP4 push. The empirical claim:

> For LLM PTQ at low bit-widths, floating-point formats (FP8 for activations, FP4 for weights) consistently beat integer formats of the same width because the wider dynamic range of FP absorbs outliers that INT clips.

### The FP formats (covered in ch-02)

- **E4M3** (4 exponent + 3 mantissa, bias = 7): used for weights and activations at FP8. Range ≈ [−448, 448]; one NaN bit-pattern; no infinities.
- **E5M2** (5 exp + 2 mantissa, bias = 15): used for gradients in FP8 training (not relevant here).
- **E2M1** (2 exp + 1 mantissa, FP4): used for weights at FP4. Represents 16 values: `{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.

### The quantization rule

For each weight tensor `W`:

```math
s \;=\; \max(|W|) / \text{max\_repr(format)}
```

```math
\hat{W} \;=\; \text{nearest\_fp}(W / s) \cdot s
```

where `nearest_fp(·)` is round-to-nearest in the FP format's representable set.

### The W4A8-FP recipe

| Knob | Value |
|---|---|
| Weight format | FP4-E2M1 |
| Activation format | FP8-E4M3 |
| Weight scale | per-tensor (power-of-2 constrained) |
| Activation scale | per-token, dynamic |
| LoRC rank | 8 |
| Hardware target | H100 (FP8 tensor cores) |

The power-of-2 scale constraint lets the rescale into the FP accumulator be a bit-shift. The activation × weight scale product is bounded so the partial sum stays in FP16/FP32 accumulator. These two constraints lose <0.05 ppl vs unconstrained.

### The empirical verdict

| Format combo | LLaMA-7B PPL | LLaMA-30B PPL |
|---|---|---|
| FP16 baseline | 5.68 | 4.10 |
| INT8 activations + INT4 weights | 6.31 | 4.95 |
| FP8 activations + INT4 weights | 5.93 | 4.32 |
| **FP8 activations + FP4 weights (W4A8-FP)** | **5.89** | **4.29** |

FP wins by ~0.4 ppl on 7B, ~0.5 ppl on 30B. The gap *widens* with model scale — the larger the model, the more the FP dynamic-range advantage pays off (because larger models have larger outliers).

This is the result that justifies the 2024 hardware push to native FP4/FP8: NVFP4 in [[nvfp4-training]], MXFP4 in [[mxfp4-pretraining]], the Blackwell tensor cores. Once you have native FP4 hardware, the integer formats become unnecessary for low-bit deployment.

---

## 5. nuQmm / LUT-GEMM — the LUT kernel alternative

[[nuqmm]] is worth knowing for the *kernel* design space (it's not a quantization algorithm; it's a GEMM kernel for already-quantized weights). The core trick: **dequant-free GEMM via LUTs**.

### Binary-coded quantization (BCQ) format

For weight tile `W ∈ ℝ^{d × G}` with group size `G` and `b` bits:

```math
W \;\approx\; \sum_{i=1}^{b} \alpha_i \cdot B_i, \qquad B_i \in \{-1, +1\}^{d \times G}, \quad \alpha_i \in \mathbb{R}^{d}
```

`α`'s are found by alternating optimization on calibration. 1-bit = sign quant; b-bit = b binary planes summed.

### The LUT-GEMM kernel

For input activation `x ∈ ℝ^G` and a BCQ weight tile:

```text
1. Split x into 8-bit-wide chunks (one byte at a time).
2. For each binary plane B_i:
   - Inner product B_i · x over each chunk depends only on the chunk's sign pattern.
   - Precompute a 256-entry LUT keyed by the 8-bit sign-pattern index.
3. Full output: y = Σ_i α_i · LUT_i[chunk_pattern]
```

No FP dequantization: weights live as packed binary planes in DRAM, the LUT is in shared memory, accumulation is in FP16/FP32. The dequant step is replaced by table lookups.

### Empirical

Speedup vs GPTQ-INT4 kernels at 3-bit on OPT-30B/175B: **~2.1×** at batch 1. The advantage shrinks at higher batch because LUT lookups don't scale as well as tensor-core MMA.

### Why GPTQ + Marlin won instead

LUT-GEMM is competitive at very small batch (decode). Marlin's INT4 × FP16 tensor-core path scales better as batch grows. Production stacks prioritise throughput across batch sizes, so the Marlin lineage won. LUT-GEMM survives in [[squeezellm]] (sensitivity-weighted k-means) and the llama.cpp k-quant family ([[gguf-k-quants]]), where the LUT structure naturally fits non-uniform codes.

---

## 6. Marlin — the production GEMM kernel for GPTQ-W4

[[marlin-kernel]] is what turns GPTQ from "research artifact" into "vLLM default". The headline: **a W4A16 GEMM that hits near-FP16 throughput at batch sizes up to ~32**, which is the production-serving range.

### The four engineering tricks

1. **Async global-to-shared copies** via `cp.async` on Ampere — weight loads happen in the shadow of dequant + MMA.
2. **Warp specialization** — some warps do memory movement, others do dequant + MMA; cooperate via mbarrier sync.
3. **Double-buffered shared memory** — while warp A consumes buffer 0, warp B fills buffer 1; the swap is a flag flip, no stall.
4. **Static weight pre-shuffle** — offline reorder of the 4-bit weights so each tensor-core thread has its operand in the right register slot after bit-shifts; eliminates per-call permute instructions.

### Why the critical batch range matters

At batch 1, W4 GEMM is purely memory-bound: 4× weight compression → 4× speedup. As batch grows, FLOPs grow linearly but weight bytes stay constant — at some batch the GEMM crosses into compute-bound. Pre-Marlin kernels (ExLlama, AutoGPTQ's CUDA kernel) couldn't keep the tensor cores fed past that crossover; their FLOPS/s collapsed.

Marlin's warp-specialized pipeline keeps both memory and compute units saturated through batch ~32, which is the range that matters for production serving (a single A100 serves ~32 concurrent decoding requests).

### Quantization format consumed

Marlin consumes the **GPTQ packed-int4** format with **`group_size = 128`** (the de facto standard since the GPTQ paper):

- Per-group scale and zero-point in FP16.
- Weights packed `8 × INT4` per 32-bit word.
- The static pre-shuffle changes in-memory layout but preserves bit content — any GPTQ-quantized checkpoint can be re-packed for Marlin offline.

### End-to-end performance

| Batch | FP16 cuBLAS | AutoGPTQ INT4 kernel | ExLlama | **Marlin** |
|---|---|---|---|---|
| 1 | 1.0× | 3.2× | 3.4× | **3.6×** |
| 4 | 1.0× | 2.1× | 2.8× | **3.5×** |
| 16 | 1.0× | 1.2× | 1.4× | **3.2×** |
| 32 | 1.0× | 0.9× | 1.0× | **2.8×** |
| 64 | 1.0× | 0.7× | 0.7× | **1.6×** |

Marlin maintains > 2.5× over FP16 through batch 32; competitors collapse past batch 8. Llama-70B end-to-end serving speedup on A100: **~2.8×**.

### Integration

- vLLM: `--quantization gptq_marlin` or `awq_marlin` automatically rewrites a GPTQ/AWQ checkpoint into the Marlin layout.
- Also exposed through NeuralMagic deepsparse, SparseML, and (community port) TensorRT-LLM.
- Hopper-native successor: [[machete-kernel]], covered in ch-19, uses TMA + WGMMA.

---

## 7. The Frantar-Alistarh lineage as one chart

```text
                       OBD (LeCun 1989)
                              │
                       OBS (Hassibi 1993)
                              │
                  ┌───────────┼───────────────┐
                  ▼           ▼               ▼
                AdaRound    OBC (Frantar 2022)
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
              SparseGPT                     GPTQ (Frantar 2022)
                                              │
                                  ┌───────────┼───────────────┐
                                  ▼           ▼               ▼
                               AutoGPTQ    Marlin          SpQR/QuIP
                              (loader)   (kernel)        (extensions)
```

The right column is the W4 weight-only LLM PTQ lineage. The left column is the sparsification lineage (same authors, same engine). Both descend from the two-page Hassibi 1993 paper.

The ZeroQuant family is parallel and independent (Microsoft, not IST-Austria) — different intellectual lineage (granularity + LKD instead of OBS), different but compatible result. In production, GPTQ-W4 (IST-Austria line) + Marlin (IST-Austria kernel) + AWQ-style scaling (MIT line, covered in ch-09) form the canonical 2026 W4A16 stack.

---

## 8. Practitioner's recipe table

| Goal | Algorithm | Knobs | Notes |
|---|---|---|---|
| **W4A16 deployment, ≥7B model** | GPTQ | `group_size=128`, `actorder=True`, `percdamp=0.01` | Default 2026 recipe |
| **W4A16 + better activation handling** | GPTQ + AWQ scale | grid-search α | See ch-09 |
| **W8A8 with no calibration** | ZeroQuant + per-token A | dynamic per-token scale | Easy, no LKD needed |
| **W4A8 with quality recovery** | ZeroQuant + LKD | `n_lkd_samples=512`, `lr=5e-7` | Adds ~30 min |
| **W4A8 on H100** | ZeroQuant-FP (FP4 + FP8) | per-tensor scale, power-of-2 | Beats INT by 0.3-0.5 ppl |
| **W3A16 (aggressive)** | GPTQ + [[quip]] rotation | group_size=128 + LDLQ rounding | See ch-13 |
| **Sub-2-bit** | [[aqlm]] | M codebooks, K vectors | See ch-14 |
| **Production GEMM kernel** | Marlin (Ampere) / Machete (Hopper) | `--quantization gptq_marlin` | See ch-19 |

---

## Connections and what's next

- **Forward to [[awq]] (ch-09)** — per-channel activation-aware scaling that often beats GPTQ at W4. Frequently combined with GPTQ in production (`awq_marlin`).
- **Forward to [[omniquant]] (ch-10)** — block-wise reconstruction with learnable equivalent transformations. Different optimisation, same calibration recipe.
- **Forward to [[spqr]] (ch-11)** — extension that preserves outlier weights in FP16; uses GPTQ as the base quantizer.
- **Forward to [[quip]] (ch-13)** — rotation preprocessing that makes GPTQ work at W3 and below.
- **Forward to [[machete-kernel]] (ch-19)** — Hopper-native W4A16 GEMM with TMA + WGMMA.
- **Forward to [[nvfp4-training]] / [[mxfp4-pretraining]] (ch-17)** — native FP4/FP8 hardware that validates ZeroQuant-FP's empirical claim at training scale.
- **Back to [[obc]] (ch-05)** — the BERT-scale predecessor; GPTQ is OBC + Cholesky + lazy block update.
- **Back to [[obs-obd]] (ch-05)** — the two-formula mathematical foundation.
- **Back to [[llm-int8]] (ch-07)** — the W8 weight + activation predecessor; GPTQ is the weight-only W4 successor.

## Further reading

- [[gptq]] — Frantar et al. 2022, the ICLR 2023 paper.
- [[obc]] — Frantar et al. 2022, the BERT-scale predecessor.
- [[zeroquant]] — Yao et al. 2022, the granularity-focused alternative.
- [[zeroquant-v2]] — Yao et al. 2023, LoRC + sensitivity study.
- [[zeroquant-fp]] — Wu et al. 2023, the FP4/FP8 verdict.
- [[nuqmm]] — Park et al. 2022, the LUT-GEMM kernel.
- [[marlin-kernel]] — Frantar et al. 2024, the production W4A16 GEMM.
- AutoGPTQ ([[autogptq]]) — the most-used GPTQ loader / packing library.
