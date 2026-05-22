<!-- chapter: ch-02
     track: math-foundations
     title: Numerical Formats Reference — FP / INT / Block-Scaled
     sources: [[ieee-754]], [[bf16]], [[fp16]], [[fp8-e4m3]], [[fp8-e5m2]], [[fp4-e2m1]], [[nf4]], [[mx-formats]], [[nvfp4]], [[int4]], [[int8]], [[bitnet-w158]]
     figures: figures/fp-bit-layouts.html
-->

# Chapter 2 — Numerical Formats Reference: FP / INT / Block-Scaled

> **Core insight.** Every modern quantization format is one of three shapes: a **floating-point** code that is a discrete log-compander (FP32 → BF16 → FP16 → FP8 → FP6 → FP4), a **fixed-point integer** code that is a uniform Bennett quantizer (INT8 → INT4 → ternary), or a **block-scaled** layout (MXFP / NVFP / NF / k-quant) that pairs a tiny per-element format with a shared scale to recover dynamic range. The format zoo looks chaotic — it's six parameters varying along three axes.
>
> **Guideline.** When picking a format, fix three numbers first: **exponent bits** (= dynamic range; FP32 = 8, BF16 = 8, FP16 = 5, E4M3 = 4, E2M1 = 2, INT = 0), **mantissa bits** (= relative precision within an exponent bin), and **scale granularity** (per-tensor / per-channel / per-group / per-block). For sub-8-bit, you cannot omit the scale; for sub-4-bit, you cannot omit the block.

---

## Why this chapter exists

There is no single "best format." A weight that lives well in NF4 dies in INT4; a gradient that lives in E5M2 saturates in E4M3; an activation that quantizes cleanly per-channel collapses per-tensor. The right format is determined by (a) the shape of the source distribution, (b) the operation that consumes the tensor downstream, and (c) the hardware that runs the kernel. To make those decisions, you need fluency in the bit layout, the rounding mode, the exponent range, and the scaling overhead of each format on the table.

This chapter assembles every number you will need to compare formats in the subsequent algorithm chapters. Three things you should walk away with:

1. A mental picture of FP32, BF16, FP16, FP8 (E4M3 / E5M2), FP4 (E2M1), INT8, INT4 — bit layout, max-finite, min-positive-normal, machine epsilon.
2. The block-scale ladder: per-tensor → per-channel → per-group → MX → NVFP4 → NF4 → BitNet w1.58.
3. Why BF16 displaced FP16 in training, why FP8 splits into E4M3 + E5M2, and why FP4 requires *two-level* scaling (block + tensor) to be usable.

---

## 1. IEEE-754 — the substrate every float inherits

From [[ieee-754]], every IEEE-style number is

```
value = (−1)^s · (1 + m / 2^M) · 2^{e − bias}      (normal,   1 ≤ e ≤ 2^E − 2)
value = (−1)^s · (0 + m / 2^M) · 2^{1 − bias}      (subnormal, e = 0, m ≠ 0)
```

with reserved encodings for `±0` (`e = 0, m = 0`), `±∞` (`e = 2^E − 1, m = 0`), and NaN (`e = 2^E − 1, m ≠ 0`).

### binary32 (FP32) and binary16 (FP16)

| Format | Sign | Exp | Mantissa | Bias | Max finite | Min normal | ε |
|--------|------|-----|---------|------|------------|------------|-----|
| FP32  | 1 | 8 | 23 | 127 | ≈ 3.40e+38 | ≈ 1.18e−38 | ≈ 1.19e−7 |
| FP16  | 1 | 5 | 10 | 15  | 65504 | ≈ 6.10e−5 | ≈ 9.77e−4 |

FP16 has 8× *better* relative precision than BF16 but `10^{68}×` *less* dynamic range. This is why FP16 needed loss-scaling (multiply loss by `2^k` pre-backward, divide gradient by `2^k` post-backward) for transformer training: attention scores and gradients routinely exceed `65504`. The recipe was formalised by Micikevicius 2017 ([[fp16]]) and became the 2017–2020 standard.

> **Practical pitfall.** Subnormal numbers (gradual underflow) on most GPUs trigger microcode slow-paths (10–100× slowdown). Production training enables `FTZ` (flush-to-zero) and `DAZ` (denormals-are-zero), sacrificing the smooth underflow for throughput. Bennett's `Δ²/12` model assumes smooth underflow, so the prediction is slightly off near zero in production runs.

---

## 2. BF16 — the training default, and why

From [[bf16]], bfloat16 is **the top 16 bits of FP32**: same sign, same 8-bit exponent (bias 127), but the mantissa is truncated from 23 → 7 bits.

| Format | Sign | Exp | Mantissa | Max finite | Min normal | ε |
|--------|------|-----|---------|------------|------------|-----|
| BF16   | 1 | 8 | 7 | ≈ 3.39e+38 | ≈ 1.18e−38 | ≈ 7.81e−3 |

Trade-off: BF16 gives up ~8× relative precision compared to FP16 in exchange for `10^{33}×` more dynamic range. **For transformer training, the precision gap doesn't matter** (SGD/Adam noise dominates), **but the range gap does** (loss, attention scores, gradients overflow FP16 routinely). BF16 also makes conversion to/from FP32 a bit-shift — no exponent re-bias.

The standard mixed-precision recipe (2020–present):

- **FP32:** master weights, optimizer state, loss accumulator.
- **BF16:** forward activations, weights for matmul, gradients.
- **Matmul:** BF16 × BF16 → FP32 tensor-core accumulator → BF16 output.
- **No loss scaling. No overflow handling.**

Hardware support is everywhere since Ampere (A100, 2020), TPU v2+, MI100+, Gaudi 1+. **Rule of thumb:** if your training report doesn't specify a format, it used BF16.

---

## 3. FP8 — E4M3 forward, E5M2 backward

From [[fp8-e4m3]] and [[fp8-e5m2]], the 2022 NVIDIA/Arm/Intel joint proposal — standardised as OCP OFP8 in 2023 — defines a two-format FP8 system. Both formats are 8 bits; they differ in how the bits are split between exponent and mantissa, and in special-value handling.

| Format | Sign | Exp | Mantissa | Bias | Max finite | Min normal | ε | Specials |
|--------|------|-----|---------|------|------------|------------|-----|----------|
| E4M3 | 1 | 4 | 3 | 7 | **448** | 2^{−6} ≈ 0.0156 | 0.125 | NaN only |
| E5M2 | 1 | 5 | 2 | 15 | **57344** | 2^{−14} ≈ 6.1e−5 | 0.25 | ±∞, NaN (IEEE) |

Two things to memorise:

- **E4M3 sacrifices the IEEE `±∞` encoding** to gain 6 extra finite values per sign. Only `S.1111.111` is reserved (as NaN). This is the canonical "non-IEEE departure": with only 256 codes you cannot afford to spend 14 of them on infinities.
- **E5M2 is exactly FP16's exponent layout** (bias 15, max `e = 30`) with the mantissa cut from 10 → 2 bits. Same dynamic range as FP16, ~8× fewer values per exponent bin.

### Recipe in production training (Transformer Engine, DeepSeek V3 FP8)

- **Forward** (weights, activations): cast to E4M3 with per-tensor or per-block scale; tensor-core matmul `E4M3 × E4M3 → FP32 accum`.
- **Backward** (gradients): cast to E5M2; same matmul shape; wider range tolerates the heavy tail of gradient magnitudes (`10^{−6}` to `10^{+1}`) without per-tensor scale gymnastics.
- **Master weights / optimizer state:** stay in FP32 or BF16.
- **Per-tensor scale tracked online** via amax history; updated each step.
- **Stochastic rounding** ([[stochastic-rounding]], ch-01 §5) on the cast from FP32 master → FP8 storage preserves the expectation of `η · g`.

On H100 SXM: FP8 tensor cores at **1979 TFLOPS** = 2× BF16, 4× FP32.

---

## 4. FP4 — never alone, always block-scaled

From [[fp4-e2m1]], FP4 in the E2M1 layout has only **16 codes total**. The reconstruction set is

```
{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
```

(8 positive values including +0, mirrored across zero). Max finite 6.0, min positive normal 1.0, min subnormal 0.5, **ε = 0.5 (50% relative precision!)**. No `±∞`, no NaN — NaN handling is delegated to the block scale.

Raw FP4 covers `[−6, +6]` with 16 levels. This is hopeless for LLM weights (typical range `[−1, +1]`) or activations (`[−10, +10]`). **Never use FP4 with a single per-tensor scale.** Two block-scaled deployments:

- **MXFP4:** E8M0 (power-of-2-only) scale per 32 elements → 4 + 8/32 = **4.25 effective bits/element**.
- **NVFP4:** FP8 E4M3 scale per 16 elements + FP32 per-tensor scale → 4 + 8/16 = **4.5 effective bits/element**.

NVFP4 wins ~0.5 PPL on Llama-7B at the cost of 0.25 extra bits. We will return to both in §6.

---

## 5. INT8 and INT4 — the uniform-quantizer workhorses

From [[int8]] and [[int4]], the integer formats are uniform Bennett quantizers. The general affine map is

```
x_q  = clip(round(x / s) + z,  q_min,  q_max)
x_hat = s · (x_q − z)
```

with **scale** `s > 0` (controls step `Δ`) and **zero-point** `z ∈ ℤ` (allows asymmetric ranges).

### INT8 — symmetric weights, asymmetric activations

Signed INT8: range `[−128, +127]`, 256 levels. Symmetric weights (`z = 0`) → cheaper matmul (no `s_w · z_w · x_q` cross-term). Asymmetric activations (post-ReLU, post-GELU positive) → use `z ≠ 0` to put the whole UINT8 range on the active side. Modern hardware throughput: cuBLAS / Marlin INT8 GEMM at **2× BF16** on Hopper.

The breakdown at LLM scale: at 6.7B+ parameters, ~0.1% of activation channels carry 100× the bulk magnitude. Per-tensor INT8 either *clips* outliers (massive per-channel error) or *includes* them (Δ blows up; bulk drowns in noise). Neither works → motivation for [[llm-int8]] (mixed-precision outlier path) and [[smoothquant]] (migrate outliers to weights). Ch-07 and ch-09.

### INT4 — group-128 is the default

Signed INT4: range `[−8, +7]`, 16 levels. Per-tensor INT4 collapses on LLMs (drops 10+ PPL on Llama-7B). The standard fix is **group-wise scaling**:

```
for each group g of size G ∈ {64, 128} within a weight row:
    s_g = max(|W[g]|) / 7              # per-group FP16 scale
    W_q[g] = clip(round(W[g] / s_g), -7, +7)
    W_hat[g] = s_g · W_q[g]
```

Effective bits per weight: `4 + 16/G`. For `G = 128` this is **4.125 bits**. Pack 2 × INT4 per byte.

| Group | Effective bits | Llama-2-7B PPL gap | Compression |
|-------|---------------|---------------------|-------------|
| 32 | 4.5 | best | 3.55× |
| 64 | 4.25 | very good | 3.76× |
| **128** | **4.125** | **good (standard)** | **3.88×** |
| 256 | 4.0625 | noticeable | 3.94× |
| per-channel | 4.0 | severe | 4.0× |

Combine group-128 INT4 with [[gptq]] (Hessian-aware error compensation; ch-08) or [[awq]] (activation-aware scaling; ch-09) and you recover ~99% of FP16 perplexity at 4× weight compression. This is the **W4A16** deployment everywhere — AutoGPTQ, AutoAWQ, llama.cpp Q4_K, bitsandbytes 4-bit, vLLM W4A16.

---

## 6. Block-scaled formats — the modern frontier

The four canonical block-scaled layouts:

### NF4 (NormalFloat-4)

From [[nf4]], NF4 is a 16-level **non-uniform** code with reconstruction values placed at equally-spaced quantiles of the standard normal CDF, normalized so `|max| = 1`:

```
[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```

This is essentially **Lloyd-Max for `N(0,1)` at `N = 16`** (ch-01 §4). LLM weights post per-block absmax normalization are well-modelled as `N(0,1)` → NF4 captures Gish-Pierce density `p^{1/3}` near-optimally. Block size 64, with **double quantization** (the 256 inner scales are themselves quantized to FP8 with an outer FP32 scale per 256 inner). Effective bits: `4 + 8/64 + 32/(64·256) ≈ 4.127`. **Free 0.3–0.5 PPL** over INT4 on Llama-class weights. Default of bitsandbytes 4-bit and QLoRA (ch-12).

### OCP MX (Microscaling) — the cross-vendor standard

From [[mx-formats]], an MX number is a `(block-scale, element-data)` pair where the block-scale `X` is an **8-bit E8M0 exponent-only format** (`X = 2^{e − 127}`, no sign, no mantissa) shared by 32 consecutive elements. Six element formats are standardised: MXFP8-E4M3, MXFP8-E5M2, MXFP6-E3M2, MXFP6-E2M3, MXFP4-E2M1, MXINT8.

```
MX vector = (X, P_0, P_1, ..., P_31)
value(P_i) = 2^{e_X − 127} · element_value(P_i)
```

Effective bits = element_bits + 8/32 = element_bits + 0.25. Quality:

- MXFP8 ≈ BF16 (trivially deployable).
- MXFP6 ≈ FP16 (~0.05 PPL gap).
- MXFP4 ≈ INT4-AWQ quality without rotation (~0.1–0.3 PPL gap from BF16).

Vendor-neutral: NVIDIA Blackwell, AMD MI355X, Intel Gaudi 3.

### NVFP4 — proprietary, denser scale

From [[nvfp4]], NVFP4 is NVIDIA's competing FP4 layout with two key differences from MXFP4:

- Block size **16** (vs 32) → halves the "outlier blast radius."
- Block scale is **FP8 E4M3** (vs E8M0) → mantissa-bearing scale, can express non-power-of-2 magnitudes like 1.5 (E8M0 must round to 1 or 2).

Plus an additional **FP32 per-tensor scale** on top. Total bit budget: `4 + 8/16 + negligible ≈ 4.5 effective bits/element`. Quality on Llama-7B at 4-bit: NVFP4 ~0.15 PPL gap from BF16; MXFP4 ~0.3 PPL gap. NVFP4 is the recommended FP4 format on Blackwell for both inference and training (ch-17).

### BitNet W1.58 — ternary, 1.58 bits per weight

From [[bitnet-w158]], BitNet b1.58 restricts every weight to `{-1, 0, +1}` — `log₂(3) ≈ 1.58 bits/weight`. Forward matmul becomes pure adds and subtracts:

```
(X_q · Wᵀ)_im = Σ_{n: W[m,n] = +1} X_q[i,n]  −  Σ_{n: W[m,n] = −1} X_q[i,n]
```

Activations stay at INT8 per-token. Storage: 5 ternary values per byte (`3^5 = 243 ≤ 256`) or simpler 2-bit/weight with one wasted code. **Requires end-to-end pretraining from scratch with the ternary constraint** (BitLinear + STE; ch-04, ch-16). Naive PTQ to ternary collapses. Empirical claim: parity with FP16 from ~3B parameters onward.

---

## 7. The format-selection cheat-sheet

```
By dynamic range (exp bits):
  FP32 / BF16 (8 bits)  ←  loss / gradient accumulators
  FP16 / E5M2 (5 bits)  ←  gradients with loss scaling / bwd-side FP8
  E4M3 (4 bits)         ←  fwd activations and weights with per-tensor scale
  E2M1 (2 bits)         ←  needs block scale (MXFP4 / NVFP4); 16 codes total
  INT  (0 bits)         ←  needs scale + (optional) zero-point per granularity

By scale granularity:
  per-tensor     →  trivial; only works for benign distributions
  per-channel    →  +1% accuracy at negligible cost (weights)
  per-group-128  →  W4A16 PTQ default; INT4 + group-128 standard
  block-32 MX    →  MXFP4 / MXFP6 / MXFP8 / MXINT8
  block-16 NVFP4 →  Blackwell-native FP4 with FP8 block scale + FP32 tensor scale
  per-token      →  activations in LLM.int8 / SmoothQuant

By production use:
  Pretraining:   BF16 (default), FP8 (DeepSeek V3, Llama 3 FP8), NVFP4 (Blackwell)
  Inference W4:  INT4 group-128 + GPTQ/AWQ; NF4 in QLoRA; MXFP4/NVFP4 on Blackwell
  Inference W4A8: QServe; needs rotation or smoothing
  Inference W4A4: QuaRot / SpinQuant + NVFP4 / INT4
  Edge / mobile: INT8 per-channel; llama.cpp Q4_K / Q5_K / Q8_0
  Sub-2-bit:     BitNet w1.58 (requires pretraining); AQLM (additive VQ)
```

> **Practical pitfall.** Every "bits per weight" number you read elsewhere should add the scale overhead. NF4 is *not* 4 bits; it's 4.127. MXFP4 is 4.25. NVFP4 is 4.5. INT4 group-128 is 4.125. The differences look small but compound into 10–15% of the storage budget at 4-bit and decide which model fits on which GPU.

---

## Connections and what's next

- **[[uniform-quantization-noise]] / ch-01** — Bennett's `Δ²/12` predicts the per-element MSE of every INT and uniformly-quantized FP format in this chapter.
- **[[information-theoretic-bounds]] / ch-01** — Gish-Pierce `p^{1/3}` density is what NF4 instantiates for the Gaussian weight prior; what NVFP4 / MXFP4 approximate via the log-spaced FP element code.
- **[[stochastic-rounding]] / ch-01** — SR on the cast to FP8 / FP4 / MX is what makes low-precision training converge.
- **[[lloyd-max-quantizer]] / ch-03** — formalizes the centroid/NN iteration; NF4 is a quantile-spaced shortcut.
- **[[smoothquant]] / ch-09, [[awq]] / ch-09** — fix the activation-outlier problem so INT8 / INT4 per-tensor activation quant becomes usable.
- **[[gptq]] / ch-08** — adds Hessian-aware error compensation to INT4 group-128; near-FP16 PPL.
- **[[deepseek-v3-fp8]] / ch-17** — first frontier-scale FP8 pretraining; per-block FP8 with FP32 accumulator every 4 WGMMA.

## Further reading

- [[ieee-754]] — the IEEE substrate (binary32 / binary16).
- [[bf16]] — Google Brain bfloat16 (2018) — the training default.
- [[fp16]] — IEEE binary16 + Micikevicius 2017 mixed-precision recipe.
- [[fp8-e4m3]] / [[fp8-e5m2]] — joint NVIDIA/Arm/Intel FP8 proposal (Micikevicius 2022); OCP OFP8 standard.
- [[fp4-e2m1]] / [[mx-formats]] / [[nvfp4]] — OCP MX and Blackwell-native NVFP4.
- [[int4]] / [[int8]] — uniform integer formats; the W4A16 / INT8 default.
- [[nf4]] — Dettmers QLoRA 2023; quantile-based 4-bit code.
- [[bitnet-w158]] — Ma et al. 2024; ternary weights, 1.58 bits.

## Companion visualization

**[figures/fp-bit-layouts.html](figures/fp-bit-layouts.html)** — side-by-side bit-layout viewer for FP32 / BF16 / FP16 / E4M3 / E5M2 / E2M1 / INT8 / INT4 / NF4 / MXFP4 / NVFP4, with a slider to perturb input values and watch the encoded bits change. Useful for building intuition about exponent vs mantissa trade-offs. *(Optional — skip on first read.)*
