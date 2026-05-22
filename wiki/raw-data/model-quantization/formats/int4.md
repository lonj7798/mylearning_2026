<!-- scope: INT4 packing layouts (W4A16, W4A8); group-wise scale; the modern weight-only PTQ default
     deps: [[int8]]
     see-also: [[gptq]], [[awq]], [[nf4]], [[marlin-kernel]]
-->

# INT4 (4-bit Integer Quantization)
- **Core Insight:** A 4-bit signed integer x_q ∈ [−8, +7] has only 16 reconstruction levels, requiring **fine-grained group-wise scaling** (one FP16 scale per group of 64 or 128 weights) plus careful packing (2 × INT4 = 1 byte) to recover quality — and once you accept the group-scale overhead, INT4 weight-only quantization (W4A16) is essentially "free" in quality for Llama-class LLMs.
- **Guideline:** Default INT4 deployment is **group-size 128, symmetric, per-channel scale, with GPTQ or AWQ calibration**; pair with a fused dequant + GEMM kernel (Marlin / Machete / Triton W4A16) to hit near-FP16 throughput on Ampere/Hopper hardware.
- **Authors:** Established as a deployment target in GPTQ (Frantar 2022) and bitsandbytes 4-bit (Dettmers); group-128 layout standardized by AutoGPTQ
- **Year:** 2022 (GPTQ + bitsandbytes 4-bit); 2023 (group-wise becomes canonical)
- **URL:** https://arxiv.org/abs/2210.17323 (GPTQ); https://github.com/PanQiWei/AutoGPTQ
- **Relevant topics:** INT4, group-wise scale, packing layout, W4A16, sub-8-bit PTQ

## Abstract
INT4 quantization represents weights with 4 bits per element, giving 16 reconstruction levels per group. Because the dynamic-range-per-step ratio of 16 levels is small (≤ 16× from min to max), naïve per-tensor or per-channel INT4 collapses on LLM weights. The standard fix is **group-wise scaling**: split each weight row into groups of 64 or 128 elements, store one FP16 scale (and optionally one INT4 zero-point) per group, and quantize within each group. With GPTQ-style Hessian-aware error compensation or AWQ-style activation-aware scaling, group-128 INT4 W4A16 recovers ~99% of FP16 perplexity at 4× weight compression. INT4 is the dominant *weight-only* deployment precision for LLM inference 2023–present (AutoGPTQ, AutoAWQ, llama.cpp Q4_K, bitsandbytes 4-bit, vLLM W4A16 via Marlin).

## Key Contributions
- 4× weight compression vs FP16; 2× over INT8.
- Group-wise scaling as the standard layout (group-64 / group-128).
- Pack 2 × INT4 per byte → trivial memory layout, 1-cycle unpack.
- Fused dequant + GEMM kernels (Marlin, Machete) hit near-FP16 throughput.
- Enables LLM serving on consumer GPUs (24 GB VRAM holds a 70B model at INT4).

## Key Figures/Tables to Study
- **Perplexity vs group size** for INT4 LLM PTQ — quality plateau at group-size ≤ 128.
- **Bit budget breakdown**: 4 (weight) + 16 (FP16 scale) / 128 (group) ≈ 4.125 effective bits/weight.

## Technical Details

### Bit representation
- **Signed INT4**: 4 bits, range [−8, +7]; two's complement.
- **Unsigned INT4** (UINT4): 4 bits, range [0, 15]; used for asymmetric with explicit zero-point.

### Quantization map (group-wise symmetric)
For each group g of size G (e.g. G = 128) within a weight row:
```
s_g = max(|W[g]|) / 7                       (per-group scale, FP16)
W_q[g] = clip(round(W[g] / s_g), −7, +7)    (INT4)
W_hat[g] = s_g · W_q[g]                      (dequant)
```

### Asymmetric variant (zero-point)
```
s_g = (max(W[g]) − min(W[g])) / 15
z_g = round(−min(W[g]) / s_g)               (UINT4 zero-point)
W_q[g] = clip(round(W[g] / s_g) + z_g, 0, 15)
W_hat[g] = s_g · (W_q[g] − z_g)
```

### Pack layout (2 × INT4 = 1 byte)
Two common conventions:
- **Interleaved (AutoGPTQ default)**: byte i contains W[2i+1] in high nibble, W[2i] in low nibble.
- **Sequential (llama.cpp q4_0)**: similar; varies by quant tier (q4_0, q4_1, q4_K).

For GPU GEMM kernels (Marlin, Machete), the pack layout is permuted to match tensor-core fragment loads — the on-disk layout is reshuffled by the quantizer.

### Bit budget with group scale
```
effective bits/weight = 4 + (sizeof(s_g) · 8) / G
                      = 4 + 16 / 128  ≈ 4.125    (FP16 scale, group-128)
                      = 4 + (16 + 4) / 128 ≈ 4.156 (with UINT4 zero-point)
```

### Group size tradeoff
| Group | Effective bits | Quality (Llama-2-7B PPL) | Memory savings vs FP16 |
|-------|---------------|--------------------------|-------------------------|
| 32 | 4.5 | best | 3.55× |
| 64 | 4.25 | very good | 3.76× |
| 128 | 4.125 | good (standard) | 3.88× |
| 256 | 4.0625 | noticeable degradation | 3.94× |
| per-channel (no group) | 4.0 | severe degradation | 4.0× |

### Calibration methods (which INT4 layout to use)
- **GPTQ** ([[gptq]]): one-shot Hessian-OBS column update; group-128 sym.
- **AWQ** ([[awq]]): activation-aware per-channel scaling + plain RTN INT4; group-128 sym.
- **HQQ** ([[hqq]]): half-quadratic optimization, data-free; group-64 asym.
- **NF4** ([[nf4]]): non-uniform 4-bit code tuned for Gaussian weights; used in QLoRA.
- **llama.cpp Q4_K**: superblock layout with mixed FP16 + 6-bit scales; group-32 effective.

### W4A16 vs W4A8 vs W4A4
- **W4A16**: weights INT4, activations BF16/FP16. Dominant deployment; activations stay in tensor-core precision.
- **W4A8**: weights INT4, activations INT8. Higher throughput, harder calibration; QServe / QUIK / Atom target this.
- **W4A4**: both 4-bit. Requires rotation ([[quarot]]) or learned equalization; aggressive, 2024+ research.

### Kernel: fused dequant + GEMM
```
for tile in output:
    load 4-bit weights as INT4 fragment
    load FP16/BF16 activations as fragment
    dequant: W_fp16 = scale[group] · int4_to_fp16(W_int4)
    accumulate: tile += W_fp16 @ A_fp16  (via tensor core)
```
Marlin ([[marlin-kernel]]) achieves ~90% of FP16 peak throughput on Ampere/Hopper.

### Failure modes
- **Per-tensor INT4** (no group): drops 10+ PPL on Llama-7B; never use.
- **Group too coarse** (≥ 256): outliers in one group blow up the entire group's scale.
- **Activations also INT4 without rotation**: severe quality loss; need [[quarot]] / [[spinquant]].
- **Calibration-set mismatch**: GPTQ calibrated on Wikitext but deployed on code → 0.5–1 PPL gap.

## Connections
- [[int8]] — same template at higher bit-width; INT4 needs group-scale, INT8 usually doesn't.
- [[gptq]] — the dominant INT4 calibration method.
- [[awq]] — activation-aware per-channel scaling + naive INT4.
- [[nf4]] — non-uniform 4-bit alternative; better for Gaussian weights.
- [[marlin-kernel]] — production W4A16 GEMM.
- [[quarot]] — rotation enables W4A4 deployment.
- [[llama-cpp-ggml]] — q4_K family of layouts.
