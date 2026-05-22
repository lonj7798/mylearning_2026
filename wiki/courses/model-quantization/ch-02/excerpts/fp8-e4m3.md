---
chapter: ch-02
course: model-quantization
phase: read
excerpt_of: "FP8 Formats for Deep Learning — E4M3 (Micikevicius et al. 2022; OCP OFP8 2023)"
source_url: https://arxiv.org/abs/2209.05433
created_at: "2026-05-21"
raw_data_source: [[raw-data/fp8-e4m3]]
---

# Excerpt: FP8 E4M3 — the forward-pass FP8

**Authors:** Paulius Micikevicius et al. (joint NVIDIA + Arm + Intel proposal, 2022); standardized as OCP OFP8 in 2023.
**Year:** 2022 (paper); 2023 (OFP8 standard); 2022 (H100 hardware).
**Venue:** arXiv 2209.05433 → ICML / OCP standard.
**URLs:** see source_url; OFP8 spec https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1

---

## The one-box layout

```
[ s | e_3 e_2 e_1 e_0 | m_2 m_1 m_0 ]
   1        4 (bias 7)        3
```

- Normal (`1 ≤ e ≤ 15`): `value = (−1)^s · (1 + m/8) · 2^{e − 7}`.
- **Exception at `e = 15, m = 7` (S.1111.111):** reserved as NaN, not finite.
- **All other `e = 15` encodings are finite** — this is the non-IEEE departure.
- Subnormal (`e = 0, m ≠ 0`): `value = (−1)^s · (m/8) · 2^{−6}`.
- `±0`: `e = 0, m = 0`.
- **No `±∞` encoding.** Only NaN reserved.

---

## Key constants

- Smallest positive normal: `2^{−6}` ≈ 1.5625e−2.
- Smallest positive subnormal: `2^{−9}` ≈ 1.9531e−3.
- Largest finite: `1.75 · 2^{8}` = **448** (= `S.1111.110`).
- Machine epsilon `ε = 2^{−3}` = 0.125 (12.5%).
- Total finite values: 254 (with `±0` and NaN distinguished).

---

## Why drop `±∞` in E4M3?

With only 256 codes, sacrificing 14 (`S.1111.???`) for `±∞` + NaN is expensive. The OFP8 spec reserves only the two `S.1111.111` encodings for NaN and lets all other `S.1111.???` be finite, giving 6 extra finite values per sign. **This is the canonical "non-IEEE" departure of E4M3.**

---

## Per-tensor / per-block scale (mandatory)

Raw E4M3 covers `[−448, +448]` with only 254 finite values; LLM activations can reach 10³ or more. A scale `s ∈ FP32` (per-tensor) or per-block (in MX / NVFP4) is multiplied in:

```
x_quant = round_to_e4m3(x_fp32 / s)
x_recon = x_quant · s
```

Scale chosen by `amax` calibration — map per-tensor max-abs to the E4M3 max (448). Dynamic-range-aware scale calibration is the core of NVIDIA Transformer Engine.

---

## E4M3 vs E5M2 — when to use which

| | E4M3 | E5M2 |
|---|---|---|
| Exponent bits | 4 | 5 |
| Mantissa bits | 3 | 2 |
| Bias | 7 | 15 |
| Max finite | 448 | 57344 |
| Min positive normal | 0.0156 | 6.1e−5 |
| ε | 0.125 (12.5%) | 0.25 (25%) |
| Special | NaN only | ±∞, NaN (IEEE-like) |

**E4M3 → finer precision (12% vs 25% relative), narrower range (`~10^{4.5}` vs `~10^{9}`).** Use E4M3 for tensors with bounded magnitude after normalization (weights, activations); use E5M2 for tensors with heavy tails (gradients).

---

## Production usage

- **Transformer Engine** (NVIDIA): automatic E4M3 / E5M2 management with amax history.
- **DeepSeek V3 FP8** ([[deepseek-v3-fp8]]): first frontier-scale (671B) FP8 native training; E4M3 for most ops, per-block FP8 scale, FP32 accumulator every 4 WGMMA.
- **MX format** ([[mx-formats]]): MXFP8 = E4M3 element + E8M0 block scale per 32.
- **NVFP4** ([[nvfp4]]): FP8 E4M3 is the *block-scale format* for the 16-element NVFP4 blocks.

---

## Hardware throughput (H100 SXM)

- FP8 tensor cores: **~1979 TFLOPS** (E4M3 or E5M2) with FP32 accumulator.
- 2× BF16 (989 TFLOPS), 4× FP32 (495 TFLOPS).

---

## Connections

- [[excerpts/ieee-754]] — FP8 inherits sign/exp/mantissa structure but sacrifices `±∞`.
- [[excerpts/fp8-e5m2]] — gradient-side partner; IEEE-like `±∞` / NaN semantics.
- [[mx-formats]] — block-scaled FP8 (MXFP8 with E4M3 element).
- [[excerpts/nvfp4]] — uses FP8 E4M3 as the per-16-element block scale.
- [[excerpts/stochastic-rounding]] — SR on FP32 → FP8 cast preserves expectation of weight update.
- [[ch-02]] — parent synthesis.
