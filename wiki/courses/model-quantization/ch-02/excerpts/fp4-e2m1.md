---
chapter: ch-02
course: model-quantization
phase: read
excerpt_of: "FP4 E2M1 (OCP MX element / NVIDIA Blackwell native)"
source_url: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
created_at: "2026-05-21"
raw_data_source: [[raw-data/fp4-e2m1]]
---

# Excerpt: FP4 E2M1 — the 16-code minimal float

**Authors:** OCP Microscaling Working Group (2023); NVIDIA Blackwell team (2024).
**Year:** 2023 (OCP MX); 2024 (Blackwell B100/B200 hardware).
**URLs:** OCP MX spec — see source_url; Blackwell announcement https://developer.nvidia.com/blog/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing/

---

## The one-box: 4 bits, 16 codes

```
[ s | e_1 e_0 | m_0 ]
   1     2 (bias 1)     1
```

- Normal (`1 ≤ e ≤ 3`): `value = (−1)^s · (1 + m/2) · 2^{e − 1}`.
- Subnormal (`e = 0, m = 1`): `value = ±0.5`.
- Zero (`e = 0, m = 0`): `±0`.
- **No `±∞`, no NaN** at element level (NaN handled by the block scale in MXFP4 / NVFP4).

---

## Full 16-code table

| s | e | m | Value |
|---|---|---|---|
| 0 | 00 | 0 | +0 |
| 0 | 00 | 1 | +0.5 (subnormal) |
| 0 | 01 | 0 | +1 |
| 0 | 01 | 1 | +1.5 |
| 0 | 10 | 0 | +2 |
| 0 | 10 | 1 | +3 |
| 0 | 11 | 0 | +4 |
| 0 | 11 | 1 | +6 |
| 1 | ... | ... | negatives of the above |

**Reconstruction set:** `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.

---

## Key constants

- Max finite: **6.0** (= `1.5 · 2^{3−1}`).
- Min positive normal: 1.0.
- Min positive subnormal: 0.5.
- Machine epsilon `ε = 2^{−1} = 0.5` (**50% relative precision!**).
- Dynamic range (normal): 6× (= `2^{2.585}`); including subnormal: 12×.

---

## Why FP4 needs block scaling

Raw FP4 covers `[−6, +6]` with only 16 levels — totally inadequate for LLM weights (typical range `[−1, +1]`) or activations (`[−10, +10]`). Two block-scaled deployments:

- **MXFP4** ([[mx-formats]]): shared E8M0 power-of-2 scale per 32 elements → `4 + 8/32 = 4.25` effective bits/element.
- **NVFP4** ([[excerpts/nvfp4]]): two-level — FP8 E4M3 scale per 16 elements + FP32 per-tensor scale → `4 + 8/16 + negligible ≈ 4.5` effective bits/element.

**Never use FP4 with a single per-tensor scale.** Naive per-tensor FP4 loses 3+ perplexity points on any LLM.

---

## FP4 vs INT4 (same bit budget, different shape)

| | FP4 E2M1 | INT4 |
|---|---|---|
| Codes | 16 (log-spaced) | 16 (uniform) |
| Max | 6 | 7 |
| Step | exponentially varying (0.5–2.0) | uniform (1.0) |
| Dynamic range / step | 12× | 14× |
| Best for | log-magnitude weights/activations | uniform-bounded data |

FP4 is better for sources with wide magnitude variation (LLM weights post-normalization, attention scores). INT4 is better for sources already in a bounded range.

---

## Hardware

- **NVIDIA Blackwell B100 / B200**: ~9 PFLOPS FP4 (dense), ~18 PFLOPS sparse.
- **AMD MI355X (2025)**: announced FP4 support.
- **Pre-Blackwell**: software FP4 via dequant-to-BF16 + BF16 tensor-core matmul.

---

## Use cases

- **Inference**: weight-only FP4 (W4A16) is plug-and-play with negligible quality loss on Llama-class models.
- **W4A4 inference**: requires rotation ([[quarot]]) or smoothing ([[smoothquant]]) for activations.
- **FP4 training**: NVFP4 native pretraining demonstrated ([[nvfp4-training]]); requires stochastic rounding and FP32 master weights.
- **FP4 KV-cache**: viable with per-block scale and per-channel K / per-token V partitioning (KVQuant-style).

Pack layout: 2 × FP4 per byte (high nibble = element 0, low nibble = element 1). Pack/unpack is a single nibble shift.

---

## Connections

- [[excerpts/ieee-754]] — E2M1 inherits sign/exp/mantissa structure (minus special values).
- [[mx-formats]] — MXFP4 = E2M1 element + E8M0 block scale per 32.
- [[excerpts/nvfp4]] — NVIDIA production FP4: E2M1 + FP8 block scale + FP32 tensor scale.
- [[int4]] — uniform 4-bit alternative.
- [[ch-02]] — parent synthesis.
