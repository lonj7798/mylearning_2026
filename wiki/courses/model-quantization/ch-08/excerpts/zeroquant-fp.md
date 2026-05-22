---
chapter: ch-08
course: model-quantization
phase: read
excerpt_of: "ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats (Wu, Yao, He 2023)"
source_url: https://arxiv.org/abs/2307.09782
arxiv: 2307.09782
created_at: "2026-05-21"
---

# Excerpt: ZeroQuant-FP — the FP4/FP8 verdict

**Authors:** Xiaoxia Wu, Zhewei Yao, Yuxiong He
**Year:** July 2023
**Raw-data source:** [[raw-data/papers/zeroquant-fp]]

---

## The empirical claim (the whole paper)

For LLM PTQ at low bit-widths, floating-point formats consistently beat integer formats of the same width because the wider dynamic range of FP absorbs outliers that INT clips.

```text
FP8 activations  >  INT8 activations  (gap widens above ~1B parameters)
FP4 weights      ≥  INT4 weights      (gap widens at 30B+, FP wins)
```

This is the result that justifies the 2024–2026 hardware push: [[nvfp4-training]], [[mxfp4-pretraining]], Blackwell tensor cores.

---

## The FP formats used

- **E4M3** (FP8, 4 exp + 3 mantissa, bias = 7): weights and activations at FP8. Range ≈ [−448, 448]; one NaN; no infinities. See [[fp8-e4m3]].
- **E5M2** (FP8, 5 exp + 2 mantissa): gradients in FP8 training (not used here).
- **E2M1** (FP4, 2 exp + 1 mantissa): weights at FP4. The 16 representable values:

```math
\{\pm 0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\}
```

Note the geometric spacing at high magnitudes ({2, 3, 4, 6}) — this is what gives FP4 its outlier-absorption advantage over INT4 (which has linear {±0..±7}).

---

## The quantization rule

For each weight tensor `W`:

```math
s \;=\; \max(|W|) / \text{max\_repr(format)}
```

```math
\hat{W} \;=\; \text{nearest\_fp}(W / s) \cdot s
```

`nearest_fp(·)` rounds to the format's representable set. For E2M1 this is a 16-level non-uniform code; for INT4 it's a 16-level uniform code.

For activations: same rule per-token (dynamic) at FP8, `max_repr(E4M3) = 448`.

---

## The W4A8-FP recipe

| Knob | Value |
|---|---|
| Weight format | FP4-E2M1 |
| Activation format | FP8-E4M3 |
| Weight scale | per-tensor (power-of-2 constrained) |
| Activation scale | per-token, dynamic |
| LoRC rank | 8 |
| Hardware target | H100 (FP8 tensor cores) |

### The two scaling constraints

For W4-E2M1 × A8-E4M3, the per-tensor scale product `s_w · s_a` must align with what the accumulator can absorb without overflow:

1. `s_w` constrained to a **power-of-two** so the rescale is a bit-shift.
2. `s_w · s_a` bounded so the partial sum stays in FP16/FP32 accumulator.

Both cost < 0.05 ppl vs unconstrained.

---

## Empirical: FP vs INT side-by-side

| Format combo | LLaMA-7B PPL | LLaMA-30B PPL |
|---|---|---|
| FP16 baseline | 5.68 | 4.10 |
| INT8 act + INT4 weight | 6.31 | 4.95 |
| FP8 act + INT4 weight | 5.93 | 4.32 |
| **FP8 act + FP4 weight (W4A8-FP)** | **5.89** | **4.29** |

FP wins by ~0.4 ppl on 7B, ~0.5 ppl on 30B. **The gap widens with model scale** — larger models have larger outliers, so the FP dynamic-range advantage compounds.

---

## Why FP beats INT at low bits (the mechanism)

Two effects:

1. **Dynamic range absorbs outliers.** INT4 represents `{−8, ..., +7}` linearly; one outlier at 100× normal magnitude forces a scale that crushes 99% of values into 1–2 levels. FP4's `{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` has 1.5 bits of mantissa devoted to the small-magnitude region (high density near 0) and 1 bit of exponent for headroom up to 6.
2. **Multiplicative rather than additive precision.** FP gives uniform *relative* error (constant ratio); INT gives uniform *absolute* error. For LLM weights which span 4–5 orders of magnitude, relative-error preservation matters more than absolute.

The mathematical intuition is exactly Lloyd-Max's optimal density `ρ(x) ∝ p(x)^{1/3}` (covered in [[ch-01]]): non-uniform spacing matches non-uniform distributions. LLM weight distributions are heavy-tailed → FP-style geometric spacing wins.

---

## LoRC integration

After FP quantization, residual `E = W − Ŵ_FP` gets a rank-r SVD correction (as in [[zeroquant-v2]]). r=8 typical; recovers ~0.2–0.4 ppl on 7B models.

---

## Why this paper matters strategically

ZeroQuant-FP is the empirical justification for the entire 2024–2026 native FP4/FP8 hardware push:

- **NVIDIA Hopper (H100)** ships FP8 tensor cores → ZeroQuant-FP's W4A8-FP is deployable today.
- **NVIDIA Blackwell** ships FP4 + FP8 tensor cores → NVFP4 inference deploys natively.
- **MXFP4 / OCP MX** is the cross-vendor standard for block-scaled FP4 → see [[ch-16]] microscaling formats.

If integers had been competitive with FP at 4 bits, none of this hardware development would have happened.

---

## Common pitfalls

- **Comparing INT4 to FP4 with wrong scale grain.** Both should use the same group_size; comparing per-tensor INT4 to per-channel FP4 is unfair. ZeroQuant-FP uses per-tensor for both.
- **Ignoring the power-of-2 scale constraint.** Without it, the rescale becomes an FP multiply — defeats the FP-native hardware advantage.
- **Assuming FP4 wins everywhere.** Below 1B, INT and FP are comparable; FP's advantage emerges at scale. The gap is also smaller for activation quantization than weight quantization.

---

## Connections

- [[excerpts/zeroquant]] — the INT predecessor.
- [[excerpts/gptq]] — the weight-only INT alternative.
- [[ch-08]] — parent synthesis.
- [[ch-02]] — FP format references (E4M3, E5M2, E2M1).
- [[ch-17]] — native FP4/FP8 training (NVFP4, MXFP4) that validates this paper's claim at training scale.
- [[ch-16]] — microscaling formats (MXFP4) that generalise per-tensor FP scales to per-block.
