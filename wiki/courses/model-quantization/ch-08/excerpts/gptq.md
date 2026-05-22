---
chapter: ch-08
course: model-quantization
phase: read
excerpt_of: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar, Ashkboos, Hoefler, Alistarh 2022)"
source_url: https://arxiv.org/abs/2210.17323
arxiv: 2210.17323
created_at: "2026-05-21"
---

# Excerpt: GPTQ — OBC at LLM scale

**Authors:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
**Year:** October 2022 (ICLR 2023)
**Raw-data source:** [[raw-data/papers/gptq]]

---

## The layer-local objective

```math
\hat{W}^{*} \;=\; \arg\min_{\hat{W} \in \mathcal{Q}} \; \bigl\| W X - \hat{W} X \bigr\|_F^2
```

`W ∈ ℝ^{d_out × d_in}`, calibration `X ∈ ℝ^{d_in × N}` (typical N = 128 × 2048). `𝒬` = representable quantized weights (per-row or per-group scaled INT-k).

Hessian:

```math
H \;=\; 2\, X X^\top \;\in\; \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}
```

Shared across all output rows of W (rows are independent given X) → compute once, reuse `d_out` times. Forward-only, no backward.

---

## The OBS column update (inherited from OBC)

For column `q` of weight row `w`:

```text
w_q   :=  quant(w_q)                                              # round to int4 grid
δ_F   := −(w_q − quant(w_q)) / [H⁻¹]_{qq}  ·  [H⁻¹]_{q, F}        # F = remaining cols
w_F   :=  w_F + δ_F
```

Literally OBC, literally OBS ([[obs-obd]]). The novelty is what makes this affordable at d = 16384.

---

## Engineering trick #1 — Cholesky reformulation

Instead of recomputing `H⁻¹` after each column, pre-compute the upper-triangular Cholesky factor `R` of `H⁻¹` once:

```math
H = L\, L^\top \;\Rightarrow\; H^{-1} = L^{-\top} L^{-1} = R^\top R
```

The quantities needed (`[H⁻¹]_qq` and `[H⁻¹]_{q, F}`) read directly from `R`. After processing column `q`, `R` gets a rank-1 downdate. Numerically more stable than maintaining `H⁻¹` directly.

Per-layer cost: `O(d³)` for the Cholesky + `d · O(d²)` per-column updates = **`O(d³)` total**.

---

## Engineering trick #2 — Lazy block update (the real GPTQ contribution)

Process columns in **blocks of B = 128**:

```text
For block b in [0, B, 2B, ..., d_in]:
    For column q in block b:
        quantize w_q; compensate ONLY columns within this block
    At block boundary:
        Apply accumulated correction to ALL remaining columns (b+B, ..., d_in)
        as a single dense GEMM.
```

The per-column updates are memory-bound (touch every remaining column). Batching them turns the inner loop into a **compute-bound dense matmul** — GPU-friendly. **~10× speedup** over per-column OBC.

---

## Damping (`percdamp`)

H is often near-singular (calibration `X` rank ≤ N; for short calibration, H is rank-deficient). Damp before Cholesky:

```math
H \;\leftarrow\; H + \lambda I, \qquad \lambda \;=\; \text{percdamp} \cdot \text{mean}(\text{diag}(H))
```

Standard `percdamp = 0.01`. Raise to 0.1 if Cholesky fails (NaN); lower to 0.001 only with very large calibration sets.

---

## Activation ordering (`actorder`)

Quantize columns in **descending `diag(H)` order**. High-activation-energy columns get the cleanest rounding (most slack from `[H⁻¹]_qq`); their errors are absorbed by lower-energy columns processed later.

Empirical effect: **~0.1–0.3 perplexity** improvement at 4-bit on Llama. Free, one sort.

---

## Grouping (`group_size`)

Per-output-channel scale is too aggressive at 4-bit. **`group_size = 128`** stores one (scale, zero_point) per 128 consecutive input dims per output row.

- Storage overhead at INT4: 16-bit scale + 4-bit zero ÷ 128 ≈ **0.156 bits/element**.
- Accuracy gain: **2–5 perplexity points** at INT3/INT4 vs per-channel.

---

## The standard recipe

| Knob | Value |
|---|---|
| Bits | 4 (also 3, 2 with [[quip]] preprocessing) |
| `group_size` | 128 |
| `actorder` | True |
| `percdamp` | 0.01 |
| Block size B | 128 |
| Calibration samples | 128 sequences × 2048 tokens |
| Calibration source | C4 / WikiText / domain text |
| Symmetric / asymmetric | asymmetric for INT4 |

---

## Empirical effect (the headline numbers)

| Model | FP16 PPL (WikiText-2) | GPTQ W4 PPL | GPTQ W3 PPL | Wall clock |
|---|---|---|---|---|
| OPT-175B | 8.34 | 8.37 (Δ +0.03) | 8.68 (Δ +0.34) | ~4 GPU-hours |
| BLOOM-176B | 8.11 | 8.21 (Δ +0.10) | 8.64 (Δ +0.53) | ~4 GPU-hours |
| OPT-66B | 9.34 | 9.55 | 9.99 | ~2 GPU-hours |
| OPT-13B | 10.13 | 10.31 | 11.61 | ~30 min |
| Llama-2-7B | 5.47 | 5.69 | 6.30 | ~10 min |

W4 GPTQ is within 0.1 ppl of FP16 at 175B scale, single-GPU job, no retraining. **Watershed for W4 LLM deployment.**

---

## Why GPTQ won the production stack

- No training data; calibration text suffices.
- No backward pass; forward-only Hessian.
- One-shot; no convergence diagnostics.
- Output format (packed INT4 + per-group FP16 scales) is exactly what [[marlin-kernel]] consumes.
- Numerically stable.

Cost: per-layer (not per-block) → doesn't see activation distribution shifts. Addressed by [[awq]] (activation-aware scaling) and [[omniquant]] (block reconstruction).

---

## Common pitfalls

- **Forgetting `actorder=True`.** Costs 0.1–0.3 ppl at no compute. Always on.
- **Per-channel instead of group_size=128 at INT4.** Costs 2–5 ppl. Always group at INT4.
- **`percdamp = 0`.** Cholesky fails with NaN; H is singular. Always damp.
- **Skipping `lm_head` quantization → forgetting to skip `embed_tokens`.** Both should typically be FP16 for quality.
- **Calibration set too small.** Below 32 sequences, the Hessian is rank-deficient (calibration rank ~ N · S, but useful rank smaller). 128 sequences × 2048 tokens is the empirical minimum for stable results.
- **Calibration domain mismatch.** Calibrating Llama on code data → quantized model worse on natural language. Use general-domain C4 / WikiText unless you specifically know the deployment domain.

---

## Connections

- [[excerpts/obc]] — the BERT-scale predecessor; GPTQ adds Cholesky reformulation + lazy block update.
- [[excerpts/obs-obd]] — the 1993 mathematical foundation.
- [[ch-08]] — parent synthesis.
- [[ch-09]] — [[awq]] is the activation-aware sibling, often combined.
- [[ch-19]] — [[marlin-kernel]] is the production GEMM that consumes GPTQ's output format.
