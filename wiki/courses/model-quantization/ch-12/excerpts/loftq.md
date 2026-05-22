---
chapter: ch-12
course: model-quantization
phase: read
excerpt_of: "LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models"
source_url: https://arxiv.org/abs/2310.08659
created_at: "2026-05-21"
---

# Excerpt: LoftQ — joint quant + LoRA initialization via alternating minimization

**Authors:** Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, Tuo Zhao
**Year:** 2023 (ICLR 2024)
**URL:** https://arxiv.org/abs/2310.08659
**Raw-data source:** [[raw-data/loftq]]

---

## The QLoRA initialization problem

QLoRA fine-tuning starts from:
- frozen base = `Q(W)` (quantization error built in).
- LoRA adapters = `A = 0, B = 0` (zero-initialised, by convention to not perturb the base).

**Initial forward pass already differs from FP by the full quantization error `||W - Q(W)||`.** At 4-bit on Gaussian weights this is small enough to recover. At 2-bit it is **catastrophic** — gradient descent cannot climb out of the hole, and QLoRA collapses.

---

## The fix — joint decomposition

Solve for `(Q, A, B)` such that `Q + B A^⊤ ≈ W`, simultaneously:

```math
\min_{Q,\ A,\ B}\ \big\lVert W - (Q + B A^\top) \big\rVert_F^2
\quad \text{s.t.}\quad Q \in \mathcal{Q}_b,\ A \in \mathbb{R}^{d_{in} \times r},\ B \in \mathbb{R}^{d_{out} \times r}
```

where `Q_b` is the b-bit quantizable values (RTN, NF4, or GPTQ — interchangeable).

---

## Alternating minimization (the algorithm)

```
Initialize Q^(0) = Q_b(W),  A^(0) = 0,  B^(0) = 0.
For t = 1, ..., T (T ≈ 5):
    # Step 1 — LoRA fit (truncated SVD of the residual)
    U, Σ, V^⊤ = SVD(W - Q^(t-1))
    B^(t) = U[:, :r] · √Σ[:r]
    A^(t) = V[:, :r] · √Σ[:r]

    # Step 2 — re-quantize (with the LoRA residual subtracted)
    Q^(t) = Q_b( W - B^(t) · A^(t)^⊤ )
```

Both steps monotonically reduce `||W - (Q + BA^⊤)||_F²`. Convergence in 4–5 iterations.

---

## After initialization

Use `Q^(T)` as the frozen 4-bit (or 2-bit) base, `(B^(T), A^(T))` as **non-zero LoRA init**. Run standard QLoRA fine-tuning from there: BF16 adapters, gradients through dequantized base.

**Key difference vs QLoRA:** the starting forward pass produces near-FP outputs (the LoRA cancels the quantization error). Downstream fine-tuning converges to higher final accuracy.

---

## Bit budget

- Q: b bits/weight (b ∈ {2, 3, 4}), group_size 64–128.
- A, B: BF16, total `r · (d_in + d_out) · 16` bits per layer.
- Effective bits/weight ≈ `b + 16r(d_in + d_out) / (d_in · d_out)`. For r=8 on 4096×4096 this is `b + 0.06`.

---

## Mixed-precision variant

LoftQ supports per-layer different b's (e.g. 2-bit for attention, 4-bit for FFN) by adjusting the Q-step per layer. Same alternating algorithm; per-layer bit-width is a free choice.

---

## Hyperparameters

| Knob | Value |
|---|---|
| LoftQ iterations T | 5 |
| Quantizer | RTN / NF4 / GPTQ (interchangeable) |
| Bits | 2, 3, 4 (per layer) |
| LoRA rank r | 8, 16, 64 |
| Group size | 64 |
| Downstream FT | QLoRA-style, AdamW |

---

## Empirical (Table 4 — 2-bit LLaMA-2-7B)

| Method | MMLU | GSM8K |
|---|---|---|
| Full FP | 45.3 | 14.0 |
| QLoRA 2-bit | **collapses** | **collapses** |
| **LoftQ 2-bit** | 41.1 | 10.2 |

At 2-bit, QLoRA's zero-init starts from a forward pass so different from FP that downstream fine-tuning never recovers. LoftQ's joint init starts near-FP and recovers most of the gap.

At 4-bit the LoftQ gain is smaller (QLoRA already works) but LoftQ still slightly wins (~0.2–0.5 MMLU).

---

## When to pick LoftQ over QLoRA

- **2-bit weight quantization:** mandatory (QLoRA collapses).
- **Mixed 2/4-bit precision:** LoftQ is the only PEFT that handles it cleanly.
- **4-bit:** small gain (~0.5 MMLU); use if the extra init cost (~minutes per layer) is acceptable.

---

## Connections

- Direct comparison: [[qlora]] (zero LoRA init).
- Same low-rank-residual idea applied to PTQ alone: [[zeroquant-v2]] (LoRC).
- Merge-friendly cousin: [[qa-lora]].
- Scale-only PEFT alternative: [[peqa]].
- 2024 successor with explicit low-rank decomposition: [[lq-lora]].
