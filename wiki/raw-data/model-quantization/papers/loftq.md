<!-- scope: LoftQ — joint quantization + LoRA initialization via alternating minimization
     deps: [[qlora]], [[zeroquant-v2]]
     see-also: [[qa-lora]], [[peqa]], [[lq-lora]]
-->

# LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models
- **Core Insight:** Naively initializing LoRA adapters at zero on a pre-quantized base means the starting point is the quantized weight `Q(W)` — strictly worse than `W` — and gradient descent has to climb out of the quantization-error hole; instead, jointly choose Q and (A, B) so that `Q + B A ≈ W`, giving an initialization that *cancels* the quantization error and a strictly better-than-QLoRA starting loss.
- **Guideline:** Whenever you'd use QLoRA, initialize with LoftQ instead — alternate `Q ← quant(W − B A)` and `(A, B) ← rank-r SVD(W − Q)` for ~5 iterations; then run standard QLoRA fine-tuning from there. Particularly important at 2-bit and mixed 2/4-bit.
- **Authors:** Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, Tuo Zhao
- **Year:** 2023 (ICLR 2024)
- **URL:** https://arxiv.org/abs/2310.08659
- **Relevant topics:** joint quant + LoRA init, alternating minimization, sub-4-bit fine-tuning

## Abstract
QLoRA fine-tuning starts from `Q(W)` for the frozen base and zero-initialized LoRA adapters — meaning the initial forward pass already differs from FP by the full quantization error. LoftQ instead jointly solves for (Q, A, B) such that `Q + B A^⊤ ≈ W`, using alternating minimization between a closed-form RTN step and a truncated-SVD step. The resulting frozen base + non-zero LoRA initialization starts essentially at FP performance, and downstream fine-tuning converges to higher final accuracy. The gap is especially large in **2-bit and 2/4-bit mixed-precision regimes** where QLoRA's initial loss is catastrophic.

## Key Contributions
- **Joint quant + LoRA initialization** via alternating minimization:
  - Q-step: `Q ← Q_b(W − B A^⊤)` (b-bit quantization, RTN or NF4).
  - LoRA-step: `B, A ← TruncSVD_r(W − Q)`.
- Provably reduces initial Frobenius error `||W − (Q + B A^⊤)||_F` monotonically across iterations.
- Demonstrates large gains over QLoRA at 2-bit and 2/4-bit mixed precision (where pure QLoRA fails to converge).
- Compatible with arbitrary quantizers (RTN, NF4, GPTQ) — slots in front of QLoRA.

## Key Figures/Tables to Study
- **Figure 1:** the initialization-error curve — LoftQ vs QLoRA over alternating iterations.
- **Table 4:** 2-bit LLaMA-2-7B MMLU/GSM8K — QLoRA collapses, LoftQ recovers most of FP.

## Technical Details

### The decomposition objective
For each layer's weight `W ∈ R^{d_out × d_in}`, solve:
```
min_{Q, A, B}  || W − (Q + B A^⊤) ||_F²
s.t.    Q ∈ Q_b   (b-bit quantizable values)
        A ∈ R^{d_in × r}, B ∈ R^{d_out × r}
```

### Alternating minimization
Initialize Q^(0) = Q_b(W), A^(0) = 0, B^(0) = 0. Then for t = 1, …, T (T ≈ 5):
```
Step 1 (LoRA fit):   (B^(t), A^(t)) ← TruncSVD_r(W − Q^(t-1))
                      i.e. compute SVD W − Q^(t-1) = U Σ V^⊤,
                      keep top-r singular vectors:
                      B^(t) = U_{:, 1:r} √(Σ_{1:r}),  A^(t) = V_{:, 1:r} √(Σ_{1:r})

Step 2 (re-quantize): Q^(t) ← Q_b( W − B^(t) A^(t)^⊤ )
```
Both steps reduce the objective; convergence in 4–5 iterations.

### After initialization
Use `Q^(T)` as the frozen 4-bit (or 2-bit) base, `(B^(T), A^(T))` as LoRA init, then run standard QLoRA fine-tuning (BF16 adapters, gradients through dequantized base).

### Bit budget
- Q: b bits/weight (b ∈ {2, 3, 4}), group_size 64–128.
- A, B: BF16, total `r · (d_in + d_out) · 16` bits per layer.
- Effective bits/weight ≈ b + 16r(d_in + d_out)/(d_in · d_out) — for r=8 on 4096×4096 this is ~b + 0.06.

### Mixed-precision variant
LoftQ supports per-layer different b's (e.g. 2-bit for attention, 4-bit for FFN) by adjusting the Q-step per layer. The same alternating algorithm applies.

### Hyperparameters
| Knob | Value |
|------|-------|
| LoftQ iterations T | 5 |
| Quantizer | RTN / NF4 / GPTQ (interchangeable) |
| Bits | 2, 3, 4 (per layer) |
| LoRA rank r | 8, 16, 64 |
| Group size | 64 |
| Downstream FT | QLoRA-style, AdamW |

## Connections
- Direct comparison: [[qlora]] (zero LoRA init).
- Same low-rank-residual idea applied to PTQ alone (no fine-tune): [[zeroquant-v2]] (LoRC).
- Merge-friendly cousin: [[qa-lora]].
- Scale-only PEFT alternative: [[peqa]].
- 2024 successor with explicit low-rank quant decomposition: [[lq-lora]].
