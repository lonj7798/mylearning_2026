<!-- scope: BiBERT — fully binary BERT with bi-attention and direction-matching distillation
     deps: bnn, xnor-net, q8bert, q-bert
     see-also: bitnet, bitnet-b158
-->

# BiBERT: Accurate Fully Binarized BERT
- **Core Insight:** Naive binary BERT loses ~20 GLUE points because (1) attention scores rebinarize-to-zero after Softmax destroys information, and (2) cross-entropy distillation from a fp teacher does not match per-direction predictions; BiBERT fixes both via bi-attention (binarize before Softmax, not after) and direction-matching distillation (DMD) that aligns each output direction independently.
- **Guideline:** For a 1-bit transformer, binarize the attention QKᵀ score matrix (not the post-Softmax probability), and use DMD loss `Σ_i ‖fp_teacher_i / ‖fp_teacher‖ − bi_student_i / ‖bi_student‖‖²` per output position to preserve directional info.
- **Authors:** Haotong Qin, Yifu Ding, Mingyuan Zhang, Qinghua Yan, Aishan Liu, Qingqing Dang, Ziwei Liu, Xianglong Liu
- **Year:** 2022 (ICLR)
- **URL:** https://arxiv.org/abs/2203.06390
- **Relevant topics:** binary BERT, bi-attention, distillation, 1-bit transformer

## Abstract
BiBERT pushes BERT to fully binary weights AND activations (1-bit everywhere). Two architecture-aware fixes recover most of the loss: bi-attention (binarize the QKᵀ score logits before Softmax instead of after — the post-Softmax distribution is too sharp for sign() to preserve information) and direction-matching distillation (DMD) which aligns the student's per-position output direction to the fp teacher's, not just the magnitudes. On GLUE, BiBERT loses ~13 points vs FP BERT (much better than the ~25 drop of naive binary BERT), with 56× model compression and 31× CPU inference speedup.

## Key Contributions
- Bi-attention: binarize attention score (pre-Softmax) not attention probability (post-Softmax).
- Direction-Matching Distillation: per-position cosine alignment loss.
- Full 1-bit weight + 1-bit activation BERT-Base.
- 56× model compression vs FP BERT.
- Demonstrates 1-bit transformer feasibility years before BitNet.

## Key Figures/Tables to Study
- **Figure 2** — pre-Softmax score distribution vs post-Softmax probability distribution; visualises why bi-attention works.
- **Table 2** — GLUE: BiBERT vs naive binary BERT vs Q8BERT.

## Technical Details

### Standard binary weight + activation
`W_b = sign(W),  A_b = sign(A)`
With XNOR + popcount matmul (see [[bnn]]).

### Bi-attention (the key architectural fix)
Standard attention: `P = Softmax(QKᵀ/√d) ∈ (0,1)^{n×n}, output = P·V`.
Naive binarisation: `sign(P)` — Softmax already pushes P close to 0 except at the argmax position; sign(P) ≈ {0,0,...,0,1,0,...} loses all soft-attention information.

Bi-attention: binarize the **pre-Softmax score** S = QKᵀ/√d:
`S_b = sign(S) ∈ {−1, +1}^{n×n}`
Then apply a re-introduced Bool-Softmax that re-normalises but stays in binary representation:
`P_b = bool_softmax(S_b)`
Or more concretely, output computed via:
`output_i = (Σ_j 1[S_b_{ij} = +1] · V_j) / count(S_b_{i,:} = +1)`
i.e. average of V over positions where the binary score is +1.

### Direction-Matching Distillation (DMD)
KL distillation aligns per-class probabilities. For binary BERT it fails because the student's binary outputs have very different magnitudes from the teacher's continuous outputs. DMD instead aligns directions:
`L_DMD = Σ_i (1 − cos(fp_teacher_i, bi_student_i))`
`     = Σ_i (1 − ⟨t_i, s_i⟩ / (‖t_i‖ · ‖s_i‖))`
per output position i. Magnitude differences absorbed by the cosine normalisation.

### Combined loss
`L = L_task(student) + α · L_DMD(teacher, student)`
α typically 1.0 throughout fine-tuning.

### Training
- Initialise from fp BERT fine-tuned on the task.
- 5–10 epochs binary QAT with STE.
- Weights binarized via sign(real-valued shadow); activations via sign() inline.
- Real-valued shadow weights clipped to [-1, 1] after each step (BNN-style).

### Empirical effect
- BERT-Base GLUE-avg: FP 84.6 → naive binary 58.4 → **BiBERT 71.7** (Δ = -12.9).
- Model size: 418 MB → 7.4 MB (56× compression).
- CPU inference: 31× faster on AVX2 (XNOR + popcount kernel).

## Connections
- [[bnn]] — binary CNN training framework BiBERT inherits.
- [[xnor-net]] — per-channel scaling that BiBERT uses for weights.
- [[q8bert]] — 8-bit baseline; BiBERT is the 1-bit extreme of the same trajectory.
- [[q-bert]] — mixed-precision (≥2-bit) BERT; BiBERT goes fully 1-bit.
- [[bitnet]] — modern LLM-era 1-bit transformer, trained from scratch (vs BiBERT's post-training binarisation).
- [[bitnet-b158]] — ternary {-1, 0, +1} relaxation that achieves better quality than strict binary.
