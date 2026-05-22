<!-- scope: OneBit — 1-bit weight LLMs via Sign-Value-Independent Decomposition (SVID) with sign matrix + scaling vectors
     deps: [[bitnet]], [[bitnet-b158]]
     see-also: [[bitnet-a48]], [[era-of-1bit-llms]]
-->

# OneBit: Towards Extremely Low-bit Large Language Models
- **Core Insight:** Rather than ternary `{−1, 0, +1}` with one global scale (BitNet b1.58), a true 1-bit weight matrix `Sign(W) ∈ {−1, +1}^{m×n}` paired with two separate per-row and per-column scaling vectors `a ∈ ℝ^m, b ∈ ℝ^n` (the Sign-Value-Independent Decomposition, SVID) recovers most of the precision of a full FP matrix at 1 bit/weight while remaining hardware-friendly (sign multiplication = XOR + popcount).
- **Guideline:** When pushing weights all the way to 1 bit (not 1.58), use OneBit-style SVID — decompose `W ≈ diag(a) · S · diag(b)` where S is sign-only, then fine-tune (a, b) and the sign matrix on calibration data; reaches 81%+ of FP performance at 1 bit/weight on LLaMA models.
- **Authors:** Yuzhuang Xu, Xu Han, Zonghan Yang, Shuo Wang, Qingfu Zhu, Zhiyuan Liu, Weidong Liu, Wanxiang Che
- **Year:** 2024 (NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2402.11295
- **Relevant topics:** 1-bit weight, SVID decomposition, sign matrix, scaling vectors

## Abstract
OneBit reaches 1-bit (binary) weight LLMs without sacrificing usable accuracy by introducing Sign-Value-Independent Decomposition (SVID): each weight matrix is factored as `W ≈ diag(a) · S · diag(b)` with S ∈ {−1, +1} a binary sign matrix and a, b ∈ ℝ continuous per-row / per-column scales. A matrix-decomposition initialization (SVD-style) is followed by quantization-aware fine-tuning. Achieves at least 81% of the non-quantized performance on LLaMA at 1-bit weights, dramatically below prior 1-bit LLM methods.

## Key Contributions
- SVID: explicit factorization that disentangles sign (1-bit) from magnitude (two FP vectors), preserving the rank-1 dominant structure that single-scale binarization loses.
- Decomposition-based initialization: derive (a, S, b) from a smart SVD of W rather than random initialization — critical for QAT convergence.
- Comprehensive evaluation on LLaMA-7B/13B/30B showing competitive performance at 1-bit weights.
- Demonstrates that even 1-bit weights — not just 1.58-bit ternary — are viable for LLMs given the right decomposition.

## Key Figures/Tables to Study
- **Figure 1:** SVID factorization diagram.
- **Figure 3:** Decomposition-based initialization vs random — orders-of-magnitude initial loss difference.
- **Table 2:** LLaMA 1-bit results — OneBit vs BitNet b1.58 (which is 1.58-bit, not 1-bit) vs prior 1-bit baselines (PB-LLM, BiLLM).

## Technical Details

### Sign-Value-Independent Decomposition
For each weight matrix W ∈ ℝ^{m×n}:
`W ≈ diag(a) · S · diag(b)`
- S ∈ {−1, +1}^{m×n}: the sign matrix, 1 bit per element.
- a ∈ ℝ^m: per-row scale (FP16).
- b ∈ ℝ^n: per-column scale (FP16).

Effective bits/weight = 1 + 16/n + 16/m ≈ 1 bit (m, n large).

### Decomposition initialization
1. Initialise from the per-column absolute mean: `b_j = (1/m) Σ_i |W_{ij}|`.
2. Initialise `a_i = (1/n) Σ_j |W_{ij}| / b_j`.
3. Initialise `S_{ij} = sign(W_{ij})`.
4. Refine via alternating SVD-style update: re-derive a, b given S and vice versa, minimizing `||W − diag(a) S diag(b)||_F`.

Without this init, randomly assigned S leads to massive initial loss that STE fine-tuning cannot recover.

### Quantization-aware fine-tuning
After SVID init, fine-tune the model with:
- S updated by STE through `sign()`,
- a, b updated by standard SGD (FP16 parameters),
- Loss = standard causal-LM cross-entropy on a small corpus, or distillation from the FP teacher.

### Inference math
`y = W x ≈ diag(a) (S (diag(b) x))`
1. Compute `x' = diag(b) · x` (n FP multiplies).
2. Compute `y' = S · x'`: each y'_i = Σ_j S_{ij} · x'_j. With S ∈ {−1, +1}, each MAC is a sign-flipped add — implementable as XOR-popcount on bit-packed S.
3. Scale: `y = diag(a) · y'` (m FP multiplies).
Total: O(mn) sign-MACs + O(m + n) FP MACs (amortised to negligible).

### Why SVID beats single-scale binarization
A single scalar scale s with `W ≈ s · sign(W)` discards all per-row / per-column magnitude variation. SVID recovers two rank-1 vectors of magnitude info per matrix — captures the dominant low-rank component of W's magnitude pattern.

### Comparison to BitNet b1.58
- BitNet b1.58: ternary `{−1, 0, +1}` weights at 1.58 bits, single per-tensor scale. Trained from scratch.
- OneBit: binary `{−1, +1}` weights at ~1 bit, two scaling vectors. Reachable by fine-tuning an FP model.

## Connections
- Direct comparators: [[bitnet]] (1-bit, scratch), [[bitnet-b158]] (1.58-bit, scratch).
- Sub-1.58-bit alternatives: BiLLM, PB-LLM (binary residual approximation).
- Sign-binary classical roots: [[bnn]], [[xnor-net]].
- Survey-style consolidation: [[era-of-1bit-llms]].
- Sub-2-bit competitors (vector codebook): [[aqlm]], [[quip-sharp]], [[vptq]].
