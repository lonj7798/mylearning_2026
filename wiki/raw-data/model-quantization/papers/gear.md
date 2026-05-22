<!-- scope: GEAR — KV cache compression combining ultra-low-bit quantization, low-rank residual approximation, and sparse outlier correction
     deps: [[kivi]], [[kvquant]]
     see-also: [[gear]], [[wkvquant]]
-->

# GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM
- **Core Insight:** Single-technique KV quantization (uniform low-bit, or pure outlier isolation, or pure low-rank) leaves accuracy on the table — the residual after aggressive low-bit quant has a low-rank structure that captures most of the systematic error and can be cheaply represented with a rank-r SVD addendum, while the remaining sparse outliers are handled by a small COO buffer.
- **Guideline:** For near-lossless 4-bit KV cache, use GEAR = INT4 (ultra-low-bit dense) + rank-2 SVD (low-rank residual matrix L = AB^T capturing systematic error) + sparse COO buffer for the 1% remaining outliers; 2.38× throughput, 2.29× peak memory reduction.
- **Authors:** Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, Tuo Zhao
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2403.05527
- **Relevant topics:** KV cache compression, low-rank residual, error compensation, sparse outliers

## Abstract
GEAR is a hybrid KV-cache compression recipe that synergistically combines three techniques: (1) ultra-low precision uniform quantization for the bulk of entries, (2) low-rank matrix approximation of the *quantization residual* to capture systematic error, and (3) sparse correction for outlier entries. The interplay is essential — quantization alone loses signal that accumulates across autoregressive decoding; the low-rank residual recovers that signal cheaply; the sparse path handles the few entries low-rank can't represent. Achieves near-lossless 4-bit KV with 2.38× throughput improvement and 2.29× peak memory reduction.

## Key Contributions
- Recognises that the post-quantization residual `R = KV − dequant(quant(KV))` has approximately low-rank structure across the (n_tokens × head_dim) matrix per head.
- GEAR formula: `KV ≈ Quant_b(KV) + A · B^T + S`, where Quant_b is uniform b-bit, A ∈ ℝ^{T×r}, B ∈ ℝ^{d×r} (small rank r=2–4), S is sparse outlier matrix.
- Streaming SVD implementation that updates A, B incrementally as new tokens arrive (no full SVD recomputation).
- Demonstrates that the three components are non-redundant: ablating any one materially degrades accuracy.

## Key Figures/Tables to Study
- **Figure 2:** Quantization residual visualization — clear low-rank pattern across the (tokens × channels) residual matrix.
- **Figure 4:** GEAR pipeline diagram — quant + low-rank + sparse.
- **Table 3:** PPL on WikiText-2 / accuracy on long-context tasks — GEAR-4bit vs KIVI-4bit vs FP16.

## Technical Details

### The decomposition
For a per-head KV matrix M ∈ ℝ^{T × d} (T = #tokens, d = head_dim):
`M ≈ Q + L + S`
- **Q = dequant(quant_b(M))**: uniform b-bit quantization (b=4 typical).
- **L = A · B^T**: low-rank residual matrix, rank r=2–4. A ∈ ℝ^{T×r}, B ∈ ℝ^{d×r}.
- **S**: sparse correction holding the top-K largest residual entries that L+Q still misses, K ≈ 1% of T·d.

### Computing L (truncated SVD)
After choosing Q, compute residual R = M − Q. SVD R = U Σ V^T; keep top-r singular vectors and values: `L = U_r Σ_r V_r^T`. Set A = U_r √Σ_r, B = V_r √Σ_r.

For streaming inference, update A, B incrementally: when a new token arrives, append to M, requantize the chunk to Q, compute residual delta r_t = m_t − q_t, project onto current B and orthogonalise. A periodic full refresh (every ~256 tokens) prevents drift.

### Sparse S
After Q + L, scan residual M − Q − L; pick top-K elements by absolute value (K = ⌈0.01 T d⌉); store as (token_idx, channel_idx, FP16 value) tuples.

### Bit budget
At b=4, r=2, K=1%T·d:
- Q: 4 bits per element.
- L: 2(T+d)·16 bits / (Td) = 32(T+d)/(Td) ≈ 32/T + 32/d → for T=2048, d=128: ~0.27 bits/element.
- S: 1% × 32 bits ≈ 0.32 bits/element.
- Total ≈ 4.6 bits/element.

### Attention math
`attn_logits = Q · (Q_K + L_K + S_K)^T / √d`
Distributed-additive: compute three matmuls and sum. Q-path uses INT4 GEMM; L-path is two small FP16 matmuls (rank r×d, very cheap); S-path is sparse SpMV.

### Streaming SVD complexity
Full O(T d r) per refresh, amortised over the refresh interval.

## Connections
- Sibling KV-quant trio: [[kivi]] (asymmetric per-channel K / per-token V), [[kvquant]] (non-uniform + dense-and-sparse), [[gear]] (this — low-rank residual).
- Sparse-outlier ancestor: [[spqr]].
- Low-rank residual lineage: LQ-LoRA-style decomposition ([[lq-lora]]).
- KV-cache compression survey: [[kv-cache-survey]].
