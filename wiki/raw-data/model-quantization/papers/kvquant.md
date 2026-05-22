<!-- scope: KVQuant — sub-4-bit KV cache via pre-RoPE K quant, per-channel non-uniform, dense-and-sparse outlier isolation
     deps: [[squeezellm]], [[spqr]]
     see-also: [[kivi]], [[gear]], [[wkvquant]]
-->

# KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
- **Core Insight:** Sub-4-bit KV cache is achievable only with four jointly-applied techniques — (1) per-channel K (matching KIVI), (2) quantize K *before* RoPE rotation (the rotation entangles channels and destroys per-channel structure), (3) sensitivity-weighted *non-uniform* per-layer datatypes (a few quantile-fit bins per K channel and per V token), and (4) dense-and-sparse decomposition keeping the top ~1% outlier elements in FP16.
- **Guideline:** For ≤3-bit KV cache enabling million-token contexts, use KVQuant: quantize K before RoPE, use a per-channel non-uniform code, isolate the 1% outliers as FP16 sparse; achieves <0.1 PPL degradation at 3-bit on Wikitext-2.
- **Authors:** Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami
- **Year:** 2024 (NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2401.18079
- **Relevant topics:** ultra-low-bit KV cache, pre-RoPE K quant, non-uniform quant, dense-and-sparse, million-token context

## Abstract
KVQuant pushes KV cache quantization to sub-4-bit (3-bit, 2-bit) via four techniques: (1) per-channel K quantization on the *pre-RoPE* representation, (2) sensitivity-weighted non-uniform datatypes per channel/token, (3) dense-and-sparse decomposition preserving the top ~1% of outlier elements in FP16, and (4) Q-norm quantization for the Query as well. Achieves <0.1 PPL degradation with 3-bit KV on Wikitext-2 and C4. Enables LLaMA-7B inference at 1M context on a single A100-80GB and 10M context across 8 GPUs, with custom CUDA kernels delivering ~1.7× speedups vs FP16 matvec.

## Key Contributions
- **Pre-RoPE quantization for K**: RoPE applies token-dependent rotation that mixes channels — quantizing K post-RoPE destroys the per-channel outlier structure KIVI exploits. KVQuant quantizes the un-rotated K and applies RoPE on the dequantized values.
- **Non-uniform per-channel codes**: each K channel learns its own quantile-fit 2/3-bit code (similar to SqueezeLLM's non-uniform weight quant but applied to KV).
- **Dense-and-sparse decomposition**: 1% of elements (the outliers) kept in a sparse FP16 buffer; the rest in low-bit dense. Borrows from SpQR.
- **Q-norm quantization** for the Query side so the attention dot-product is computed entirely in low-bit.
- Custom CUDA kernels demonstrating 10M-token contexts on 8 GPUs.

## Key Figures/Tables to Study
- **Figure 2:** K cache distribution *before* vs *after* RoPE — pre-RoPE is per-channel structured, post-RoPE is mixed.
- **Figure 5:** Dense-and-sparse decomposition diagram — top-1% elements split off to FP16.
- **Table 3:** PPL on Wikitext-2 at 2/3/4-bit KV — KVQuant vs KIVI vs uniform per-token vs FP16.

## Technical Details

### Pre-RoPE K quantization
Standard RoPE: K_t = RoPE(W_k · x_t), where RoPE rotates pairs of channels (i, i+d/2) by token-dependent angle θ_t.
KVQuant stores W_k · x_t (the pre-RoPE K) in INT2/3, applies RoPE *on dequant* at attention time.
Cost: one extra dequant + RoPE step per attention call, fully fused into the attention kernel.
Why: pre-RoPE K has a stationary per-channel outlier distribution; post-RoPE distribution drifts because adjacent channels get rotated into each other by token-dependent angles.

### Per-channel non-uniform code
For each channel c, compute calibration histogram and fit a k-means codebook of size 2^B (B=2 or 3). Store the codebook (negligible amortised cost) and per-element index. Equivalent to SqueezeLLM's non-uniform weight quant but applied per K channel.

V uses per-token uniform quant (consistent with KIVI's finding that V is token-wise).

### Dense-and-sparse decomposition
Identify the top-1% (by absolute value) of pre-RoPE K and per-token V elements per layer; store them in a sparse FP16 vector (index + value). Dense path uses the non-uniform code at 2/3-bit. At attention time, dense path computes most of the dot product; sparse FP16 contributions are added as a correction.

Borrowed from [[spqr]] (which did this for weights).

### Q-norm quantization
The Query is also quantized so the QK^T dot-product happens in low-bit arithmetic. Per-token symmetric INT4/INT8.

### Effective bit budget at 2-bit
- Dense: 2 bits/element.
- Sparse: 1% × (24 bits index + 16 bits FP16) ≈ 0.4 bits/element amortised.
- Total ≈ 2.4 bits/element.

### Context length math
For LLaMA-7B at FP16: KV cache size per token = 2 × 32 layers × 32 heads × 128 dim × 2 bytes = 512 KB/token. 1M tokens = 512 GB. At 2-bit KVQuant: 64 KB/token → 64 GB for 1M tokens — fits in one A100-80GB.

## Connections
- Sibling KV-quant: [[kivi]] (per-channel K, per-token V at INT2 but post-RoPE), [[gear]] (low-rank residual), [[wkvquant]] (joint W+KV).
- Dense-and-sparse weight ancestor: [[spqr]].
- Non-uniform weight ancestor: [[squeezellm]].
- KV-cache compression survey: [[kv-cache-survey]].
- Per-channel vs per-token analytical: [[per-channel-vs-per-token-kv]].
