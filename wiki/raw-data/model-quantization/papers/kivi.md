<!-- scope: KIVI — asymmetric KV cache quant: K per-channel (outliers are channel-wise) + V per-token, INT2 viable
     deps: [[llm-int8]]
     see-also: [[kvquant]], [[gear]], [[wkvquant]]
-->

# KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache
- **Core Insight:** Empirically, the K cache has *channel-wise* outliers (a few channels are consistently 10–100× larger than others) while the V cache outliers are *token-wise* — so K should be quantized per-channel and V per-token, breaking the natural symmetry that all prior KV-quant work assumed.
- **Guideline:** For 2-bit KV cache quantization, use KIVI: K per-channel INT2 with group size 32 along the token axis, V per-token INT2 with group size 32 along the channel axis; tuning-free, drop-in, 2.6× peak-memory reduction and 2.35–3.47× throughput.
- **Authors:** Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu
- **Year:** 2024 (ICML 2024)
- **URL:** https://arxiv.org/abs/2402.02750
- **Relevant topics:** KV-cache quantization, per-channel K, per-token V, INT2, tuning-free

## Abstract
KIVI is the first KV-cache quantization scheme to recognize that K and V have *different* outlier structures and should be quantized along *different* axes. Empirically, K cache outliers are concentrated in a few persistent channels (channel-wise), while V cache shows no consistent channel pattern but token-by-token variation. KIVI applies per-channel quantization for K and per-token quantization for V, both at INT2, with no fine-tuning. Achieves 2.6× peak memory reduction, 4× larger batch size, and 2.35–3.47× throughput improvement on LLama, Falcon, Mistral.

## Key Contributions
- Diagnostic study showing that K and V have *different* outlier distributions in trained LLMs.
- Asymmetric quantization rule: K per-channel, V per-token, both with small grouping (32) along the orthogonal axis.
- INT2 quantization that is fully tuning-free (no calibration / fine-tuning needed).
- Hardware-friendly streaming implementation: tokens accumulate in a small FP16 buffer and are quantized in chunks of 32 to avoid recomputation of per-channel/per-token statistics.
- Open-source CUDA kernels integrating quantized KV into PagedAttention.

## Key Figures/Tables to Study
- **Figure 2:** K vs V cache distributions on LLaMA-2 — clear channel-wise pattern for K, token-wise for V. This is the load-bearing empirical finding.
- **Figure 4:** Streaming pipeline diagram — residual FP16 window + INT2 packed past.
- **Table 4:** PPL / accuracy at INT2 / INT4 — KIVI vs symmetric per-token, vs per-channel uniform, vs FP16.

## Technical Details

### The asymmetric quantization rule
For a KV tensor of shape `(n_tokens, n_heads, head_dim)`:
- **K cache:** quantize along the *token* axis but groups *channels*: each channel c has its own scale s_K[c] computed from the past tokens; INT2 representation per (token, channel). Group size 32 along token axis means stats refresh every 32 tokens.
- **V cache:** quantize along the *channel* axis but groups *tokens*: each token t has its own scale s_V[t]; INT2 per (token, channel). Group size 32 along channel axis.

### Why K is channel-wise outlier-concentrated
The K projection W_k inherits the residual-stream outlier channels from the input. Those channels are persistent across all tokens. If you quant K per-token, the scale chosen for token t is dominated by these outlier channels, crushing precision for the bulk channels. Quant per-channel uses a separate scale for each problem channel, so outlier channels get their own scale and bulk channels keep tight precision.

### Why V is token-wise
V is computed from the same hidden states but multiplied by attention weights at use time. Empirically V exhibits no consistent outlier channel pattern; token-wise variation dominates. Per-token quant fits this better.

### Streaming implementation
At decode time, new tokens append to the KV cache. KIVI maintains:
- A small FP16 residual buffer of the last < g=32 tokens.
- Quantized INT2 packed past for older tokens.
When the buffer fills to 32, the chunk is quantized (per-channel for K, per-token for V) using its own statistics and appended to the packed past.

### Attention math
`attn_logits = Q (K_int2 · s_K[c])^T / √d`
Standard fused-kernel pattern: dequantize K and V on-the-fly during the attention matmul, no materialization. K dequant uses one FP16 multiply per channel per token; V dequant one FP16 multiply per token per channel. Both cheap relative to the dot products.

### Bit budgets
- INT2 + group32 + FP16 scale: effective 2 + 16/32 = 2.5 bits per element.
- INT4 + group32 + FP16 scale: 4 + 16/32 = 4.5 bits per element.

### Accuracy
LLaMA-2-7B at INT2 KV: WikiText-2 PPL increases by <0.2 vs FP16 KV. LongBench tasks within 1 point of FP16.

## Connections
- Sibling KV-quant work with different focus: [[kvquant]] (ultra-low-bit, non-uniform + dense-and-sparse), [[gear]] (low-rank residual), [[wkvquant]] (joint W4 + KV4).
- Window-aware sibling: [[skvq]] (recent tokens at higher precision).
- Quality-adaptive sibling: [[qaq]].
- Outlier-channel observation parallels: [[llm-int8]] (residual stream outliers).
- KV-cache compression survey: [[kv-cache-survey]].
- Per-channel vs per-token analytical study: [[per-channel-vs-per-token-kv]].
