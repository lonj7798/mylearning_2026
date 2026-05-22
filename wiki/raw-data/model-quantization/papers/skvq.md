<!-- scope: SKVQ — sliding-window-aware KV cache quantization keeping recent tokens at high precision
     deps: [[kivi]], [[kvquant]]
     see-also: [[qaq]], [[gear]], [[per-channel-vs-per-token-kv]]
-->

# SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models
- **Core Insight:** Attention queries decay sharply across token distance — the most-recent tokens carry the bulk of attention mass — so a sliding window of recent tokens deserves high-precision KV while older tokens can be aggressively quantized; combined with channel reordering (sorting similar-magnitude channels together) this enables 2-bit K and 1.5-bit V with minimal accuracy loss.
- **Guideline:** For million-token contexts, use SKVQ = (1) channel reorder for similar-magnitude grouping, (2) clipped dynamic group-level quant, (3) sliding window of W recent tokens kept FP16 / INT4 while history goes INT2 K / INT1.5 V; up to 7× faster decoding on 7B models at 1M context on a single 80GB GPU.
- **Authors:** Haojie Duanmu, Zhihang Yuan, Xiuhong Li, Jiangfei Duan, Xingcheng Zhang, Dahua Lin
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2405.06219
- **Relevant topics:** sliding-window KV quant, channel reorder, recency-aware compression, ultra-low-bit KV

## Abstract
SKVQ proposes a sliding-window-aware KV cache quantization scheme that keeps a window of the most recent tokens at high precision and aggressively compresses the older history. Coupled with a channel-reorder step that groups similar-magnitude channels (improving per-group quant fit) and clipped dynamic group-level quantization, SKVQ pushes the KV cache to 2-bit keys and 1.5-bit values with minimal accuracy loss. Enables up to 1M context length on an 80GB GPU for 7B models with up to 7× faster decoding.

## Key Contributions
- Sliding-window precision schedule: last W tokens (W ≈ 128) kept at FP16 or INT4; older history at INT2 K / INT1.5 V.
- Channel reorder pre-processing: permute channels so adjacent channels have similar dynamic range, improving per-group quant fit. Permutation is folded into adjacent weights (free at inference).
- Clipped dynamic group quantization: per-group scale clipped at a learned percentile to suppress outlier influence.
- Demonstrates 1M-token contexts on 7B models on a single 80GB GPU, with 7× decoding speedup.

## Key Figures/Tables to Study
- **Figure 2:** Attention mass distribution by distance from the current query — sharp recency bias justifying the window approach.
- **Figure 4:** Channel reorder before / after — clear blocking into similar-magnitude bands.
- **Table 2:** LongBench tasks at varying compression — SKVQ vs KIVI vs uniform KV.

## Technical Details

### Sliding-window precision schedule
Let W be the window size (e.g. 128). For a token at distance d from the current step:
- d < W (recent): KV stored at FP16 (or INT4 if memory pressured).
- d ≥ W (history): KV stored at INT2 / INT1.5.
At each decode step, the boundary slides forward by 1 — the now-too-old (d = W) token gets re-quantized down to low-bit and appended to the compressed history buffer.

### Channel reorder
For K (and similarly V): from calibration, compute per-channel max magnitude; sort channels in descending order. Permute the channel axis of K (and the matching weights W_k, W_q to preserve attention math) so similar-magnitude channels are adjacent.

The permutation is absorbed into adjacent weights:
- W_k ← W_k[P, :] (rows reorder).
- W_q ← W_q[P, :] (matching rows, so QK^T unchanged).
Zero runtime cost.

After reorder, group-of-32 per-group quant fits well because the channels in each group have similar magnitudes.

### Clipped dynamic group quantization
Per group g of size 32, compute the absmax m_g. Clip at the c-th percentile of {m_g}:
`m̂_g = min(m_g, percentile_c({m_g}))`
`scale_g = m̂_g / (2^{b-1} − 1)`
c=99 typical — kills the single-group outliers that would otherwise dominate the scale.

### V at 1.5 bits
1.5 bits = pack pairs of V elements into a single 3-bit code from an 8-entry codebook (vector quant of pairs). Codebook fit per-token.

### Bit budgets
- K: 2 bits + FP16 group scale per 32 → 2.5 bits/element.
- V: 1.5 bits + FP16 codebook per group → 1.75 bits/element.
- Recent window (W=128): FP16, contributes ≈ 0 to amortised cost for T » W.

### Throughput
7B model at 1M context: KV at FP16 = 250 GB (impossible on 80GB). KV at SKVQ-2/1.5 = ~30 GB, fits. Decoding speedup 7× because each attention call reads 8× less HBM.

## Connections
- KV-quant siblings: [[kivi]], [[kvquant]], [[gear]], [[wkvquant]], [[qaq]].
- Recency-bias predecessor: StreamingLLM attention sink work (architecture, not quant).
- Channel-reorder lineage: shared with [[duquant]], [[rptq]].
- KV-cache compression survey: [[kv-cache-survey]].
