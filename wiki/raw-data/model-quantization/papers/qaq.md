<!-- scope: QAQ — quality-adaptive per-token / per-head bit allocation for KV cache
     deps: [[kivi]], [[gptq]]
     see-also: [[kvquant]], [[skvq]]
-->

# QAQ: Quality Adaptive Quantization for LLM KV Cache
- **Core Insight:** Key and Value caches have different quantization sensitivity (K affects attention scores via softmax → highly sensitive; V is averaged → less sensitive), and within each cache different tokens / heads contribute unequally to attention; bit budget should be *allocated adaptively* rather than uniformly assigned.
- **Guideline:** For 10× KV compression with negligible quality loss, use QAQ: assign K-cache more bits than V-cache; within each, give attention-heavy tokens / heads higher precision; bit allocation driven by per-element sensitivity surrogate (gradient × value).
- **Authors:** Shichen Dong, Wen Cheng, Jiayu Qin, Wei Wang
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2403.04643
- **Relevant topics:** adaptive bit allocation, KV cache quantization, outlier handling, attention sensitivity

## Abstract
QAQ allocates KV-cache quantization bits *adaptively* — observing that K and V have distinct sensitivity profiles and that within each, attention-heavy tokens and heads carry disproportionate impact. The framework provides (1) differential quantization strategies for K vs V based on sensitivity analysis, (2) outlier handling for the small fraction of high-magnitude entries, and (3) attention-aware bit allocation. Achieves up to 10× compression of the KV cache with negligible impact on model performance.

## Key Contributions
- Empirical study showing the K cache is several times more quantization-sensitive than V (a 1-bit drop in K hurts PPL more than a 1-bit drop in V).
- Sensitivity-based per-token / per-head adaptive bit allocation — top tokens / heads by attention weight or by gradient magnitude get more bits.
- Outlier path: sparse FP16 buffer for the highest-magnitude entries inside both K and V.
- 10× compression ratio with minimal quality loss demonstrated across model sizes.

## Key Figures/Tables to Study
- **Figure 1:** PPL degradation curves for K vs V at varying bit counts — K's sensitivity is markedly higher.
- **Figure 3:** Bit allocation map across heads / tokens — heat-map style.
- **Table 1:** QAQ vs uniform KV4 vs KIVI on long-context QA at fixed compression ratio.

## Technical Details

### K-vs-V sensitivity
QAQ runs calibration to measure the increase in output KL divergence per bit removed from K vs from V:
`s_K = ΔKL / Δbits_K`,  `s_V = ΔKL / Δbits_V`
Consistently observes s_K > s_V across LLaMA / Mistral. Therefore the optimal allocation gives K more bits (e.g. K at 3, V at 2 to average 2.5 bits).

### Adaptive per-token bit allocation
For each token t, compute attention weight from t to all queries: `a_t = mean_q softmax(QK^T)_{q,t}`. Tokens with high a_t are attended-to often and so contribute strongly; allocate them more bits.

Simple recipe:
- Top-10% by a_t: INT4.
- Middle 80%: INT2.
- Bottom 10%: INT1 (or evicted, if combined with eviction policies).

### Per-head bit allocation
Same idea across heads: sensitivity measured per head; high-sensitivity heads get higher bits.

### Outlier path
Entries above a per-channel threshold τ (chosen so ~1% are above) are stored as FP16 (index + value) and skipped in the low-bit path.

### Sensitivity surrogate
For each element x:
`s_x = |grad_x L| · |x|`
(gradient × value, first-order Taylor approximation of the loss change from removing x). Used to rank elements for bit allocation.

### Implementation overhead
Per-token / per-head bit allocation requires a small metadata table per layer (∼32 bytes/token). Mixed-precision attention kernel dequantizes the appropriate bit-width per element.

## Connections
- KV-quant siblings: [[kivi]] (axis-asymmetric), [[kvquant]] (non-uniform + dense-and-sparse), [[gear]] (low-rank residual), [[skvq]] (window-aware).
- Joint W+KV: [[wkvquant]].
- Bit-allocation lineage: [[hawq]] (mixed-precision weight quant), [[bit-pruning]].
- KV-cache compression survey: [[kv-cache-survey]].
