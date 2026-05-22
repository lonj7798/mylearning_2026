<!-- scope: earliest 2023 KV-cache quantization attempts as part of larger inference systems
     deps: [[zeroquant]], [[llm-int8]]
     see-also: [[flexgen]], [[kivi]], [[kvquant]], [[gear]], [[atom]]
-->

# KV-Cache Quantization — 2023 Early Attempts
- **Core Insight:** Long-context LLM inference is memory-bound on the KV cache (KV grows as `2 · L · H · d_head · n_layers · bytes`), and the cache tolerates aggressive INT8 / INT4 quantization much better than weights or activations — because each KV vector is only read once per output token, the error doesn't compound across layers the way weight quant error does.
- **Guideline:** As a first-cut KV cache quant in 2023, use per-token INT8 absmax on both K and V before write to cache; this halves the memory at essentially no perplexity cost. Sub-INT8 requires per-channel asymmetry for K (KIVI's later observation) and per-token symmetry for V.
- **Authors:** consolidated entry covering early 2023 KV-quant work in [[flexgen]] (Sheng et al), [[llm-qat]] (Liu et al), and ZeroQuant-derived KV pipelines
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2303.06865 (FlexGen) · https://arxiv.org/abs/2305.17888 (LLM-QAT KV variant)
- **Relevant topics:** KV cache quantization, long-context inference, INT8 / INT4 KV, per-token vs per-channel scale

## Abstract
KV cache quantization in 2023 was driven by two distinct settings: (a) offloaded / memory-constrained inference (FlexGen) where the KV cache lives in CPU DRAM or NVMe and bandwidth into the GPU is the bottleneck; (b) long-context decode on a single GPU where the KV cache exceeds the model weights in memory. Early recipes were uniformly simple: per-token absmax INT8 (or INT4 in FlexGen) on K and V before write to cache. These first attempts established the empirical fact that KV4 + W16 holds quality much better than W4 + KV16 — because KV error is read-once-per-token while weight error compounds layer-after-layer — motivating the mature 2024 lineage ([[kivi]], [[kvquant]], [[gear]], [[atom]]).

## Key Contributions
- Empirical demonstration that KV cache quantization is *easier* than weight quantization at the same bit-width.
- Per-token INT8 absmax established as the standard baseline.
- Quantify the memory contribution of KV vs weights at long context: at L = 32k, KV often dominates.
- Identifies the asymmetry: K and V have different statistical structure (V is per-token-bounded, K can have per-channel outliers) — picked up in 2024 by [[kivi]].

## Technical Details

### Standard 2023 KV INT8 recipe
For attention head h at layer ℓ, write K_t, V_t (each ∈ R^{d_head}) on each new token t:
```
s_K = max|K_t| / 127
K_q = round(K_t / s_K)            ∈ INT8^{d_head}
s_V = max|V_t| / 127
V_q = round(V_t / s_V)            ∈ INT8^{d_head}
cache: store (K_q, s_K), (V_q, s_V) per token per head per layer
```
Read-time: dequantize on the fly inside the attention kernel.

### FlexGen KV4 variant
For NVMe-offloaded inference, FlexGen pushes K, V to INT4 with per-token absmax and per-head asymmetry. Quality loss <0.1 ppl on OPT-30B benchmarks despite the more aggressive bit-width — because reads are one-shot per token.

### Memory accounting
For LLaMA-2-7B with n_layers=32, n_heads=32, d_head=128:
- KV per token = `2 · 32 · 32 · 128 · 2 bytes (FP16) = 524 KB`
- At 32k context: ~16 GB KV vs 13 GB weights — KV dominates.
- INT8 KV: 8 GB. INT4: 4 GB.

### Why KV quant tolerates lower bits
- A KV vector is dequantized once per query token and used in one attention dot product.
- Error does not feed into the next layer's weights or accumulate across positions in the same way activation quant error does.
- KV is computed from already-clean FP16 activations, so the error is purely the quant-noise injected at cache-write time.

### Hyperparameters (typical)
| Knob | Value |
|------|-------|
| K/V bits | 8 (mainstream), 4 (FlexGen offload) |
| Scale granularity | per-token absmax (per head) |
| Symmetric / asymmetric | symmetric (2023 default) |
| Calibration | none — dynamic scales |

## Connections
- 2023 instantiations: [[flexgen]] (NVMe-offloaded W4KV4), [[llm-qat]] (W4A8KV4 in QAT loop).
- 2024 maturation — per-channel K, per-token V: [[kivi]].
- 2024 ultra-low-bit KV with non-uniform + sparse: [[kvquant]].
- KV error-compensated: [[gear]].
- KV inside W4A4 inference: [[atom]] (W4A4KV4 with sub-channel reorder).
- Joint W+KV quant calibration: [[wkvquant]].
- Surveys: [[kv-cache-survey]], [[kv-cache-compression-survey-2025]].
