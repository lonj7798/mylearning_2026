<!-- scope: WKVQuant — first joint W4 + KV4 quantization with past-only quant and 2D KV strategy
     deps: [[gptq]], [[awq]]
     see-also: [[kivi]], [[kvquant]], [[gear]]
-->

# WKVQuant: Quantizing Weight and Key/Value Cache for Large Language Models Gains More
- **Core Insight:** The sweet spot in LLM PTQ is *not* W4A4 (activations hurt accuracy a lot) and *not* W4A16 (memory still bounded by KV cache at long contexts) — it's W4 + KV4, which preserves activation precision while still capturing the KV-cache memory win; joint calibration of W and KV scales together beats sequential calibration.
- **Guideline:** When activation quantization is hurting your accuracy but you still need throughput, use WKVQuant: quantize weights to INT4 and KV cache to INT4 jointly with a single calibration objective, leave activations at FP16; achieves near weight-only PPL with near weight-activation memory savings.
- **Authors:** Yuxuan Yue, Zhihang Yuan, Haojie Duanmu, Sifan Zhou, Jianlong Wu, Liqiang Nie
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.12065
- **Relevant topics:** joint W+KV quantization, past-only quantization, 2D KV strategy, cross-block reconstruction

## Abstract
WKVQuant is the first PTQ framework to explicitly target weight + key/value cache quantization (leaving activations at FP16) as a deliberate sweet spot. It introduces (1) past-only quantization for attention — quantize the cached past tokens but not the current token's K/V — to preserve attention computation precision; (2) a two-dimensional KV quantization strategy that picks the better of per-channel vs per-token axis per layer; and (3) cross-block reconstruction regularization for joint parameter optimization. Achieves memory savings approaching weight-activation quant while matching weight-only quant accuracy.

## Key Contributions
- Identifies W4-KV4 as the practical sweet spot — captures most memory wins of W4A4 without the activation-quant accuracy cliff.
- Past-only quantization: only the *cached* K/V (the past) is INT4; the *current* K/V (newly computed at this step) is FP16, so the freshest attention contribution is precise.
- 2D KV quant strategy: per layer, automatically choose per-channel or per-token quant for K and V based on calibration variance — both KIVI's K and V choices subsumed.
- Cross-block reconstruction regularization: joint loss across multiple transformer blocks during calibration, capturing inter-block dependencies that per-block calibration misses.

## Key Figures/Tables to Study
- **Figure 1:** Memory + accuracy Pareto comparing W-only / W+A / W+KV configurations.
- **Figure 3:** Past-only quantization schematic in the attention kernel.
- **Table 2:** LLaMA W4-KV4 PPL with WKVQuant vs sequential W4 + KIVI vs OmniQuant W4A4.

## Technical Details

### Past-only KV quantization
At decode step t, attention uses K_{1..t}, V_{1..t}. WKVQuant stores K_{1..t-1}, V_{1..t-1} in INT4, computes K_t, V_t in FP16 fresh; only writes the *previous* K_{t-1}, V_{t-1} to the INT4 cache (not the brand-new one). The attention dot-product is split:
`attn = softmax([Q · K_{1..t-1}^{INT4} / √d, Q · K_t^{FP16} / √d]) · [V_{1..t-1}^{INT4}; V_t^{FP16}]`
The newest contribution is exact; only the historical past is quantized.

### 2D KV quantization (axis selection)
For each layer, calibrate variance of K along the channel axis vs the token axis; pick the lower-variance axis for grouping. Same for V. Generalises KIVI (which hard-codes K per-channel, V per-token) to a per-layer adaptive choice.

### Cross-block reconstruction loss
`L = Σ_{b=1..B} || h_b^FP − h_b^quant ||² + λ Σ_{b<b'} || h_{b'}^FP − h_{b'}^quant ||²`
The second term jointly optimises later blocks given the propagated quant error of earlier blocks — addresses the failure mode where per-block calibration looks fine but errors compound through the stack.

### Weight quant
Standard GPTQ-style per-channel INT4 with group size 128.

### Activation
Left at FP16 throughout — this is the deliberate choice that distinguishes WKVQuant from W4A4 work.

### Memory accounting
For LLaMA-2-70B at 2048 context, batch 8:
- W4: weight memory 140 GB → 35 GB.
- KV4: KV memory 5.3 GB → 1.3 GB.
- A: activations small at decode time, FP16 fine.
Total ~36.3 GB — fits comfortably on A100-80GB with serving room.

## Connections
- KV-only siblings: [[kivi]] (asymmetric per-channel K), [[kvquant]] (sub-4-bit, non-uniform), [[gear]] (low-rank residual).
- Weight-only ancestors: [[gptq]], [[awq]].
- Activation-quant alternative it deliberately avoids: [[smoothquant]], [[omniquant]].
- KV-cache compression survey: [[kv-cache-survey]].
