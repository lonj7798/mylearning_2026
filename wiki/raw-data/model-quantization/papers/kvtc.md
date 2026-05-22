<!-- scope: KV-cache transform coding for compact reusable-cache storage
     deps: [[kvquant]], [[kv-cache-compression-survey-2025]], [[product-quantization]]
     see-also: [[adaptive-kv-cache-quant]], [[turboquant]], [[coupled-quant-eviction]]
-->

# KV Cache Transform Coding for Compact Storage in LLM Inference
- **Core Insight:** KV caches can be compressed like media signals: decorrelate features, adaptively quantize coefficients, and entropy-code the result to store reusable caches far more compactly than ordinary inference-time quantization.
- **Guideline:** Use transform coding when the deployment problem is persistent or reusable KV-cache storage, especially shared-prefix chat/code workflows, rather than only the live attention compute path.
- **Authors:** Konrad Staniszewski, Adrian Lancucki
- **Year:** 2025 (rev. 2026; ICLR 2026)
- **URL:** https://arxiv.org/abs/2511.01815
- **Relevant topics:** KV cache compression, transform coding, PCA decorrelation, adaptive quantization, entropy coding, reusable prompt cache

## Abstract
KVTC compresses key-value caches for compact on-GPU and off-GPU storage. It targets scenarios where caches are reused across turns or shared-prefix prompts, such as iterative code editing and chat. The method uses PCA-based feature decorrelation, adaptive quantization, and entropy coding. It leaves model weights unchanged, requires only brief calibration, and reports up to 20x compression while maintaining reasoning and long-context accuracy, with higher compression in selected use cases.

## Key Contributions
- Reframes KV-cache compression as transform coding rather than plain scalar quantization.
- Combines PCA decorrelation, adaptive quantization, and entropy coding.
- Targets reusable or stale caches that would otherwise consume GPU memory, require offload, or force recomputation.
- Evaluates across Llama 3, Mistral NeMo, and R1-Qwen 2.5 model families.
- Reports broad benchmark coverage including reasoning, coding, long-context, and QA tasks.

## Key Figures/Tables to Study
- Codec pipeline diagram: PCA transform -> quantization -> entropy coding -> reconstruction.
- Compression/accuracy table over AIME25, GSM8K, LiveCodeBench, LongBench, MATH-500, MMLU, Qasper, and RULER.
- Comparison against token eviction, plain quantization, and SVD methods.
- On-GPU vs off-GPU storage scenario analysis.

## Technical Details

### Codec pipeline
1. Collect a short calibration sample of KV-cache tensors.
2. Fit a PCA-like transform to decorrelate feature dimensions.
3. Quantize transformed coefficients with adaptive precision.
4. Entropy-code the quantized coefficients for compact storage.
5. Decode when the cache is reused.

### Deployment boundary
KVTC is not a direct replacement for live low-bit attention kernels such as KIVI or TurboQuant. Its strongest fit is a serving system where large prefixes repeat and caches need to be stored compactly between turns or requests.

### Why classical compression matters
The method borrows the core idea of media codecs: decorrelate first, quantize second, entropy-code last. This makes it a useful bridge from the course's classical quantization chapters to modern KV-cache serving.

## Connections
- [[product-quantization]] / [[vector-quantization]] — classical compression lineage.
- [[kvquant]] / [[kivi]] — live KV quantization baselines.
- [[adaptive-kv-cache-quant]] — adaptive token-level bit allocation; KVTC adapts coefficient storage after transform coding.
- [[coupled-quant-eviction]] — both address KV-cache storage beyond simple per-token quantization.
- [[kv-cache-compression-survey-2025]] — KVTC belongs in the broader cache-compression bucket, not only the quantization bucket.

## Notes
This is a boundary source: it uses quantization, but the main course reason to include it is to show how KV-cache compression in 2026 blends quantization, transforms, entropy coding, offload, and cache reuse.
