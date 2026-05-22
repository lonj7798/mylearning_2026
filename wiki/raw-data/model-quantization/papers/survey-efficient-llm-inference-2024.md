<!-- scope: 2024 survey of efficient LLM inference — covers quantization alongside KV-cache compression, speculative decoding, sparsity, MoE
     deps: [[survey-llm-quantization-2024]]
     see-also: [[kvquant]], [[kivi]], [[gptq]], [[awq]], [[bitnet-b158]]
-->

# A Survey on Efficient Inference for Large Language Models (Zhou / Yuan / Wan 2024)
- **Core Insight:** Quantization is one of four orthogonal levers for efficient LLM inference — alongside KV-cache compression, speculative decoding, and architectural pruning/sparsity — and the production-best deployment usually combines multiple levers (e.g. W4-AWQ weights + 4-bit KV cache + speculative draft model) rather than maximizing any single one.
- **Guideline:** When optimizing an LLM inference pipeline, do not over-invest in pushing quantization to sub-4-bit; instead pair moderate quantization (W4A16 with AWQ/GPTQ) with KV-cache quantization, speculative decoding, and FlashAttention-style attention kernels — the marginal returns of pushing one lever are smaller than picking up the next lever.
- **Authors:** Various 2024 surveys, e.g. Yuan et al. "LLM Inference Unveiled" (2024); Zhou et al. "A Survey of Resource-efficient LLM and Multimodal Foundation Models" (2024); Wan et al. "Efficient Large Language Models" (2024)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2401.18079 ("LLM Inference Unveiled: Survey and Roofline Model Insights"); https://arxiv.org/abs/2312.03863 (Wan 2024 "Efficient Large Language Models: A Survey")
- **Relevant topics:** efficient inference, quantization survey, KV cache, speculative decoding, sparsity, roofline analysis

## Abstract
The 2024 efficient-LLM-inference surveys give a holistic view of inference optimization, situating quantization as one tool among many. They organize the space around the inference bottleneck: prefill is *compute-bound* (helped by quantizing activations and weights together), decode is *memory-bandwidth-bound* (helped by weight-only quantization and KV-cache compression). Coverage includes: quantization (GPTQ, AWQ, SmoothQuant, FP8/FP4), KV-cache methods (quant: KIVI/KVQuant; eviction: H2O/StreamingLLM; merging: CacheGen), speculative decoding (Medusa, Eagle, draft-model speculative), MoE inference (expert parallelism, token routing), and attention kernels (FlashAttention, PagedAttention, RingAttention). Includes a **roofline analysis** showing which optimizations matter at which sequence length / batch size / model size operating point.

## Key Contributions
- Roofline view of LLM inference: identifies which optimization helps in which regime.
- Unified taxonomy across quantization, KV cache, speculative decoding, sparsity, and kernels.
- Empirical decode-bound vs prefill-bound analysis: 70B Llama generation at batch-1 is ~95% memory-bandwidth bound on H100 → weight-only quantization is the dominant speedup.
- Highlights complementarity: W4A16 + KV2 + speculative gives ~5–8× speedup, more than any single technique.
- Surveys hardware (H100 / B100 / MI300 / Gaudi) and framework (vLLM / TensorRT-LLM / SGLang) support.

## Key Figures/Tables to Study
- **Roofline plot**: arithmetic intensity (FLOPS / byte) vs operation; shows decode is in the memory-bound region for typical batch sizes.
- **Speedup decomposition**: end-to-end speedup attributable to weight quant vs KV quant vs speculative vs kernel optimization.
- **Method-coverage matrix**: which framework supports which technique on which hardware.

## Technical Details

### Inference phase decomposition
LLM serving has two distinct phases:
1. **Prefill**: process the prompt (batch_size × seq_len × hidden) in parallel; compute-bound.
2. **Decode**: generate tokens one at a time (batch_size × 1 × hidden); memory-bandwidth-bound at small batch.

### Roofline analysis
For an op with arithmetic intensity I = FLOPS / bytes, the achievable performance is `min(peak_FLOPS, I × peak_bandwidth)`.

For Llama-70B decode at batch-1:
- Weight reads: ~140 GB per token (FP16)
- FLOPs: ~140 GFLOPs per token
- Arithmetic intensity: ~1 FLOP/byte
- H100 ridge point: ~30 FLOPS/byte
- ⇒ Bandwidth-bound; quantizing weights to INT4 directly gives ~4× speedup.

### Quantization coverage
- **Weight-only (W4A16)**: GPTQ, AWQ, SqueezeLLM, NF4, HQQ. Dominant for decode-bound workloads.
- **Weight+activation (W8A8, W4A4)**: SmoothQuant, QuaRot, SpinQuant. Helps prefill and high-batch decode.
- **FP8 / FP4**: native-format Hopper / Blackwell deployment.
- **KV-cache quant (W16A16 KV4/KV2)**: KIVI, KVQuant; reduces KV memory and bandwidth.
- **Mixed-precision per-tensor**: GGUF k-quants.

### Non-quantization levers covered

**KV cache compression** (non-quantization):
- Eviction: H2O, StreamingLLM (keep recent + heavy-hitter tokens).
- Merging: CacheGen, MiniCache.
- Hierarchical: PagedAttention (vLLM).

**Speculative decoding**:
- Draft-model speculative (small LM proposes, big LM verifies).
- Self-speculative (Medusa heads, Eagle).
- Verification kernels.

**Sparsity / pruning**:
- Unstructured: SparseGPT (Frantar 2023).
- Structured 2:4 sparsity (NVIDIA Ampere+).
- Mixture-of-Experts (Mixtral, DeepSeek MoE).

**Attention kernels**:
- FlashAttention 1/2/3: tiling for O(N) memory.
- PagedAttention: virtual memory for KV cache.
- Ring/Sequence-parallel attention: long-context.

### Hardware coverage
- **NVIDIA H100/H200**: FP8 / INT8 tensor cores; ~1979 TFLOPS FP8.
- **NVIDIA Blackwell B100/B200**: FP4 / NVFP4 / MXFP4 tensor cores; ~9 PFLOPS FP4.
- **AMD MI300X/MI355X**: FP8, MX format support.
- **Intel Gaudi 3**: FP8, MXFP6.
- **Apple M-series**: ANE INT8/FP16; Metal compute.

### Framework coverage
| Framework | W4A16 | W8A8 | KV quant | Speculative | FP8 | FP4 |
|-----------|-------|------|----------|--------------|-----|-----|
| vLLM | ✓ (GPTQ/AWQ/Marlin) | ✓ (FP8) | ✓ (KVQuant) | ✓ | ✓ | ✓ (NVFP4) |
| TensorRT-LLM | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SGLang | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| llama.cpp | ✓ (gguf) | ✓ (q8_0) | ✓ | ✓ | partial | partial |
| HF transformers | ✓ (bnb/gptq/awq) | ✓ | partial | partial | ✓ | partial |

### Stacking the levers
Production deployment typically combines:
- W4 weight quant (AWQ or GPTQ) → 4× memory reduction.
- KV4 quant (KIVI or KVQuant) → 4× KV memory.
- PagedAttention → batch-efficient KV management.
- Speculative decoding (Eagle or draft model) → ~2× decode speedup.
- FlashAttention 3 → ~30% prefill speedup.

Net: 10–15× throughput vs FP16 baseline at iso-quality.

### Open challenges identified
- W4A4 reliability across model families.
- KV-cache quant interaction with long context (10M+ tokens).
- Speculative decoding for low-resource languages / specialized domains.
- Quantization × MoE expert routing.
- Online (on-device) calibration.

## Connections
- [[survey-llm-quantization-2024]] — quant-specific survey; this one situates quant in the broader inference stack.
- [[survey-low-bit-llm-2024]] — sub-8-bit specific.
- [[kvquant]] / [[kivi]] — KV-cache quant covered as one lever.
- [[gptq]] / [[awq]] — weight-only PTQ workhorses.
- [[marlin-kernel]] / [[machete-kernel]] — kernel-level optimizations.
- [[smoothquant]] / [[quarot]] — activation quant.
- [[transformer-engine]] — NVIDIA FP8 framework support.
