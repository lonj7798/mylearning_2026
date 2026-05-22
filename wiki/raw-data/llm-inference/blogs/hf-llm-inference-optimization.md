<!-- scope: Hugging Face optimization guidance for LLM inference
     see-also: huggingface-inference, ttft-tpot-itl
-->

# Hugging Face LLM Inference Optimization
- **Core Insight:** Hugging Face guidance frames inference optimization as coordinated choices across attention kernels, KV cache, batching, quantization, and generation settings.
- **Guideline:** Use Hugging Face optimization posts for practical knobs, then validate with runtime-specific benchmarks.
- **Authors:** Hugging Face
- **Year:** 2023-2026
- **URL:** https://huggingface.co/docs/transformers/llm_optims
- **Relevant topics:** KV cache, FlashAttention, quantization, assisted generation, batching

## Abstract
Hugging Face documentation and blog material on LLM inference optimization explains how Transformers and related tooling improve generation performance through KV caching, optimized attention implementations, quantization, static caches, assisted/speculative generation, and hardware-aware settings.

## Key Contributions
- Makes generation-time cache behavior accessible to practitioners.
- Documents when optimized attention and quantization help.
- Connects decoding controls to latency and memory use.
- Provides reference APIs for experimentation before moving to specialized servers.

## Key Figures/Tables to Study
- Transformers LLM optimization docs: cache and attention settings.
- Generation docs: logits processors, sampling, and stopping controls.
- Quantization docs: bitsandbytes/GPTQ/AWQ-style deployment paths.

## Technical Details
Transformers-level optimization is excellent for correctness experiments and smaller deployments, but server runtimes add continuous batching, admission control, and endpoint metrics. Static caches can reduce compilation overhead in some paths, while quantization lowers weight memory but does not eliminate KV-cache growth with context length.

## Connections
- [[huggingface-inference]] is the ecosystem summary.
- [[ttft-tpot-itl]] maps optimizations to user-facing metrics.
