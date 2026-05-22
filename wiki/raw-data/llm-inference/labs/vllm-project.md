<!-- scope: vLLM project summary for LLM serving systems
     see-also: vllm-benchmarks, vllm-docs
-->

# vLLM Project
- **Core Insight:** vLLM turns KV-cache paging plus continuous batching into a general-purpose high-throughput LLM serving engine.
- **Guideline:** Use vLLM as the default reference runtime for PagedAttention, OpenAI-compatible serving, prefix caching, and benchmarkable scheduler behavior.
- **Authors:** vLLM project, UC Berkeley Sky Computing Lab, community contributors
- **Year:** 2023-2026
- **URL:** https://docs.vllm.ai/
- **Relevant topics:** PagedAttention, continuous batching, prefix caching, OpenAI server, tensor parallelism

## Abstract
vLLM is an open-source LLM inference and serving engine designed for high throughput and efficient memory use. It exposes offline generation APIs and an OpenAI-compatible server, supports many Hugging Face models, and implements scheduler and cache-management techniques such as PagedAttention, continuous batching, prefix caching, chunked prefill, speculative decoding, and distributed execution.

## Key Contributions
- Introduces PagedAttention as a practical KV-cache memory manager.
- Provides production-oriented OpenAI-compatible serving.
- Supports a broad model catalog and quantization/runtime options.
- Offers benchmark scripts for offline and online serving.
- Serves as a common baseline for inference research and systems comparisons.

## Key Figures/Tables to Study
- vLLM docs architecture pages: engine, scheduler, worker, and KV-cache manager.
- PagedAttention paper/code paths: block tables and cache allocation.
- Benchmark docs: throughput versus serving benchmark distinction.

## Technical Details
vLLM batches requests continuously at decode steps rather than waiting for whole requests to finish. KV cache is divided into blocks and addressed through block tables, reducing fragmentation and enabling efficient memory sharing. The project also supports prefix caching, LoRA serving, structured outputs, multi-GPU parallelism, and multiple attention backends.

## Connections
- [[vllm-benchmarks]] captures the official benchmark workflow.
- [[gpt-oss-inference]], [[llama-3-inference]], and [[qwen-3-inference]] are common served models.
