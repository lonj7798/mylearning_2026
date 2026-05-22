<!-- scope: TensorRT-LLM official docs as a practitioner source
     see-also: nvidia-inference, genai-perf
-->

# TensorRT-LLM Docs
- **Core Insight:** TensorRT-LLM docs show how LLM inference becomes an engine-building and runtime-configuration problem on NVIDIA GPUs.
- **Guideline:** Use TensorRT-LLM docs for optimized deployment details: precision, engine build, parallelism, paged KV cache, and batching.
- **Authors:** NVIDIA
- **Year:** 2023-2026
- **URL:** https://nvidia.github.io/TensorRT-LLM/
- **Relevant topics:** engine build, paged KV cache, inflight batching, quantization, Triton backend

## Abstract
TensorRT-LLM is NVIDIA's optimized inference library for large language models. Its documentation covers model support, engine build workflows, runtime APIs, quantization, distributed execution, attention/KV-cache options, and Triton deployment through the TensorRT-LLM backend.

## Key Contributions
- Documents accelerator-specific optimization paths for LLM inference.
- Connects model conversion/build steps to runtime serving behavior.
- Covers quantization and parallelism modes used in production GPU deployments.
- Explains integration with Triton and NVIDIA performance tooling.

## Key Figures/Tables to Study
- Architecture and feature docs: runtime components and supported optimizations.
- Build examples: model-specific engine construction.
- Triton backend docs: in-flight batching and serving configuration.

## Technical Details
TensorRT-LLM deployments should record engine build parameters because they constrain runtime shape support and performance. Precision, KV-cache paging, maximum input/output lengths, tensor parallelism, and batching policy can change both memory footprint and latency.

## Connections
- [[nvidia-inference]] gives the broader NVIDIA stack.
- [[genai-perf]] measures endpoints produced by this stack.
