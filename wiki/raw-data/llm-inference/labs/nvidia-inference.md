<!-- scope: NVIDIA inference stack for LLM deployment
     see-also: tensorrt-llm-docs, genai-perf
-->

# NVIDIA Inference
- **Core Insight:** NVIDIA's LLM inference stack combines optimized kernels, TensorRT-LLM engines, Triton serving, and benchmarking tools into an accelerator-centered deployment path.
- **Guideline:** Use NVIDIA docs to understand production GPU serving knobs: engine build, paged KV cache, in-flight batching, quantization, and GenAI-Perf measurement.
- **Authors:** NVIDIA
- **Year:** 2023-2026
- **URL:** https://nvidia.github.io/TensorRT-LLM/
- **Relevant topics:** TensorRT-LLM, Triton, NIM, in-flight batching, quantization, GenAI-Perf

## Abstract
NVIDIA's LLM inference ecosystem includes TensorRT-LLM for optimized model execution, Triton Inference Server for serving, NIM microservices for packaged deployment, and GenAI-Perf for measurement. The stack exposes GPU-specific optimizations such as fused kernels, quantization, paged KV cache, in-flight batching, tensor parallelism, and speculative decoding.

## Key Contributions
- Provides a production-oriented path for optimized NVIDIA GPU inference.
- Integrates engine compilation, serving, and measurement tooling.
- Supports quantization modes and distributed parallelism for large models.
- Documents practical deployment concepts used in MLPerf and enterprise serving.

## Key Figures/Tables to Study
- TensorRT-LLM docs: feature matrix and backend architecture.
- Triton TensorRT-LLM backend docs: in-flight batching and request scheduling.
- GenAI-Perf docs: endpoint benchmark metric definitions.

## Technical Details
TensorRT-LLM typically builds optimized engines for a target model, precision, and parallelism configuration. Runtime choices such as KV-cache paging, maximum batch size, maximum sequence length, and quantization shape memory residency and latency. Triton adds serving concerns: batching, model instances, metrics, and endpoint protocols.

## Connections
- [[tensorrt-llm-docs]] covers documentation details.
- [[genai-perf]] is the associated measurement tool.
