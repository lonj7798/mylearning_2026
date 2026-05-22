<!-- scope: llama.cpp project summary for local and CPU/GPU LLM inference
     see-also: gemma-inference, phi-inference
-->

# llama.cpp
- **Core Insight:** llama.cpp made quantized local LLM inference practical across CPUs, consumer GPUs, and edge devices.
- **Guideline:** Use llama.cpp as the reference for GGUF quantized local serving, but separate model quality effects from quantization and backend effects.
- **Authors:** Georgi Gerganov and contributors
- **Year:** 2023-2026
- **URL:** https://github.com/ggml-org/llama.cpp
- **Relevant topics:** GGUF, quantization, CPU inference, local server, Metal/CUDA/Vulkan

## Abstract
llama.cpp is an open-source C/C++ inference runtime for Llama-family and many other transformer models. It supports GGUF model files, many quantization formats, CPU and accelerator backends, grammar/structured generation features, embeddings, and a local HTTP server.

## Key Contributions
- Popularized GGUF as a portable quantized model format.
- Enables local inference on commodity hardware.
- Provides many weight quantization tradeoffs for memory, speed, and quality.
- Includes `llama-server` for local OpenAI-compatible-style serving workflows.

## Key Figures/Tables to Study
- README/build docs: backend support and server usage.
- Quantization docs/examples: GGUF quant families and memory tradeoffs.
- Server docs: context, batching, slots, and sampling parameters.

## Technical Details
llama.cpp performance depends on quantization type, CPU SIMD/GPU backend, context length, batch size, and whether layers are offloaded to GPU. KV cache remains a major memory term at long context even when weights are heavily quantized. It is best for local, edge, and experimentation scenarios, not as a drop-in replacement for high-throughput multi-GPU serving.

## Connections
- [[gemma-inference]] and [[phi-inference]] are common compact-model targets.
- [[ttft-tpot-itl]] still applies to local interactive serving.
