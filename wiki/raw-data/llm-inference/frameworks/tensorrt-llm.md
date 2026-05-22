<!-- scope: TensorRT-LLM framework serving architecture and APIs
     deps: [[pagedattention]], [[continuous-batching]]
     see-also: [[tensorrt-llm-paged-kv]]
-->

# TensorRT-LLM
- **Core Insight:** TensorRT-LLM is NVIDIA's optimized LLM inference stack combining model compilation, runtime kernels, in-flight batching, and deployment integrations.
- **Guideline:** Use TensorRT-LLM when NVIDIA GPU throughput/latency and production deployment matter enough to accept engine build and platform-specific complexity.
- **Authors:** NVIDIA
- **Year:** 2023-present
- **URL:** https://nvidia.github.io/TensorRT-LLM/ and https://github.com/NVIDIA/TensorRT-LLM
- **Relevant topics:** TensorRT engines, in-flight batching, paged KV cache, quantization, tensor parallelism, Triton, executor API

## Abstract
TensorRT-LLM provides tools and runtime components for building and serving optimized LLM engines on NVIDIA GPUs. It supports model conversion/build flows, a Python and C++ runtime, an executor API, Triton deployment, paged KV cache, in-flight batching, quantization, speculative decoding, and multi-GPU/multi-node parallelism.

## Key Contributions
- Uses TensorRT-optimized graphs and custom kernels for NVIDIA GPU inference.
- Implements in-flight batching so active requests can join/leave batches during generation.
- Supports paged and contiguous KV-cache modes, with cache reuse, offload, and eviction features.
- Provides deployment paths through Python APIs, C++ runtime, and Triton backend.
- Offers a wide set of NVIDIA-focused optimizations such as FP8/INT4 quantization and communication-aware parallel execution.

## Key Figures/Tables to Study
- TensorRT-LLM docs homepage: API and deployment map.
- Paged Attention, IFB, and Request Scheduling docs: batching/cache design.
- KV cache management docs: block hierarchy, pools, events, reuse, and eviction.
- `tensorrt_llm/runtime` and executor docs: serving runtime boundary.

## Technical Details
Public APIs:
- Build/convert models into TensorRT-LLM engines or use supported high-level workflows.
- Serve through TensorRT-LLM APIs, Triton backend, or integrations that wrap the executor.
- Configure `KvCacheConfig`, batching, parallelism, quantization, and model-specific build options.

Scheduler/cache approach:
- In-flight batching continuously updates the active batch as requests finish and new requests enter.
- Paged KV cache splits KV storage into blocks/pages instead of per-request contiguous allocation.
- KV cache reuse can keep blocks across requests and evict using policies described in NVIDIA docs.
- Scheduler decisions are coupled to engine capacity, KV memory, and decode/prefill state.

Relevant code/docs:
- Docs: https://nvidia.github.io/TensorRT-LLM/
- GitHub: https://github.com/NVIDIA/TensorRT-LLM
- Paged attention/IFB: https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html
- KV management: https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-management.html

Strengths:
- Deep NVIDIA optimization and production deployment path.
- Strong fit for teams standardized on NVIDIA GPUs and Triton-like serving.
- Rich low-level controls for quantization, parallelism, and cache behavior.

Limitations:
- Less portable than PyTorch-first engines.
- Engine build and version compatibility add operational overhead.
- Custom model support may require conversion/build work rather than simply loading a Hugging Face checkpoint.

## Connections
- See [[tensorrt-llm-paged-kv]] for paged KV details.
- Compare with [[vllm]] and [[sglang]] for Python-first serving stacks.
- Connects to [[continuous-batching]], [[pagedattention]], and production deployment chapters.
