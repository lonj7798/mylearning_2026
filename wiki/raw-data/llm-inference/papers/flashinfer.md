<!-- scope: serving-oriented attention engine and kernel library for LLM inference
     deps: flashattention-2, flashdecoding
     see-also: cuda-graphs-inference, tensor-parallel-inference
-->

# FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving
- **Core Insight:** LLM serving needs a flexible attention engine that handles heterogeneous KV-cache layouts, dynamic requests, and CUDA graph constraints, not only a single fast kernel.
- **Guideline:** Use FlashInfer-style operator libraries when building serving systems that need pluggable prefill/decode/paged attention kernels across model variants.
- **Authors:** Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, Luis Ceze
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2501.01005
- **Relevant topics:** LLM serving kernels, KV-cache formats, JIT templates, CUDAGraph, load-balanced scheduling, paged attention

## Abstract
FlashInfer is a customizable GPU attention engine for LLM serving. It addresses the mismatch between diverse serving workloads and fixed attention kernels by supporting block-sparse and composable KV-cache formats, JIT-specialized attention templates, and scheduling that remains compatible with CUDA graph capture. The paper reports kernel-level and end-to-end improvements in inter-token latency, long-context inference, and parallel generation.

## Key Contributions
- Treats KV-cache heterogeneity as a first-class operator design problem.
- Introduces composable formats and block-sparse representations for attention over cached tokens.
- Provides customizable attention templates with JIT compilation for variants such as GQA, RoPE, and quantized KV cache.
- Adds load-balanced scheduling that works with static CUDA graph requirements.
- Integrates with serving frameworks such as SGLang, vLLM, and MLC-Engine.

## Key Figures/Tables to Study
- Architecture overview: maps FlashInfer into serving engines and operator families.
- KV-cache format diagrams: useful for contrasting ragged, paged, and block-sparse layouts.
- CUDAGraph-compatible scheduling section: explains the static-shape tension in dynamic serving.
- End-to-end benchmark tables: compare kernel wins with serving-level latency changes.

## Technical Details
FlashInfer separates the serving attention problem into prefill, decode, and append/update operations over a KV cache. It exposes kernels that can work with ragged batches, paged caches, grouped-query attention, compressed KV cache, and customized score modification.

The important serving detail is shape regularization. CUDA graphs prefer fixed execution shapes, while request batches are dynamic. FlashInfer's scheduling and batching design groups work so kernels can be efficient and graph-capturable without giving up all flexibility.

The project has also become an integration point for attention, GEMM, communication, and sampling kernels through official documentation and NVIDIA releases.

## Connections
- [[flashdecoding]] is the decode-specific ancestor for long-context attention.
- [[cuda-graphs-inference]] explains why graph-compatible scheduling matters.
- [[xformers-memory-efficient-attention]] is a more general PyTorch operator library; FlashInfer is more serving-specific.
