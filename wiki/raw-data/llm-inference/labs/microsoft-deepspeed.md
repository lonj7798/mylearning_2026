<!-- scope: Microsoft DeepSpeed inference and FastGen project summary
     see-also: llama-3-inference, mixtral-inference
-->

# Microsoft DeepSpeed Inference
- **Core Insight:** DeepSpeed Inference focuses on distributed transformer inference through kernel injection, tensor parallelism, quantization, and memory-efficient generation.
- **Guideline:** Use DeepSpeed as a reference for distributed inference techniques, especially when training infrastructure and inference infrastructure overlap.
- **Authors:** Microsoft DeepSpeed team
- **Year:** 2021-2026
- **URL:** https://www.deepspeed.ai/inference/
- **Relevant topics:** tensor parallelism, kernel injection, quantization, DeepSpeed-FastGen, ZeRO-Inference

## Abstract
DeepSpeed includes inference components for serving large transformer models with optimized kernels, tensor parallelism, quantization, and memory management. Related projects and posts such as DeepSpeed-FastGen discuss high-throughput generation and system optimizations for LLM inference.

## Key Contributions
- Brings distributed systems techniques from training into inference.
- Provides kernel injection and tensor-parallel execution for transformer layers.
- Explores memory-offload and quantization strategies for large-model serving.
- Offers an important historical bridge from training-scale systems to LLM serving engines.

## Key Figures/Tables to Study
- DeepSpeed inference docs: API options and tensor-parallel configuration.
- DeepSpeed-FastGen materials: generation throughput and system design.
- ZeRO-Inference docs/posts: offload strategy for memory-constrained inference.

## Technical Details
DeepSpeed Inference can replace transformer modules with optimized kernels and shard computation across devices. It is most relevant when the deployment already uses DeepSpeed or when very large dense models require tensor parallelism and memory management. Modern serving comparisons should include vLLM/SGLang/TensorRT-LLM because scheduler and KV-cache management have become central.

## Connections
- [[nvidia-inference]] provides an accelerator-vendor optimized path.
- [[vllm-project]] provides a serving-engine-first contrast.
