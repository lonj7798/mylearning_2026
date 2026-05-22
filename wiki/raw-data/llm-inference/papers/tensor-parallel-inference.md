<!-- scope: synthesis card for tensor-parallel transformer inference
     deps: transformer-inference-loop
     see-also: pipeline-parallel-inference, expert-parallel-inference
-->

# Tensor Parallel Inference
- **Core Insight:** Tensor parallelism shards individual transformer layers across devices, trading memory capacity and parallel compute for collective communication on each layer.
- **Guideline:** Use tensor parallelism within high-bandwidth GPU islands when one replica is too large or when per-token latency benefits from splitting large GEMMs.
- **Authors:** synthesis from Megatron-LM, DeepSpeed Inference, vLLM, TensorRT-LLM, and Megatron Core sources
- **Year:** synthesis card, 2019-2025 sources
- **URL:** https://arxiv.org/abs/1909.08053 ; https://arxiv.org/abs/2207.00032 ; https://docs.vllm.ai/en/v0.7.2/serving/distributed_serving.html
- **Relevant topics:** tensor parallelism, Megatron-LM, all-reduce, column/row parallel linear layers, multi-GPU serving

## Abstract
This is a synthesis card rather than one artifact. Tensor parallel inference applies Megatron-style intra-layer model parallelism to serving: split attention and MLP weight matrices across GPUs, compute partial results in parallel, and use collectives to combine activations where needed. Serving frameworks such as vLLM expose this through `tensor_parallel_size`, while DeepSpeed Inference and TensorRT-LLM provide optimized kernels and communication paths.

## Key Contributions
- Megatron-LM showed practical transformer intra-layer parallelism with minimal synchronization points.
- DeepSpeed Inference applied model parallelism and fused kernels to low-latency transformer inference.
- vLLM exposes tensor parallel serving for single-node and multi-node deployment.
- TensorRT-LLM combines tensor parallelism with optimized engine build/runtime choices.

## Key Figures/Tables to Study
- Megatron-LM tensor parallel diagrams: column-parallel and row-parallel linear layers.
- Megatron-LM transformer MLP/attention partitioning equations.
- DeepSpeed Inference architecture: kernel injection plus multi-GPU inference.
- vLLM distributed serving examples: `--tensor-parallel-size` usage and memory-capacity guidance.

## Technical Details
In the MLP, a column-parallel first projection splits output channels across GPUs, applies the activation locally, then a row-parallel second projection produces partial outputs that are reduced. In attention, heads or projection dimensions can be partitioned so each GPU handles a slice of Q/K/V and output projection work.

The decode phase is latency-sensitive because every generated token requires all layers and their collectives. Tensor parallelism works best with fast interconnects such as NVLink/NVSwitch. Across slower networks, communication can dominate token latency unless the model is too large to fit otherwise.

KV cache is also sharded consistently with the attention partition. Frameworks must coordinate cache allocation, scheduling, and collective groups.

## Connections
- [[pipeline-parallel-inference]] shards by layer depth rather than within each layer.
- [[expert-parallel-inference]] shards MoE experts and introduces token routing/all-to-all communication.
- [[cuda-graphs-inference]] can reduce repeated launch overhead when TP execution shapes are stable.
- [[flashinfer]] provides kernels that may run inside each TP shard.
