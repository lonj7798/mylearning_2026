<!-- scope: synthesis card for expert-parallel Mixture-of-Experts inference
     deps: tensor-parallel-inference
     see-also: pipeline-parallel-inference, flashinfer
-->

# Expert Parallel Inference
- **Core Insight:** MoE inference scales model capacity by sharding experts, but serving performance depends on routing balance, all-to-all communication, and per-expert batching.
- **Guideline:** Use expert parallelism for sparse MoE models when expert weights cannot or should not be replicated, and budget engineering effort for routing, load balancing, and fused grouped GEMMs.
- **Authors:** synthesis from GShard, DeepSpeed-MoE/Inference, Megatron Core, and TensorRT-LLM sources
- **Year:** synthesis card, 2020-2025 sources
- **URL:** https://arxiv.org/abs/2006.16668 ; https://arxiv.org/abs/2207.00032 ; https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html ; https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html
- **Relevant topics:** MoE inference, expert parallelism, token routing, all-to-all, grouped GEMM, load balancing

## Abstract
This is a synthesis card rather than one artifact. Expert parallelism places different MoE experts on different devices. Tokens are routed to top-k experts, transferred to the owning devices, processed in expert-specific batches, and gathered back. The approach lets sparse models have far more parameters than activated compute per token, but the serving system must manage routing imbalance and communication overhead.

## Key Contributions
- GShard demonstrated conditional computation with automatic sharding and top-2 expert routing at very large scale.
- DeepSpeed Inference covers dense and sparse transformer inference with multi-GPU and heterogeneous strategies.
- Megatron Core documents expert parallel size, expert tensor parallelism, and MoE communication overlap.
- TensorRT-LLM documents expert parallel inference support and deployment tradeoffs.

## Key Figures/Tables to Study
- GShard MoE layer diagram: router, top-k expert selection, and combine weights.
- TensorRT-LLM expert parallelism docs: practical tensor-vs-expert parallel comparison.
- Megatron Core MoE feature docs: expert/model parallel configuration flags.
- DeepSpeed-MoE/Inference materials: sparse model serving and kernel/communication concerns.

## Technical Details
At each MoE layer, the router computes token-to-expert assignments. Expert parallel systems perform an all-to-all so tokens arrive at the GPU that owns each selected expert. Each device runs its local experts, often with grouped GEMM to improve utilization over many small expert batches. Outputs are sent back and combined according to router weights.

Inference adds burstiness: prompts and generated tokens can route unevenly, and decode batches may be small. Capacity factors, token dropping, redundant experts, expert tensor parallelism, and communication overlap are tools to control latency.

Expert parallelism composes with tensor and pipeline parallelism, but the parallel-group design becomes a major part of the serving architecture.

## Connections
- [[tensor-parallel-inference]] can be used inside each expert or for non-MoE layers.
- [[pipeline-parallel-inference]] can split MoE and dense layers by depth.
- [[flashinfer]] includes MoE/GEMM operator concerns in newer serving-kernel stacks.
- [[cuda-graphs-inference]] is harder with EP because token routing changes expert batch shapes.
