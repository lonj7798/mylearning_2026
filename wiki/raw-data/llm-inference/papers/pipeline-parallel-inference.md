<!-- scope: synthesis card for pipeline-parallel transformer inference
     deps: transformer-inference-loop
     see-also: tensor-parallel-inference, expert-parallel-inference
-->

# Pipeline Parallel Inference
- **Core Insight:** Pipeline parallelism shards a model by depth, allowing models that exceed one device's memory to run by passing activations through stage GPUs.
- **Guideline:** Use pipeline parallelism when tensor parallelism is insufficient or hardware is spread across nodes, but expect token latency bubbles and lower utilization at small batch sizes.
- **Authors:** synthesis from GPipe, Megatron-LM, DeepSpeed, and vLLM sources
- **Year:** synthesis card, 2018-2025 sources
- **URL:** https://arxiv.org/abs/1811.06965 ; https://arxiv.org/abs/2104.04473 ; https://docs.vllm.ai/en/v0.7.2/serving/distributed_serving.html
- **Relevant topics:** pipeline parallelism, layer sharding, microbatching, pipeline bubbles, distributed serving

## Abstract
This is a synthesis card rather than one artifact. Pipeline parallelism partitions a neural network into consecutive layer stages. It was popularized for training by GPipe and later combined with tensor parallelism in Megatron-LM. In inference, it is mainly a memory-capacity and cluster-topology tool: each token flows through stage 1, then stage 2, and so on. Throughput improves only when enough requests or microbatches keep stages occupied.

## Key Contributions
- GPipe formalized layer partitioning plus microbatch pipelining for giant networks.
- Megatron-LM combined pipeline and tensor parallelism for trillion-parameter transformer scaling.
- vLLM exposes `pipeline_parallel_size` for inference serving, usually combined with tensor parallelism across nodes.
- DeepSpeed provides pipeline abstractions and hybrid parallel layouts.

## Key Figures/Tables to Study
- GPipe pipeline schedule diagrams: understand bubbles and microbatch fill/drain.
- Megatron-LM interleaved pipeline schedule: shows how virtual stages reduce idle time.
- vLLM distributed serving docs: practical guidance for TP within node and PP across nodes.
- DeepSpeed pipeline docs: stage topology and process-grid concepts.

## Technical Details
For autoregressive decode, each generated token depends on the previous token, so a single request cannot fully fill a deep pipeline. Serving systems rely on multiple concurrent requests, microbatching, or continuous batching to keep all stages busy.

Pipeline parallelism has lower per-layer communication volume than tensor parallelism because only activations pass between stages, not collectives inside every layer. The cost is sequential stage dependency and load-balancing sensitivity: a slow or memory-heavy stage bottlenecks the entire pipeline.

In practice, serving deployments often use TP inside a node with high bandwidth and PP across nodes where cross-node all-reduce would be too expensive.

## Connections
- [[tensor-parallel-inference]] is the common complement for splitting large layers.
- [[cuda-graphs-inference]] can graph repeated per-stage decode work if shapes are stable.
- [[expert-parallel-inference]] creates a different kind of pipeline pressure because token routing can be imbalanced.
- [[flashinfer]] sits below these strategies as an operator library used inside each stage.
