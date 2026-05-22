<!-- scope: OSDI 2024 paper on disaggregating prefill and decode for goodput
     deps: [[continuous-batching]]
     see-also: [[splitwise]], [[mooncake]], [[prefill-decode-disaggregation]]
-->

# DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
- **Core Insight:** Prefill and decode interfere when colocated because they stress different resources and latency metrics; assign them to separate GPU pools and optimize each phase separately.
- **Guideline:** Consider prefill/decode disaggregation when TTFT and TPOT SLOs cannot both be met by a colocated continuous-batching engine.
- **Authors:** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2401.09670
- **Relevant topics:** prefill/decode disaggregation, goodput, TTFT, TPOT, placement, parallelism, interconnect bandwidth

## Abstract
DistServe argues that colocating prefill and decode couples two phases with different bottlenecks. Prefill is compute-intensive and determines time-to-first-token, while decode is memory-bandwidth-intensive and determines time per output token. DistServe assigns the phases to different GPUs, chooses phase-specific parallelism/resource allocations, and places workers according to network bandwidth so KV transfer does not erase the benefit of disaggregation.

## Key Contributions
- Frames LLM serving objective as goodput under separate TTFT and TPOT SLOs.
- Identifies prefill-decode interference in colocated systems.
- Separates prefill and decode onto different GPU pools.
- Co-optimizes resource allocation and parallelism for each phase.
- Includes placement decisions to reduce disaggregation communication cost.
- Reports substantially higher request rates or tighter SLOs than colocated baselines.

## Key Figures/Tables to Study
- Prefill/decode interference measurements.
- Architecture diagram showing prefill workers, decode workers, and KV transfer.
- Goodput curves under TTFT/TPOT SLO combinations.
- Placement/resource allocation evaluation across model sizes and workloads.

## Technical Details
The prefill worker computes prompt activations and the initial KV cache. The decode worker receives the KV state and performs autoregressive generation. This boundary introduces a KV transfer cost proportional to prompt length, model layers, KV heads, head dimension, and dtype, so placement must account for interconnect bandwidth.

DistServe's scheduler/resource planner searches allocations that maximize the rate of requests satisfying both TTFT and TPOT constraints. The key is not raw throughput; tokens that miss SLO do not count as useful capacity.

## Connections
- [[splitwise]] studies phase splitting and hardware specialization from a similar premise.
- [[mooncake]] extends disaggregation around a KVCache-centric production architecture.
- [[prefill-decode-disaggregation]] synthesizes the design pattern across systems.

## Notes
The paper's main metric is SLO-constrained goodput, not maximum raw token throughput.
