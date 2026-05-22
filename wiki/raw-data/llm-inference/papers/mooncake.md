<!-- scope: Moonshot AI Mooncake paper on KVCache-centric disaggregated LLM serving
     deps: [[distserve]]
     see-also: [[splitwise]], [[cachegen]], [[admission-control-goodput]]
-->

# Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
- **Core Insight:** In long-context production serving, KV cache is a first-class distributed resource that should guide scheduling, storage, and overload decisions.
- **Guideline:** For long-context services, design around KV-cache locality, transfer, and admission rather than treating cache as private GPU-only state.
- **Authors:** Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2407.00079
- **Relevant topics:** KV cache, disaggregated serving, prefill/decode split, CPU/DRAM/SSD cache tiers, early rejection, SLOs

## Abstract
Mooncake describes the serving platform behind Kimi. It separates prefill and decode clusters and uses underutilized CPU, DRAM, and SSD resources to form a distributed KV-cache layer. Its scheduler balances effective throughput with latency SLOs, and unlike papers assuming all requests are processed, Mooncake explicitly handles overload through prediction-based early rejection.

## Key Contributions
- Presents a production-oriented KVCache-centric disaggregated architecture.
- Separates prefill and decode clusters while adding a disaggregated KV-cache store.
- Uses non-GPU memory and storage tiers for KV cache.
- Introduces a KVCache-centric scheduler for throughput/SLO tradeoffs.
- Adds prediction-based early rejection for overload scenarios.
- Reports large gains in long-context simulated and production workloads.

## Key Figures/Tables to Study
- System architecture showing prefill cluster, decode cluster, and KV-cache storage.
- Scheduler flow for cache-aware placement and SLO-aware decisions.
- Long-context workload evaluation.
- Production workload results for Kimi.

## Technical Details
Mooncake treats KV cache as data that may be generated, transferred, stored, reused, or rejected based on end-to-end utility. The cache size scales linearly with prompt/context length and model KV dimensions, making long-context prompts expensive to admit blindly.

Disaggregation requires moving KV state from prefill to decode and possibly through cache tiers. The scheduler must reason about transfer bandwidth, cache residency, decode capacity, and whether a request can still meet latency objectives. Prediction-based early rejection protects goodput under overload by avoiding work that is unlikely to complete within SLO.

## Connections
- Extends [[distserve]] with a production KV-cache storage viewpoint.
- Relates to [[cachegen]] because both target KV transfer/loading cost.
- Feeds into [[admission-control-goodput]] through early rejection under overload.

## Notes
Mooncake is especially relevant for long-context services where cached state dominates request cost.
