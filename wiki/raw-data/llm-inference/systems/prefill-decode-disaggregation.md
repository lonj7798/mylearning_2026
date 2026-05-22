<!-- scope: synthesis source page for prefill/decode disaggregation in LLM serving systems
     deps: [[distserve]], [[splitwise]]
     see-also: [[mooncake]], [[sarathi-serve]], [[admission-control-goodput]]
-->

# Prefill-Decode Disaggregation
- **Core Insight:** Prefill and decode are different workloads, so separating them can remove interference and allow phase-specific resource allocation.
- **Guideline:** Disaggregate only when phase-specific gains exceed KV transfer, orchestration, and placement costs.
- **Authors:** Synthesis around Splitwise, DistServe, Mooncake, and vLLM disaggregated-prefill docs
- **Year:** 2023-present
- **URL:** https://arxiv.org/abs/2311.18677 ; https://arxiv.org/abs/2401.09670 ; https://arxiv.org/abs/2407.00079
- **Relevant topics:** prefill, decode, disaggregation, KV transfer, TTFT, TPOT, goodput, cluster placement

## Abstract
Prefill processes the prompt in parallel and creates the initial KV cache. Decode consumes that cache one token at a time and appends new KV entries. Colocated systems run both phases on the same workers, which simplifies state management but creates interference. Disaggregated systems assign prefill and decode to separate workers or pools, transfer KV state between them, and tune resources for TTFT and TPOT independently.

## Key Contributions
- Separates a core serving design pattern from individual systems.
- Clarifies when disaggregation helps: distinct bottlenecks, strict SLOs, long prompts, or specialized hardware.
- Clarifies when it hurts: high KV-transfer cost, weak interconnect, small prompts, or low load.
- Connects phase splitting, goodput optimization, and KVCache-centric storage.

## Key Figures/Tables to Study
- Splitwise phase characterization figures.
- DistServe architecture and goodput optimization results.
- Mooncake KVCache-centric architecture and cache-tier diagrams.
- vLLM disaggregated prefill examples for implementation shape.

## Technical Details
The state boundary is the KV cache. Its transfer size is roughly `2 * layers * kv_heads * head_dim * prompt_tokens * dtype_bytes`, plus metadata. Long prompts therefore increase both prefill compute and handoff cost. Placement should minimize cross-network transfer and keep decode workers close to the cache source or cache store.

Resource allocation differs by phase. Prefill generally benefits from compute throughput and larger prompt batches. Decode is latency-sensitive per token and often memory-bandwidth constrained. Good disaggregation lets each phase use different parallelism, batching limits, and hardware choices.

Schedulers must decide where a request's prefill runs, where decode runs, when KV is transferred, and whether a request is worth admitting under current SLO/load. Failures and cancellations also require cache cleanup across components.

## Connections
- [[distserve]] is the primary goodput-optimized disaggregation paper.
- [[splitwise]] motivates phase splitting through workload characterization.
- [[mooncake]] makes the KV-cache layer explicit and production-oriented.
- [[sarathi-serve]] is a colocated strategy that attacks similar prefill/decode interference.
