<!-- scope: 2026 position paper arguing LLM serving needs formal scheduling/routing/cache optimization
     deps: [[continuous-batching]], [[pagedattention]]
     see-also: [[niyama]], [[admission-control-goodput]], [[vllm-disaggregated-prefill-2026]]
-->

# Position: LLM Serving Needs Mathematical Optimization and Algorithmic Foundations, Not Just Heuristics
- **Core Insight:** Modern LLM serving systems still lean on generic policies such as FIFO, join-shortest-queue, round-robin, and LRU, but LLM inference has unique structure that needs specialized optimization models.
- **Guideline:** Treat request routing, scheduling, cache eviction, and admission control as formal optimization problems once a deployment has meaningful SLOs or mixed workloads.
- **Authors:** Zijie Zhou
- **Year:** 2026
- **URL:** https://arxiv.org/abs/2605.01280
- **Relevant topics:** scheduling, routing, KV cache eviction, admission control, optimization foundations, vLLM, SGLang

## Abstract
This position paper argues that LLM inference serving has outgrown general-purpose distributed-systems heuristics. The distinctive properties of LLM inference include prefill/decode phase asymmetry, continuously growing KV cache memory, unknown output lengths, and continuous batching constraints. The paper calls for mathematical models and algorithmic foundations that can produce policies with performance guarantees instead of deployment-specific heuristics.

## Key Contributions
- Identifies a mismatch between classical serving heuristics and LLM-specific workloads.
- Names the core structure a serving algorithm must model: phase asymmetry, growing KV state, uncertain generation length, and batch coupling.
- Frames KV eviction, request routing, scheduling, and admission control as research problems rather than engineering afterthoughts.
- Connects practical systems such as vLLM and SGLang to operations-research style optimization.

## Key Figures/Tables to Study
- Problem taxonomy: maps routing, scheduling, cache eviction, and admission control to missing formal models.
- Examples comparing FIFO/LRU-style policies with LLM-aware alternatives.

## Technical Details

### Why ordinary heuristics break
Generic web-serving policies assume requests have relatively fixed service costs and mostly independent memory footprints. LLM requests do not:
- prefill is compute-heavy and prompt-length dependent,
- decode is memory-bandwidth-heavy and output-length dependent,
- each active sequence grows its KV cache over time,
- continuous batching couples requests at every decode step.

### Course takeaway
This paper is useful late in the course after learners understand vLLM/SGLang mechanics. It explains why production inference is moving from "which engine is faster?" to "which scheduling and cache policy fits my workload and SLO?"

## Connections
- [[niyama]] — concrete QoS-driven scheduling system in this direction.
- [[admission-control-goodput]] — operational counterpart: count successful requests under latency SLOs.
- [[vllm-disaggregated-prefill-2026]] — real deployment case where phase asymmetry forced topology changes.
- [[sglang-hicache]] — cache hierarchy example where LRU alone is not enough.
