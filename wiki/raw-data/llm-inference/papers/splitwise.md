<!-- scope: phase-splitting paper for generative LLM inference
     deps: [[continuous-batching]]
     see-also: [[distserve]], [[mooncake]], [[prefill-decode-disaggregation]]
-->

# Splitwise: Efficient Generative LLM Inference Using Phase Splitting
- **Core Insight:** Prompt processing and token generation have distinct compute, memory, latency, and power profiles, so serving can improve by splitting phases across tailored resources.
- **Guideline:** Profile prefill and decode separately before choosing hardware, batching, or parallelism policy.
- **Authors:** Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Inigo Goiri, Saeed Maleki, Ricardo Bianchini
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2311.18677
- **Relevant topics:** phase splitting, prefill, decode, hardware specialization, energy efficiency, serving cost

## Abstract
Splitwise characterizes generative LLM inference as two main phases: compute-intensive prompt computation and memory-intensive token generation. Even with batching and scheduling, decode can underutilize compute resources. The paper proposes phase splitting as a way to allocate different resources and policies to each phase, improving efficiency, cost, and power behavior.

## Key Contributions
- Provides detailed characterization of prefill versus decode performance.
- Shows why token generation underutilizes GPU compute relative to prompt processing.
- Proposes splitting phases so each can use better-suited hardware and batching.
- Analyzes latency, throughput, memory, and power implications.
- Helps motivate later disaggregated serving systems.

## Key Figures/Tables to Study
- Phase characterization plots for compute utilization, memory behavior, and latency.
- Cost/power comparisons across hardware choices.
- Phase-splitting architecture diagrams.
- Sensitivity results under different prompt and output lengths.

## Technical Details
Prefill performs parallel computation across the prompt and tends to benefit from high compute throughput. Decode advances one token at a time per sequence and repeatedly reads weights plus growing KV cache, making it more memory-bandwidth sensitive.

Phase splitting introduces a state handoff: the prefill phase must transfer KV cache and request metadata to the decode phase. The efficiency gain depends on whether better phase-specific utilization exceeds the added communication and orchestration cost.

## Connections
- [[distserve]] operationalizes phase splitting as goodput-optimized disaggregation.
- [[mooncake]] makes KV cache the central object in a disaggregated design.
- [[sarathi-serve]] is a colocated alternative that interleaves phase work with chunking.

## Notes
Use this primarily as a characterization source for why phase-specific serving policies matter.
