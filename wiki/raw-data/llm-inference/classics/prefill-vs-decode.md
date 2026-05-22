<!-- scope: Two-phase autoregressive inference model: prompt prefill then token decode
     deps: [[attention-is-all-you-need]], [[kv-cache-memory-formula]]
     see-also: [[batching-for-inference]], [[openai-streaming-and-token-usage]]
-->

# Prefill vs Decode
- **Core Insight:** LLM inference has two different phases: prefill processes the input prompt in parallel and writes KV cache; decode generates one token at a time using that cache.
- **Guideline:** Optimize and measure TTFT and TPOT separately; prefill is often compute-heavy, while decode is often memory-bandwidth and scheduling limited.
- **Authors:** Amey Agrawal et al. (SARATHI); Yinmin Zhong et al. (DistServe)
- **Year:** 2023 / 2024
- **URL:** https://arxiv.org/abs/2308.16369 ; https://arxiv.org/abs/2401.09670
- **Relevant topics:** prefill, decode, TTFT, TPOT, chunked prefill, disaggregated serving

## Abstract
Serving papers such as SARATHI and DistServe formalize the split between prefill and decode. Prefill ingests the prompt and creates KV-cache state. Decode repeatedly consumes that state to produce output tokens. The phases have different bottlenecks, latency metrics, batching behavior, and placement strategies.

## Key Contributions
- Names the two major inference phases and their different performance profiles.
- Connects prefill to time-to-first-token (TTFT).
- Connects decode to time-per-output-token (TPOT) or inter-token latency.
- Motivates chunked prefill, hybrid batches, and prefill/decode disaggregation.
- Explains why long prompts and long outputs stress different resources.

## Key Figures/Tables to Study
- **SARATHI overview:** Chunked prefills plus decode-maximal batching.
- **DistServe architecture:** Separate prefill and decode workers.
- **Latency breakdown plots:** TTFT vs TPOT under mixed workloads.
- **Pipeline-bubble figures:** Prefill/decode imbalance creates scheduling inefficiency.

## Technical Details
Prefill:

```text
input: prompt tokens length S
work: process all S positions, compute attention over prompt, write K,V for each layer
latency metric: TTFT
```

Decode:

```text
input: previous generated token plus KV cache
work: one forward step per output token, append one K,V entry per layer
latency metric: TPOT / ITL
```

Chunked prefill splits a long prompt into smaller chunks so decode work can be interleaved. Disaggregation routes prefill and decode to different GPU pools so their resource plans do not interfere.

## Connections
- [[kv-cache-memory-formula]]: prefill allocates and populates cache; decode grows it token by token.
- [[batching-for-inference]]: continuous batching schedules prefill and decode requests together.
- [[attention-complexity]]: full prompt attention makes long prefill expensive.
- [[openai-streaming-and-token-usage]]: streaming exposes decode tokens as they arrive after prefill finishes.
