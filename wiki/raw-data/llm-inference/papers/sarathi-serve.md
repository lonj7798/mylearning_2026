<!-- scope: Sarathi-Serve scheduler for chunked prefill and stall-free LLM serving
     deps: [[continuous-batching]]
     see-also: [[distserve]], [[splitwise]], [[prefill-decode-disaggregation]]
-->

# Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
- **Core Insight:** Split long prefills into chunks and schedule them with decodes so new requests can enter without stalling ongoing generation.
- **Guideline:** Use chunked prefill when long prompts inflate TTFT or disrupt decode latency under continuous batching.
- **Authors:** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2403.02310
- **Relevant topics:** chunked prefill, stall-free scheduling, TTFT, TPOT, decode-maximal batching, latency-throughput tradeoff

## Abstract
Sarathi-Serve observes that prefill and decode have different latency and utilization profiles: prefill is compute-heavy and high latency, while decode is short per step but memory/batch sensitive. Naive batching can make long prefills block ongoing decodes or force poor throughput. Sarathi-Serve uses chunked-prefills and stall-free schedules to admit requests while continuing decode progress, improving the tradeoff between throughput and per-token latency.

## Key Contributions
- Extends Sarathi's chunked-prefill idea to online LLM serving.
- Creates stall-free schedules that avoid pausing ongoing decodes when new prefills arrive.
- Uses chunk size as a control knob between TTFT and throughput.
- Studies interaction among prefill, decode, batching, and pipeline utilization.
- Shows improved latency-throughput behavior versus conventional schedulers.

## Key Figures/Tables to Study
- Timeline diagrams contrasting full prefill blocking versus chunked-prefill interleaving.
- Chunk-size sensitivity plots for TTFT, TPOT, and throughput.
- Scheduler pseudocode/architecture for stall-free admission.
- Evaluation under mixed prompt/output length distributions.

## Technical Details
Chunked prefill splits a prompt into multiple token chunks. Each chunk advances the request's KV cache for part of the prompt; once all chunks finish, the request can decode normally. Because each chunk is bounded, the scheduler can place decode iterations between prefill chunks rather than letting one long prompt monopolize the GPU.

The scheduler aims to keep decode traffic moving while using prefill chunks to saturate compute. KV cache grows as chunks are processed, so admission still depends on memory budget. Smaller chunks improve responsiveness but can reduce compute efficiency due to more scheduling and kernel overhead; larger chunks improve throughput but risk decode stalls.

## Connections
- Builds on [[continuous-batching]] but changes the unit of prefill scheduling.
- Motivates colocated alternatives to disaggregation in [[distserve]] and [[splitwise]].
- A key technique for [[admission-control-goodput]] because chunk size affects SLO feasibility.

## Notes
Sarathi-Serve is the online-serving successor to the earlier SARATHI chunked-prefill paper.
