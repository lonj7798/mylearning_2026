<!-- scope: vLLM asynchronous KV offloading connector for CPU-backed cache and larger concurrent workloads
     deps: [[kv-cache-memory-formula]], [[vllm-kv-cache-manager]]
     see-also: [[cachegen]], [[sglang-hicache]], [[vllm-disaggregated-prefill-2026]]
-->

# vLLM KV Offloading Connector
- **Core Insight:** vLLM's newer connector API can asynchronously load and store KV cache outside GPU memory, letting CPU RAM act as a larger cache tier without blocking the engine's main scheduling loop.
- **Guideline:** Use KV offloading when GPU KV capacity causes preemption/recompute or when repeated long prefixes can be served from CPU cache faster than recomputing prefill.
- **Authors:** vLLM team
- **Year:** 2026
- **URL:** https://vllm.ai/blog/2026-01-08-kv-offloading-connector
- **Relevant topics:** KV offload, CPU cache, connector API, prefix reuse, preemption, throughput

## Abstract
The vLLM KV offloading connector extends the connector API to support asynchronous loading and storing of KV data. Earlier connector paths were synchronous, which blocked the engine while external KV transfer happened. The new offloading connector includes a CPU backend and reports large TTFT and throughput gains when cached prefixes can be loaded from CPU memory rather than recomputed, while also helping overloaded systems avoid recomputing preempted requests.

## Key Contributions
- Makes external KV loading/storing asynchronous so serving can continue while transfers occur.
- Provides a native CPU backend for larger KV capacity than GPU HBM.
- Improves behavior under preemption: evicted KV can be reloaded instead of recomputed.
- Exposes a pluggable backend API for future storage tiers.
- Reports throughput increases as CPU KV cache hit rate rises.

## Key Figures/Tables to Study
- Single-request TTFT figures: loading CPU KV vs recomputing prefill.
- Concurrent throughput figures: throughput vs CPU cache hit rate.
- Connector API diagram: where offload/load hooks integrate with request lifecycle.

## Technical Details

### Why CPU KV cache helps
GPU HBM is fast but small. CPU RAM is slower but much larger and often enough for prefix reuse or preemption recovery. If the prompt is long, loading previously computed KV can beat recomputing prefill.

### CLI/API surface
The blog describes the newer simple CLI shape:
```
--kv_offloading_backend native --kv_offloading_size <size_in_GB>
```
and older `--kv-transfer-config` forms using `OffloadingConnector`.

### Operational caveat
Offload is workload-dependent. It helps most when cache hit rate is high, prompts are long, or preemptions would otherwise force recomputation. It can add transfer pressure when hits are rare.

## Connections
- [[sglang-hicache]] — SGLang's broader GPU/CPU/distributed cache hierarchy.
- [[cachegen]] — compression/offload research direction.
- [[vllm-disaggregated-prefill-2026]] — connector API also underlies prefill/decode disaggregation.
- [[kv-cache-memory-formula]] — explains why KV cache capacity becomes the bottleneck.
