<!-- scope: vLLM 2026 disaggregated prefilling docs and MORI-IO single-node case study
     deps: [[prefill-vs-decode]], [[prefill-decode-disaggregation]], [[pagedattention]]
     see-also: [[distserve]], [[splitwise]], [[mooncake]], [[serving-optimization-foundations-2026]]
-->

# vLLM Disaggregated Prefilling and MORI-IO KV Transfer
- **Core Insight:** vLLM now supports running prefill and decode in separate instances connected by KV-transfer connectors, so deployments can tune TTFT and ITL independently and avoid prefill jobs disrupting decode cadence.
- **Guideline:** Use prefill/decode disaggregation when prefill bursts cause ITL spikes or when prefill and decode need different parallelism/hardware; measure goodput, not only aggregate token throughput.
- **Authors:** vLLM team, AMD, Embedded LLM contributors
- **Year:** 2026
- **URL:** https://docs.vllm.ai/en/v0.21.0/features/disagg_prefill/ ; https://vllm.ai/blog/2026-04-07-moriio-kv-connector
- **Relevant topics:** disaggregated prefilling, KV transfer, NIXL, MORI-IO, TTFT, ITL, goodput

## Abstract
vLLM's disaggregated prefilling mode separates the prompt-prefill phase and token-decode phase into different vLLM instances. A connector transfers KV cache blocks and metadata between the prefill and decode sides. The official docs describe connectors such as NixlConnector and OffloadingConnector, while the 2026 MORI-IO blog demonstrates single-node disaggregation on 8x AMD MI300X with Qwen3-235B-A22B-FP8, reporting 2.5x higher goodput under a latency SLO compared with colocated serving.

## Key Contributions
- Defines a production interface for transferring KV cache between prefill and decode instances.
- Lets prefill and decode use different tensor/pipeline parallel choices or even different hardware.
- Exposes connector abstractions: Connector, LookupBuffer, Pipe, scheduler connector, and worker connectors.
- Documents NIXL-style and MORI-IO/RDMA transfer paths.
- Reframes evaluation around ITL stability and goodput under SLO, not raw throughput alone.

## Key Figures/Tables to Study
- vLLM docs workflow diagram: prefill instance, decode instance, connector, and KV transfer.
- MORI-IO request-flow diagrams: read mode vs write mode.
- MORI-IO result table: Qwen3-235B-A22B-FP8, 8 req/s, 2000-token prompts, 1000-token outputs, 2.5x goodput headline.

## Technical Details

### Request flow
1. Proxy routes the prompt to a prefill instance.
2. Prefill computes prompt KV cache and returns or stores remote block metadata.
3. Decode instance obtains the KV cache through a connector.
4. Decode runs the memory-bandwidth-bound generation loop without being interrupted by large prefill GEMMs.

### Connector abstraction
The vLLM docs place disaggregated prefilling under `vllm/distributed/kv_transfer` and describe:
- `Connector`: retrieves KV caches from producer to consumer.
- `LookupBuffer`: inserts and blocking-selects matching KV entries.
- `Pipe`: FIFO tensor transmission.

### Trade-off
Disaggregation often improves ITL and SLO attainment, but it can worsen TTFT because KV transfer is now explicit. It is useful when decode stability matters more than the lowest possible first-token latency.

## Connections
- [[prefill-vs-decode]] — conceptual prerequisite.
- [[distserve]] / [[splitwise]] / [[mooncake]] — research systems that motivated this production feature.
- [[admission-control-goodput]] — best metric card for evaluating this mode.
- [[vllm-kv-offloading-connector]] — sibling KV-transfer/offload mechanism.
