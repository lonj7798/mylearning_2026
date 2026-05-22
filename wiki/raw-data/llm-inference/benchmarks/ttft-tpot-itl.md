<!-- scope: latency metrics used to describe streaming LLM serving behavior
     see-also: sharegpt-workload, goodput-slo, genai-perf
-->

# TTFT, TPOT, and ITL Metrics
- **Core Insight:** LLM serving latency is multi-dimensional: prompt prefill determines time-to-first-token, decode determines per-token cadence, and users feel both.
- **Guideline:** Report TTFT, TPOT, inter-token latency, output throughput, and end-to-end latency separately instead of collapsing them into one average.
- **Authors:** NVIDIA GenAI-Perf, vLLM, Anyscale, Databricks MosaicML, broader serving community
- **Year:** 2023-2026
- **URL:** https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
- **Relevant topics:** latency metrics, streaming, prefill, decode, SLA/SLO measurement

## Abstract
TTFT measures how long a user waits before the first generated token appears. TPOT measures average decode time per emitted token, often excluding the first token. ITL measures the gap between consecutive streamed tokens and reveals jitter that averages can hide. These metrics map directly to the two phases of transformer inference: prompt prefill and autoregressive decode.

## Key Contributions
- Separates prompt-processing latency from token-generation cadence.
- Makes streaming UX measurable through first-token delay and token gaps.
- Explains why input length, output length, batch shape, and queueing must be reported with latency.
- Provides the vocabulary used by GenAI-Perf, vLLM benchmarks, LLMPerf, and production SLO dashboards.

## Key Figures/Tables to Study
- GenAI-Perf metric definitions: canonical benchmark terms for TTFT, inter-token latency, request latency, and throughput.
- vLLM benchmark output tables: practical examples of throughput and latency summaries under request-rate sweeps.

## Technical Details
TTFT includes request queueing, scheduling, tokenization if measured client-side, prompt prefill, and generation of the first token. TPOT is usually `(end_time - first_token_time) / (output_tokens - 1)`, so it emphasizes decode steady state. ITL records each adjacent token gap and should be summarized with percentiles, not just means. Benchmarks should disclose whether network time, tokenizer time, warmup, streaming transport, and failed requests are included.

## Connections
- [[sharegpt-workload]] supplies realistic prompt/output length distributions.
- [[goodput-slo]] turns latency metrics into SLO-qualified throughput.
- [[genai-perf]] and [[llmperf]] operationalize these measurements.
