<!-- scope: official SGLang benchmark methodology and benchmark scripts
     see-also: sharegpt-workload, sglang-project, goodput-slo
-->

# SGLang Benchmarks
- **Core Insight:** SGLang benchmarks emphasize serving throughput and latency for structured generation, prefix reuse, and high-throughput OpenAI-compatible serving.
- **Guideline:** Benchmark both generic ShareGPT serving and workload-specific SGLang programs when prefix sharing or constrained decoding matters.
- **Authors:** SGLang project
- **Year:** 2023-2026
- **URL:** https://docs.sglang.ai/references/benchmark_and_profiling.html
- **Relevant topics:** serving benchmark, RadixAttention, prefix cache, structured generation, request-rate sweep

## Abstract
SGLang documents benchmark and profiling flows for server throughput, latency, and specialized workloads. Its serving benchmarks resemble vLLM-style online tests, while the project also highlights workloads where RadixAttention, constrained decoding, and frontend/backend co-design change performance.

## Key Contributions
- Provides scripts for measuring OpenAI-compatible serving performance.
- Highlights prefix-cache-sensitive workloads, not only independent chat requests.
- Connects benchmark results to runtime features such as RadixAttention and continuous batching.
- Encourages profiling to identify backend, kernel, or scheduling bottlenecks.

## Key Figures/Tables to Study
- SGLang benchmark/profiling docs: command patterns and profiling targets.
- SGLang runtime docs: RadixAttention and scheduling features to vary during benchmark runs.

## Technical Details
Important benchmark variables include request rate, dataset, model, tokenizer, maximum running requests, chunked prefill settings, radix cache state, tensor parallelism, and constrained decoding backend. For agentic or structured workloads, repeated prefixes can dominate performance, so cache hit rate should be reported with latency and throughput.

## Connections
- [[sglang-project]] describes the runtime features behind the benchmark numbers.
- [[sharegpt-workload]] covers the common chat replay workload.
- [[goodput-slo]] applies latency constraints to SGLang capacity.
