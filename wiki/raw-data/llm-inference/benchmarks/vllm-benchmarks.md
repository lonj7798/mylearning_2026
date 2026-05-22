<!-- scope: official vLLM benchmark scripts and metrics
     see-also: sharegpt-workload, ttft-tpot-itl, vllm-project
-->

# vLLM Benchmarks
- **Core Insight:** vLLM benchmarking distinguishes offline engine throughput from online serving latency under request arrival rates.
- **Guideline:** Choose `benchmark_throughput.py` for engine capacity and `benchmark_serving.py` for user-facing SLO behavior.
- **Authors:** vLLM project
- **Year:** 2023-2026
- **URL:** https://docs.vllm.ai/en/latest/contributing/benchmarks.html
- **Relevant topics:** benchmark scripts, serving latency, throughput, ShareGPT, request-rate sweep

## Abstract
vLLM provides benchmark entry points for offline throughput, serving endpoints, latency, prefix caching, and model-specific scenarios. The serving benchmark can replay ShareGPT and synthetic workloads against OpenAI-compatible servers while reporting request throughput, token throughput, TTFT, TPOT, and ITL percentiles.

## Key Contributions
- Separates offline throughput from online serving measurements.
- Supports multiple serving backends through OpenAI-compatible APIs.
- Includes realistic and synthetic dataset modes.
- Reports latency percentiles that map to streaming user experience.
- Makes request-rate sweeps practical for finding saturation points.

## Key Figures/Tables to Study
- `benchmarks/benchmark_serving.py`: online serving workload driver and metric computation.
- `benchmarks/benchmark_throughput.py`: offline engine throughput path without HTTP overhead.
- vLLM docs benchmark examples: command-line flags for dataset, request rate, model, tokenizer, and backend.

## Technical Details
Serving experiments should pin model, tokenizer, tensor parallelism, max model length, GPU type, precision, scheduler knobs, and request-rate distribution. The same model can look strong in offline tokens/sec but fail online p95 TTFT or TPOT targets once queueing appears. vLLM benchmark results are most useful when paired with latency SLOs and goodput.

## Connections
- [[vllm-project]] explains the runtime being benchmarked.
- [[sharegpt-workload]] supplies common benchmark traces.
- [[goodput-slo]] adds SLO-qualified capacity analysis.
