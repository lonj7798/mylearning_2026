<!-- scope: LLMPerf benchmark suite for hosted and self-hosted LLM APIs
     see-also: ttft-tpot-itl, goodput-slo
-->

# LLMPerf
- **Core Insight:** LLMPerf treats LLM serving as an API workload where latency, throughput, and generated token counts must be measured together.
- **Guideline:** Use LLMPerf-style tests to compare API-serving configurations, but control prompt/output length and concurrency before comparing providers or runtimes.
- **Authors:** Anyscale
- **Year:** 2023
- **URL:** https://github.com/ray-project/llmperf
- **Relevant topics:** API benchmark, concurrency, latency, throughput, hosted endpoints

## Abstract
LLMPerf is an open benchmark suite for evaluating LLM API performance across providers and self-hosted deployments. It issues concurrent requests, collects token counts and latency metrics, and reports throughput and percentile summaries.

## Key Contributions
- Makes hosted LLM API performance measurable from the client side.
- Supports comparisons across providers and OpenAI-compatible endpoints.
- Emphasizes token-normalized throughput and latency rather than request counts alone.
- Provides scripts that can be adapted to production-like load tests.

## Key Figures/Tables to Study
- LLMPerf GitHub README: benchmark setup and reported metrics.
- Test runner code: concurrency model and request timing behavior.

## Technical Details
Client-side API benchmarks are sensitive to region, network path, retries, rate limits, prompt text, requested output length, and streaming mode. LLMPerf is useful for black-box systems, while engine-native benchmarks such as vLLM throughput tests isolate server internals. Results should include percentile latency, prompt tokens/sec, output tokens/sec, and error rate.

## Connections
- [[anyscale-llm-serving]] provides production serving context from the same ecosystem.
- [[ttft-tpot-itl]] explains the latency metrics.
