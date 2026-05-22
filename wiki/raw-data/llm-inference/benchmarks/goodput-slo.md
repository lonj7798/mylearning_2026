<!-- scope: goodput measurement for LLM serving under latency SLOs
     see-also: ttft-tpot-itl, vllm-benchmarks, sglang-benchmarks
-->

# Goodput and SLOs
- **Core Insight:** Maximum throughput is not useful if requests miss latency objectives; goodput counts only work completed within SLOs.
- **Guideline:** Report goodput with explicit TTFT, TPOT, and end-to-end latency thresholds when sizing production serving clusters.
- **Authors:** vLLM project, LMSYS/academic serving literature, production inference community
- **Year:** 2023-2026
- **URL:** https://docs.vllm.ai/en/latest/contributing/benchmarks.html
- **Relevant topics:** SLO, goodput, latency constraints, capacity planning, request-rate sweep

## Abstract
Goodput is the rate of successful work that satisfies service-level objectives. For LLMs, SLOs commonly constrain time to first token, time per output token, inter-token latency, or full request latency. Goodput exposes the real serving capacity before queueing and tail latency make the system unacceptable.

## Key Contributions
- Converts latency percentiles into an admission/capacity planning metric.
- Penalizes overload regimes where token throughput remains high but users wait too long.
- Supports direct comparison of scheduler policies under production latency targets.
- Encourages request-rate sweeps to find the sustainable operating point.

## Key Figures/Tables to Study
- vLLM benchmark docs and examples: request-rate sweeps and latency outputs.
- Production SLO dashboards: plots of request rate versus p95 TTFT/TPOT and goodput.

## Technical Details
Define success before running the benchmark: e.g. TTFT below 2 seconds, p95 TPOT below 100 ms/token, and no errors. Goodput can be expressed as requests/sec, output tokens/sec, or completed tokens/sec that meet all SLO predicates. Because output length changes request duration, goodput should be reported with token distributions and not only request counts.

## Connections
- [[ttft-tpot-itl]] defines the latency predicates.
- [[sharegpt-workload]] provides realistic request mixes for goodput testing.
