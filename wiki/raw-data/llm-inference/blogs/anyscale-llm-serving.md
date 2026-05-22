<!-- scope: Anyscale LLM serving guidance and production benchmark context
     see-also: llmperf, goodput-slo
-->

# Anyscale LLM Serving
- **Core Insight:** Anyscale frames LLM serving as distributed application serving with autoscaling, batching, endpoint SLOs, and cost-performance tradeoffs.
- **Guideline:** Use Anyscale material for production serving concepts and LLMPerf context, while validating specific runtime claims with current benchmarks.
- **Authors:** Anyscale
- **Year:** 2023-2026
- **URL:** https://www.anyscale.com/blog/continuous-batching-llm-inference
- **Relevant topics:** continuous batching, Ray Serve, LLMPerf, autoscaling, production SLO

## Abstract
Anyscale's LLM serving posts and tooling explain production inference through Ray Serve, continuous batching, request scheduling, autoscaling, and performance benchmarking. The ecosystem includes LLMPerf for endpoint benchmarking and Ray-based deployment patterns.

## Key Contributions
- Explains continuous batching for practitioners.
- Connects serving-engine behavior to distributed application deployment.
- Provides benchmark tooling for API-level comparisons.
- Emphasizes cost, latency, and throughput tradeoffs in production.

## Key Figures/Tables to Study
- Continuous batching blog diagrams: why iteration-level batching improves utilization.
- LLMPerf repository: benchmark runner and metrics.
- Ray Serve LLM docs: deployment and autoscaling patterns.

## Technical Details
Ray Serve can orchestrate model replicas and route traffic, while the model worker may use vLLM or another optimized backend. Autoscaling decisions need SLO metrics, not just GPU utilization. Queueing delay can dominate TTFT at high load even if per-token decode remains fast.

## Connections
- [[llmperf]] is the related benchmark suite.
- [[goodput-slo]] captures SLO-aware capacity planning.
