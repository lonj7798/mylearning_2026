<!-- scope: MLPerf Inference large language model benchmark tasks
     see-also: genai-perf, ttft-tpot-itl
-->

# MLPerf Inference LLM
- **Core Insight:** MLPerf Inference turns LLM serving into standardized benchmark scenarios with accuracy, latency, and throughput constraints.
- **Guideline:** Use MLPerf results for hardware/system comparisons, but read the scenario, precision, accuracy target, and latency constraint before applying numbers to chat serving.
- **Authors:** MLCommons
- **Year:** 2023-2026
- **URL:** https://mlcommons.org/benchmarks/inference-datacenter/
- **Relevant topics:** standardized benchmark, datacenter inference, Llama, latency constraints, accuracy target

## Abstract
MLPerf Inference is a standardized benchmark suite for datacenter and edge inference. Recent datacenter rounds include LLM workloads such as Llama-family text generation, evaluated under defined scenarios, quality targets, and latency constraints. Submissions document hardware, software stack, precision, and measured performance.

## Key Contributions
- Provides audited cross-vendor inference comparisons.
- Combines performance with accuracy or quality requirements.
- Defines scenarios such as Offline and Server that stress different deployment modes.
- Documents full system configurations, not only model-level results.

## Key Figures/Tables to Study
- MLPerf Inference result tables: system, accelerator count, scenario, and performance.
- Benchmark rules: latency constraints and quality targets for each LLM task.

## Technical Details
MLPerf Server is closest to online serving because it uses query arrivals and latency limits. Offline measures maximum throughput when all work is available up front. LLM submissions can use quantization or optimized kernels if they satisfy quality targets. The benchmark is less flexible than ShareGPT replay, but stronger for reproducible hardware comparisons.

## Connections
- [[nvidia-inference]] is often represented in MLPerf submissions.
- [[ttft-tpot-itl]] covers serving metrics that MLPerf-style summaries may aggregate differently.
