<!-- scope: NVIDIA GenAI-Perf benchmark tool and LLM serving metrics
     see-also: ttft-tpot-itl, goodput-slo
-->

# NVIDIA GenAI-Perf
- **Core Insight:** GenAI-Perf standardizes client-side measurement for generative AI endpoints, including streaming latency and token throughput.
- **Guideline:** Use GenAI-Perf when comparing deployed endpoints, and report its exact input/output length settings plus concurrency or request-rate mode.
- **Authors:** NVIDIA
- **Year:** 2024-2026
- **URL:** https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
- **Relevant topics:** benchmark tooling, Triton, TensorRT-LLM, TTFT, ITL, throughput

## Abstract
GenAI-Perf is part of NVIDIA's performance tooling for measuring generative AI models served through supported backends. It reports request latency, time to first token, inter-token latency, output token throughput, request throughput, and related summaries for LLM endpoints.

## Key Contributions
- Gives a repeatable CLI for endpoint-level LLM performance tests.
- Defines latency metrics that match streaming generation behavior.
- Integrates with NVIDIA serving stacks such as Triton and TensorRT-LLM.
- Supports synthetic input/output length control for reproducible sweeps.

## Key Figures/Tables to Study
- GenAI-Perf README metric table: definitions for TTFT, ITL, request latency, and throughput.
- CLI examples: how concurrency, prompts, and output lengths are configured.

## Technical Details
GenAI-Perf is a client benchmark, so its results include endpoint behavior as observed over the network path used in the test. Synthetic runs should sweep input length, requested output length, and concurrency/request rate. For streaming endpoints, TTFT and ITL are the key user-perceived metrics; for non-streaming endpoints, request latency and total output throughput dominate.

## Connections
- [[ttft-tpot-itl]] provides metric interpretation.
- [[tensorrt-llm-docs]] and [[nvidia-inference]] are common stacks measured with this tool.
