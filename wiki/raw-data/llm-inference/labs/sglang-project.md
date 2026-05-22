<!-- scope: SGLang project summary for structured and high-throughput LLM serving
     see-also: sglang-benchmarks, sglang-docs
-->

# SGLang Project
- **Core Insight:** SGLang co-designs a programming interface and runtime so structured generation, prefix reuse, and serving optimizations are visible to the system.
- **Guideline:** Study SGLang when workloads contain shared prefixes, multi-call programs, constrained decoding, or agentic control flow.
- **Authors:** SGLang team and community contributors
- **Year:** 2024-2026
- **URL:** https://docs.sglang.ai/
- **Relevant topics:** RadixAttention, structured generation, constrained decoding, prefix cache, OpenAI server

## Abstract
SGLang is an open-source serving system and programming framework for language-model applications. It includes an efficient backend runtime with RadixAttention, continuous batching, constrained decoding, tensor parallelism, and OpenAI-compatible APIs, plus frontend abstractions for multi-step generation programs.

## Key Contributions
- Uses RadixAttention to reuse KV cache across shared prompt prefixes.
- Integrates structured/constrained generation into the serving stack.
- Supports both programmatic SGLang workflows and standard API serving.
- Provides benchmark/profiling tooling for server performance.

## Key Figures/Tables to Study
- SGLang runtime docs: RadixAttention and cache reuse.
- Structured output docs: constrained decoding backends and performance implications.
- Benchmark docs: server launch and load-test commands.

## Technical Details
RadixAttention indexes reusable prefixes in a radix tree so requests with overlapping prompts can share cached KV states. This matters for few-shot prompts, agents, retrieval templates, and multi-turn workflows. SGLang's scheduler must balance prefix-cache hits, prefill work, decode steps, and constrained decoding overhead.

## Connections
- [[sglang-benchmarks]] covers performance measurement.
- [[deepseek-v3-inference]] and [[deepseek-r1-inference]] are common large-model serving targets.
