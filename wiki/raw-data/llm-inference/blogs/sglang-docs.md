<!-- scope: SGLang official documentation as a practitioner source
     see-also: sglang-project, sglang-benchmarks
-->

# SGLang Docs
- **Core Insight:** SGLang docs connect language-model application structure directly to runtime optimizations such as RadixAttention and constrained decoding.
- **Guideline:** Use SGLang docs when teaching how workload structure can be exposed to the serving system.
- **Authors:** SGLang project
- **Year:** 2024-2026
- **URL:** https://docs.sglang.ai/
- **Relevant topics:** RadixAttention, structured outputs, runtime configuration, serving

## Abstract
The SGLang documentation covers installation, server launch, OpenAI-compatible APIs, structured outputs, frontend language features, runtime options, benchmark/profiling, and model support. It is the primary operational reference for the SGLang stack.

## Key Contributions
- Documents both frontend programming patterns and backend serving.
- Explains runtime features including prefix caching and constrained decoding.
- Provides benchmark and profiling workflows.
- Shows practical settings for high-throughput model serving.

## Key Figures/Tables to Study
- Runtime and backend docs: scheduling, cache, and server controls.
- Structured output docs: constrained decoding options.
- Benchmark/profiling page: commands and measured metrics.

## Technical Details
SGLang documentation is especially useful for workloads that are not independent one-shot chats. Shared prefixes, multi-step programs, and constrained outputs should be represented in the benchmark workload or the runtime advantage may be invisible.

## Connections
- [[sglang-project]] summarizes the system.
- [[sglang-benchmarks]] captures measurement workflow.
