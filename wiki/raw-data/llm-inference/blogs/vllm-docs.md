<!-- scope: vLLM official documentation as a practitioner source
     see-also: vllm-project, vllm-benchmarks
-->

# vLLM Docs
- **Core Insight:** The vLLM docs are the operational map for running PagedAttention-based LLM serving in practice.
- **Guideline:** Use the docs for current CLI/API flags, but use papers and code paths for stable conceptual explanations.
- **Authors:** vLLM project
- **Year:** 2023-2026
- **URL:** https://docs.vllm.ai/
- **Relevant topics:** serving setup, OpenAI server, engine args, KV cache, prefix caching

## Abstract
The vLLM documentation covers installation, supported models, offline inference, OpenAI-compatible serving, engine arguments, distributed serving, quantization, structured outputs, and benchmarks. It is the most current practitioner reference for vLLM behavior.

## Key Contributions
- Documents concrete serving commands and engine configuration.
- Lists supported models, quantization backends, and deployment modes.
- Explains production features such as prefix caching, speculative decoding, and distributed serving.
- Links benchmark scripts and performance guidance.

## Key Figures/Tables to Study
- Serving docs: OpenAI-compatible server flags.
- Engine arguments reference: scheduler, memory, parallelism, and cache controls.
- Benchmark docs: official scripts and metric outputs.

## Technical Details
The docs change frequently as vLLM adds model support and engine features. When citing them, pin the concept and record the URL; when reproducing results, record package version, model revision, and command-line arguments.

## Connections
- [[vllm-project]] is the project summary.
- [[vllm-benchmarks]] is the benchmark source page.
