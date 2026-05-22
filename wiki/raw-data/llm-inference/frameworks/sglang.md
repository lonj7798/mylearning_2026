<!-- scope: framework-level source page for SGLang serving architecture and APIs
     deps: [[continuous-batching]]
     see-also: [[sglang-scheduler]], [[sglang-radixattention]], [[sglang-structured-output]]
-->

# SGLang
- **Core Insight:** SGLang combines a frontend language for LLM programs with a high-performance runtime centered on RadixAttention and a fast scheduler.
- **Guideline:** Use SGLang when prompt/program structure, prefix reuse, structured outputs, and low-latency serving are primary concerns.
- **Authors:** SGLang project
- **Year:** 2023-present
- **URL:** https://docs.sglang.ai/ and https://github.com/sgl-project/sglang
- **Relevant topics:** RadixAttention, OpenAI-compatible API, native runtime API, continuous batching, chunked prefill, PD disaggregation, structured outputs

## Abstract
SGLang is an inference framework for LLMs and vision-language models. It offers OpenAI-compatible endpoints, a native `/generate` runtime API, an offline engine API, and an optional frontend language for programming multi-call LLM workflows. The backend runtime emphasizes prefix-cache reuse with RadixAttention, continuous batching, paged attention, chunked prefill, speculative decoding, and distributed serving.

## Key Contributions
- Introduces RadixAttention, which stores reusable KV-cache prefixes in a radix tree.
- Provides both OpenAI-compatible APIs and SGLang-native APIs for lower-level control.
- Supports structured outputs using JSON schema, regex, EBNF, and structural tags through grammar backends.
- Includes a high-performance serving runtime with scheduler, memory pool, radix cache, and model worker managers.
- Extends to production patterns such as model gateway/router, prefill/decode disaggregation, LoRA, quantization, metrics, tracing, and hierarchical KV caching.

## Key Figures/Tables to Study
- SGLang docs homepage: feature list and serving modes.
- OpenAI-compatible API docs: public serving contract.
- `python/sglang/srt/managers/scheduler.py`: runtime scheduler and request lifecycle.
- `python/sglang/srt/mem_cache/radix_cache.py`: prefix-cache tree.
- Structured-output docs: OpenAI and native examples.

## Technical Details
Public APIs:
- Launch server: `python -m sglang.launch_server --model-path <model>`.
- OpenAI-compatible API: `/v1/chat/completions`, completions, embeddings, and related endpoints.
- Native runtime API: `/generate` accepts text/input IDs plus `sampling_params`.
- Offline API: Python engine path for local inference without HTTP.

Scheduler/cache approach:
- Runtime scheduler admits queued requests into running batches subject to token, request, and memory constraints.
- RadixAttention matches prompt prefixes against a radix tree of cached KV blocks.
- Chunked prefill and continuous batching help mix prompt processing with decode steps.
- Optional hierarchical cache extends GPU-resident prefix reuse into host or external storage tiers.

Relevant code/docs:
- Docs: https://docs.sglang.ai/
- OpenAI APIs: https://docs.sglang.ai/basic_usage/openai_api.html
- Sampling/native API: https://docs.sglang.ai/basic_usage/sampling_params.html
- Source: https://github.com/sgl-project/sglang/tree/main/python/sglang

Strengths:
- Excellent fit for workloads with shared prompts, agent loops, few-shot templates, and structured decoding.
- Rich serving surface without giving up low-level runtime controls.
- Active support for modern inference features such as PD disaggregation and hierarchical KV caching.

Limitations:
- Runtime has many tuning flags; defaults may not be optimal for every workload.
- Feature behavior can depend on attention backend, grammar backend, model family, and hardware.
- The frontend-language concept is powerful but separate from the common OpenAI-compatible serving path.

## Connections
- Parent page for [[sglang-scheduler]], [[sglang-radixattention]], and [[sglang-structured-output]].
- Compare with [[vllm]], especially scheduler and prefix-cache design.
- RadixAttention is a contrasting answer to block-hash prefix caching in [[vllm-kv-cache-manager]].
