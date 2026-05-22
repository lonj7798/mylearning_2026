<!-- scope: framework-level source page for vLLM serving architecture and APIs
     deps: [[pagedattention]], [[continuous-batching]]
     see-also: [[vllm-scheduler]], [[vllm-kv-cache-manager]], [[vllm-structured-output]]
-->

# vLLM
- **Core Insight:** vLLM turns PagedAttention, continuous batching, and an OpenAI-compatible frontend into a practical high-throughput LLM serving engine.
- **Guideline:** Use vLLM when GPU KV-cache efficiency and broad Hugging Face model serving matter more than owning a custom inference stack.
- **Authors:** vLLM project / UC Berkeley Sky Computing Lab and community
- **Year:** 2023-present
- **URL:** https://docs.vllm.ai/en/latest/ and https://github.com/vllm-project/vllm
- **Relevant topics:** OpenAI-compatible serving, PagedAttention, continuous batching, prefix caching, quantization, distributed inference

## Abstract
vLLM is an open-source inference and serving engine for LLMs and multimodal models. Its public surface includes an offline `LLM` API and an HTTP server launched with `vllm serve`, which implements OpenAI-compatible Chat Completions, Completions, Responses, embeddings, tokenizer, score, rerank, and other endpoints. Internally, vLLM separates frontend request handling from engine scheduling, KV-cache allocation, model execution, sampling, and output streaming.

## Key Contributions
- Exposes a familiar OpenAI-compatible API while preserving vLLM-specific controls through extra request parameters.
- Uses PagedAttention and a block-based KV cache so requests do not require large contiguous per-sequence allocations.
- Implements iteration-level scheduling with continuous batching, chunked prefill, preemption, and prefix-cache awareness.
- Supports production serving features such as LoRA adapters, quantization, speculative decoding, structured outputs, multimodal inputs, Ray Serve integration, and distributed execution.
- Provides code-level documentation for core modules, which makes scheduler and KV-cache behavior inspectable.

## Key Figures/Tables to Study
- `vllm serve` docs: public serving contract and supported APIs.
- `vllm/v1/core/sched/scheduler.py`: request admission, running/waiting queues, and token-budget decisions.
- `vllm/v1/core/kv_cache_manager.py`: block allocation, prefix caching, free/evict behavior, and cache events.
- `vllm/v1/worker` and `vllm/v1/engine`: boundary between scheduler output and model runner execution.

## Technical Details
Public APIs:
- Offline API: instantiate `vllm.LLM` and call generation methods from Python.
- Online API: run `vllm serve <model>`; clients can use the official OpenAI Python client with `base_url=http://host:port/v1`.
- Extra vLLM sampling and serving parameters are passed through OpenAI requests using `extra_body`.

Scheduler/cache approach:
- The scheduler builds each step from waiting and running requests under token, sequence, encoder, and KV-cache constraints.
- Prefill and decode share the same scheduler loop; chunked prefill lets long prompts consume only part of a step budget.
- KV cache is allocated in fixed-size blocks and referenced through block tables rather than one contiguous tensor per request.
- Automatic prefix caching can reuse already-computed prompt blocks when block hashes match.

Relevant code/docs:
- Server docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- Server args: https://docs.vllm.ai/en/stable/configuration/serve_args/
- Scheduler API docs: https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/
- KV-cache manager docs: https://docs.vllm.ai/en/latest/api/vllm/v1/core/kv_cache_manager/
- Source: https://github.com/vllm-project/vllm/tree/main/vllm

Strengths:
- Strong default for high-throughput GPU serving with minimal API migration from OpenAI clients.
- Large feature surface and model coverage compared with narrower engines.
- Paged KV cache makes long-context and multi-request workloads more memory efficient.

Limitations:
- Fast-moving internals mean code paths and tuning flags can change across releases.
- Best performance depends on hardware, attention backend, model architecture, quantization, and request mix.
- Custom model or kernel work requires understanding vLLM's engine, worker, and executor boundaries.

## Connections
- Foundation for [[vllm-scheduler]], [[vllm-kv-cache-manager]], and [[vllm-structured-output]].
- Implements the system ideas in [[pagedattention]] and [[continuous-batching]].
- Compare against [[sglang]], [[tensorrt-llm]], [[hf-tgi]], and [[llama-cpp-server]].
