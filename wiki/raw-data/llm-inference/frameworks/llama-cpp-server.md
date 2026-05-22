<!-- scope: llama.cpp llama-server API, batching, slots, and KV-cache behavior
     deps: [[continuous-batching]]
     see-also: [[hf-tgi]], [[vllm]]
-->

# llama.cpp Server
- **Core Insight:** `llama-server` brings llama.cpp's portable GGML/GGUF inference runtime to an HTTP server with slots, continuous batching, and OpenAI-compatible endpoints.
- **Guideline:** Use `llama-server` for local/edge CPU-GPU serving and quantized GGUF models, not as a drop-in replacement for high-throughput datacenter GPU engines.
- **Authors:** ggml-org / llama.cpp community
- **Year:** 2023-present
- **URL:** https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- **Relevant topics:** GGUF, quantization, OpenAI-compatible API, slots, continuous batching, KV cache, speculative decoding

## Abstract
`llama-server` is the HTTP server included with llama.cpp. It exposes a web UI, native completion endpoints, OpenAI-compatible chat/completions/responses/embeddings routes, schema-constrained JSON, function calling, monitoring endpoints, multimodal support, speculative decoding, continuous batching, and parallel decoding with multi-user support.

## Key Contributions
- Makes local quantized GGUF inference accessible through familiar HTTP APIs.
- Uses server slots to represent concurrent sequences/request state.
- Supports continuous batching and parallel decoding across slots.
- Provides cache controls such as idle-slot caching and slot save/restore features.
- Runs across CPU and multiple accelerator backends supported by llama.cpp.

## Key Figures/Tables to Study
- `tools/server/README.md`: server features, routes, and CLI flags.
- `tools/server/README-dev.md`: slot abstraction and batching internals.
- Server code under `tools/server`: request routing, slots, metrics, and OpenAI compatibility.
- llama.cpp core KV-cache code: interaction between slots and `llama_decode`.

## Technical Details
Public APIs:
- Launch with `llama-server -m <model.gguf>` plus context, backend, batching, and server flags.
- Use OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, responses, and embeddings routes.
- Native `/completion` endpoint exposes llama.cpp-specific generation controls.

Scheduler/cache approach:
- Each server slot manages one sequence-like request state.
- The server batches compatible slot work and calls llama.cpp decode routines.
- KV cache is allocated according to context and parallelism settings; slots share total context capacity.
- Idle slot caching and save/restore can preserve prompt/KV state for repeated interactions.

Relevant code/docs:
- Server README: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Server developer README: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md
- Server source: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- Main repo: https://github.com/ggml-org/llama.cpp

Strengths:
- Excellent portability and local deployment story.
- GGUF quantization makes consumer and edge hardware practical.
- API compatibility lets many clients switch from hosted APIs with minimal changes.

Limitations:
- Throughput and multi-tenant scheduling are not comparable to GPU-first cluster engines.
- Slot/context sizing is a practical constraint and can be confusing under high concurrency.
- Feature behavior depends heavily on backend, model architecture, and compile flags.

## Connections
- Good contrast with [[vllm]], [[sglang]], and [[hf-tgi]] because portability is the main design center.
- Slot batching connects to [[continuous-batching]].
- Structured JSON support connects to [[vllm-structured-output]] and [[sglang-structured-output]].
