---
chapter: ch-18
course: llm-inference
phase: read
excerpt_of: "llama.cpp + llama-server — portable CPU/Apple-Silicon-first LLM inference"
source_url: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
created_at: "2026-05-21"
---

# Excerpt: llama.cpp + llama-server — the portability stack

**Authors:** ggml-org / Georgi Gerganov + llama.cpp community
**Year:** 2023–present
**URLs:** https://github.com/ggml-org/llama.cpp / https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
**Raw-data source:** [[raw-data/llama-cpp-server]]

---

## The architectural bet

**C/C++ portability + gguf model format + slot-based batching + every backend that exists.**

This is the most-deployed LLM inference engine in the world by user count: Ollama, LM Studio, GPT4All, koboldcpp, MSTY, Jan, and dozens of other apps wrap llama.cpp.

---

## Backends supported

```
GGML compute graph compiles to one of:
  • CPU              — AVX2, AVX-512, AVX-VNNI (x86), NEON (ARM)
  • CUDA / cuBLAS    — NVIDIA
  • Metal            — Apple Silicon (M1/M2/M3/M4)
  • HIP / ROCm       — AMD
  • SYCL             — Intel Arc, Intel GPU
  • Vulkan           — cross-vendor (works on almost any GPU)
  • OpenCL           — legacy
  • RPC              — distribute layers to remote backend
  • Kompute          — Vulkan compute, alternative path
```

A single `.gguf` file runs unchanged across every backend the runtime supports.

---

## The slot model

`llama-server` exposes N slots (concurrent sequences). Each slot has its own state + KV cache portion. Requests are assigned to free slots; the server batches per-slot decode work into one `llama_decode` call.

```bash
./llama-server \
    -m Llama-3-8B-Instruct-Q4_K_M.gguf \
    --port 8080 \
    --ctx-size 16384 \              # total context across slots
    --parallel 8 \                  # 8 slots
    --cont-batching \               # continuous batching across slots
    --threads 8 \                   # CPU threads
    --n-gpu-layers 33 \             # offload first 33 layers to GPU
    --metrics
```

Without `--cont-batching`, slots exist but don't batch with each other — you get per-slot throughput, not aggregated.

---

## The OpenAI surface

```
POST /v1/chat/completions          # OpenAI Chat Completions
POST /v1/completions               # OpenAI Completions
POST /v1/embeddings                # OpenAI Embeddings
GET  /v1/models                    # list available models
```

Plus llama.cpp native extras:

```
POST /completion                   # full native generation control
POST /tokenize, /detokenize        # tokenizer access
GET  /slots                        # introspect slot state (debug)
GET  /health, /metrics             # ops
POST /apply-template               # apply chat template only
POST /reranking                    # reranking via embedding model
```

---

## The gguf format

A single self-describing binary container:

```
GGUF file layout:
  • magic "GGUF" + version
  • metadata key-value pairs:
       general.architecture = "llama"
       llama.context_length = 8192
       llama.embedding_length = 4096
       llama.block_count = 32
       llama.attention.head_count = 32
       llama.attention.head_count_kv = 8       # GQA
       tokenizer.ggml.tokens = [...]
       tokenizer.chat_template = "..."         # Jinja2 chat template
  • tensor info (name, shape, dtype, offset)
  • tensor data (quantized weights, packed per gguf quant scheme)
```

The runtime reads the gguf, infers the architecture from `general.architecture`, applies the matching model code path, and runs. No per-model loader code needed if the architecture is supported.

See ch-08 (model-quantization course ch-19) for the k-quant + IQ-quant ladder.

---

## Slot caching — `--slot-save-path`

```bash
./llama-server \
    -m model.gguf \
    --slot-save-path /tmp/slot_kv \   # idle-slot KV save location
    --parallel 4
```

When a slot becomes idle, its KV state can be saved to disk and restored later. For chat applications where the same conversation may resume hours later, this avoids recomputing the prefix.

---

## Speculative decoding

```bash
./llama-server \
    -m Llama-3-70B-Instruct-Q4_K_M.gguf \
    -md Llama-3-1B-Instruct-Q4_K_M.gguf \    # draft model
    --draft 16                                # speculate up to 16 tokens
```

n-gram speculation also supported via `--draft-min` / `--draft-max` flags — uses prompt n-grams as drafts, no separate model required (related to prompt-lookup decoding, ch-15).

---

## Real performance numbers — Llama-3-8B Q4_K_M

| Backend | Tokens/sec (decode, batch=1) |
|---------|------------------------------:|
| Apple M2 Max (Metal) | 35–45 |
| Apple M3 Max (Metal) | 45–60 |
| Apple M4 Pro (Metal) | 55–75 |
| NVIDIA RTX 4090 (CUDA) | 100–140 |
| AMD Ryzen 9 7950X (CPU, 32 threads) | 15–22 |
| Intel Core Ultra 7 (Vulkan iGPU) | 8–15 |
| Raspberry Pi 5 (CPU) | 1–3 |

For multi-tenant high-QPS: nowhere near vLLM. llama.cpp wins on portability + edge, not on cluster throughput.

---

## Strengths

- The only credible portable option (CPU + Metal + every GPU vendor)
- gguf is self-describing → easy model loading
- OpenAI-compatible out of the box
- Active community + dozens of frontend apps wrapping it

## Limits

- Throughput per request and concurrent-request count nowhere near vLLM
- Slot/context sizing constraints can be confusing
- Feature support varies by backend (e.g. some quants don't work on Metal)
- Multi-tenant scheduling is basic compared to GPU-first stacks

---

## Connections

- [[excerpts/gguf-k-quants]] (model-quantization ch-19) — the quant format llama.cpp consumes.
- [[excerpts/continuous-batching]] (ch-04) — the batching primitive applied at slot level.
- [[ch-18]] — parent framework decision rubric.
