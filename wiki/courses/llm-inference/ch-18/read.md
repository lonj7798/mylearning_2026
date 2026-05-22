<!-- chapter: ch-18
     track: framework-internals
     title: Production Serving Stacks — TensorRT-LLM, TGI, LightLLM, LMDeploy, DeepSpeed-FastGen, llama.cpp
     sources: [[tensorrt-llm]], [[tensorrt-llm-paged-kv]], [[hf-tgi]], [[lightllm]], [[lmdeploy]], [[deepspeed-fastgen]], [[llama-cpp-server]], [[nvidia-inference]], [[huggingface-inference]]
     figures: figures/framework-decision-matrix.html
-->

# Chapter 18 — Production Serving Stacks: TensorRT-LLM, TGI, LightLLM, LMDeploy, DeepSpeed-FastGen, llama.cpp

> **Core insight.** vLLM and SGLang are not the only options. The 2025 LLM serving landscape has at least seven production stacks, each making a different bet about what to optimize first: NVIDIA's TensorRT-LLM optimizes for *maximum-throughput on NVIDIA SKUs* through ahead-of-time engine compilation + FP8/NVFP4 + Triton; HuggingFace TGI optimizes for *deployment ergonomics + ecosystem fit* through a Rust router + Python shards + first-class HF integration; LightLLM optimizes for *token-level cache accuracy + Python readability*; LMDeploy optimizes for *FasterTransformer-lineage performance on InternLM/Qwen + persistent-batch semantics*; DeepSpeed-FastGen contributed *Dynamic SplitFuse* (the chunked-prefill ancestor) and bridges DeepSpeed-trained models; llama.cpp wins everywhere there is no GPU.
>
> **Guideline.** Match the stack to the hardware and the ops culture. NVIDIA cloud + max throughput → **TensorRT-LLM** (accept engine-build overhead). HuggingFace-shaped org → **TGI** (zero-friction deployment, lose ~10–20 % to vLLM). Edge / Apple Silicon / CPU → **llama.cpp llama-server**. InternLM/Qwen + want OpenAI server → **LMDeploy**. Research scheduler / token-level cache → **LightLLM**. DeepSpeed-trained model with kernel-injection → **DeepSpeed-FastGen** (now maintenance mode, treat as scheduling-design reference). Don't pick by benchmark numbers alone — the right framework is the one your team can debug at 3 a.m.

---

## Why this chapter exists

After vLLM (ch-16) and SGLang (ch-17), the natural question is: why do other frameworks even exist? The answer is that "best serving framework" is workload- and org-conditional, and the gap between "vLLM works fine" and "we picked the wrong stack" is real. NVIDIA-only orgs want TensorRT-LLM's FP8 path. HuggingFace-shaped teams want TGI's Docker-first ergonomics. Edge / consumer deployments live or die on llama.cpp's CPU + Metal support. Chinese frontier labs (DeepSeek, InternLM, Qwen) ship LMDeploy alongside vLLM/SGLang because of model-architecture-specific kernel tuning.

Three things to walk away with:

1. **The architectural diversity is real, not aesthetic.** TGI's Rust router + Python shards is a different deployment surface than vLLM's monolithic Python engine. TensorRT-LLM's ahead-of-time engine build is a different operational model than vLLM's JIT model load.
2. **Each framework's distinctive contribution.** TensorRT-LLM's gptAttentionPlugin, TGI's safetensors-aware router, LightLLM's TokenAttention, LMDeploy's TurboMind persistent batch, DeepSpeed-FastGen's Dynamic SplitFuse, llama.cpp's slot-based batching.
3. **The decision rubric.** A concrete matrix of (hardware × workload × ops culture) → recommended framework, with the rationale.

This chapter pulls from [[tensorrt-llm]], [[tensorrt-llm-paged-kv]], [[hf-tgi]], [[lightllm]], [[lmdeploy]], [[deepspeed-fastgen]], [[llama-cpp-server]], [[nvidia-inference]], and [[huggingface-inference]].

---

## 1. TensorRT-LLM — NVIDIA's compiled-engine path

[[tensorrt-llm]] (NVIDIA, 2023–present) is the NVIDIA-blessed LLM serving stack. Its central architectural bet: **compile the model to a TensorRT engine ahead of time**, sacrificing flexibility for kernel-level optimization that JIT systems cannot match.

### 1.1 Architecture

```
HF checkpoint   ─┐
                 ↓
              convert_checkpoint.py   →  TRT-LLM intermediate format
                                          (model config + quantized weights)
                 ↓
              trtllm-build   →  TensorRT engine (.engine file, SKU-specific)
                 ↓
              ModelRunner (Python) / executor (C++) / Triton backend  →  inference
```

The `.engine` file is a serialized TensorRT graph compiled for one specific (model, GPU SKU, max-batch, max-seq-len, precision) tuple. Rebuild if any of those changes. The compile is slow (5–30 min for a 70B model) but produces tight kernels.

### 1.2 The features that make it fast

| Feature | What it does |
|---------|--------------|
| **In-flight batching** | Continuous batching: requests join/leave the active batch each iteration (ch-04 equivalent) |
| **Paged KV cache** | Block-based KV storage; same idea as vLLM PagedAttention. See [[tensorrt-llm-paged-kv]] |
| **KV reuse + offload + prioritized eviction** | Prefix-cache analogue + CPU offload for cold blocks |
| **`gptAttentionPlugin`** | Custom CUDA kernel that fuses MHA + paged KV + KV-cache quant + rotary embedding |
| **FP8 / NVFP4 native** | Tensor cores consume FP8 (Hopper) / NVFP4 (Blackwell) directly via Transformer Engine |
| **Triton integration** | First-class backend for NVIDIA Triton Inference Server (dynamic batching, metrics, gRPC) |

### 1.3 Config — `KvCacheConfig` + build args

```python
from tensorrt_llm.runtime import ModelRunner, KvCacheConfig

# KV-cache configuration (runtime)
kv_config = KvCacheConfig(
    max_tokens=524288,                # total KV slot budget
    sink_token_length=4,              # attention-sink window (ch-08)
    enable_block_reuse=True,          # prefix cache
    onboard_blocks=True,              # CPU offload
    cross_kv_cache_fraction=0.0,      # cross-attn fraction (encoder-decoder)
    kv_cache_dtype="fp8",             # KV quant
)

runner = ModelRunner.from_dir(
    "./engine",
    rank=0,
    kv_cache_config=kv_config,
)
```

Engine build:

```bash
trtllm-build \
    --checkpoint_dir ./converted_llama_3_8b_fp8 \
    --output_dir ./engine \
    --gemm_plugin fp8 \
    --gpt_attention_plugin fp8 \
    --use_paged_context_fmha enable \
    --max_batch_size 64 \
    --max_input_len 8192 \
    --max_seq_len 16384 \
    --use_fp8_context_fmha enable
```

### 1.4 The Triton serving layer

```
Client (gRPC / HTTP)  →  Triton  →  TRT-LLM backend (C++ executor)
                              ↓
                         Dynamic batching, metrics, model versioning
```

Production NVIDIA deployments run TRT-LLM under Triton, not bare. Triton supplies the production ergonomics TRT-LLM itself doesn't (auth, observability, model lifecycle).

### 1.5 Strengths and limits

**Strengths.** Highest absolute throughput on NVIDIA hardware. FP8/NVFP4 done right. Tightly integrated with NVIDIA's Transformer Engine. MLPerf submissions almost always TRT-LLM-based.

**Limits.** Engine-build operational overhead. SKU-specific binaries (H100 engine won't run on B100 optimally). Less portable than PyTorch-first stacks. Custom model support requires conversion work (not "just load a HuggingFace checkpoint").

---

## 2. HuggingFace TGI — ecosystem-fit production server

[[hf-tgi]] (Hugging Face, 2022–present) is the production serving counterpart to the Transformers library. Architectural bet: **Rust router for high-concurrency request handling + Python model shards for flexibility + Docker-first ops**.

### 2.1 Architecture — router + shards

```
┌─────────────────────────────────────────────────────────────┐
│  text-generation-launcher (CLI)                              │
│   • Spawns shard processes (one per GPU under TP)            │
│   • Spawns router (Rust)                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Router (Rust, axum HTTP server)                             │
│   • /generate, /generate_stream                              │
│   • /v1/chat/completions (OpenAI compat)                     │
│   • Request validation (max_input_length, max_total_tokens)  │
│   • Continuous batching: assembles batches from waiting queue│
│   • Forwards to model shards via gRPC                        │
└─────────────────────────────────────────────────────────────┘
                            ↓ gRPC
┌─────────────────────────────────────────────────────────────┐
│  Model shard processes (Python, one per TP rank)             │
│   • Loads weights via safetensors + transformers              │
│   • Runs PagedAttention (FlashInfer / vLLM kernels)          │
│   • Streams generated tokens back to router                  │
└─────────────────────────────────────────────────────────────┘
```

The Rust router is the key engineering choice: it handles HTTP / TLS / streaming at low overhead, leaving Python free to focus on model execution. Most other frameworks (vLLM, SGLang) put the HTTP layer in Python.

### 2.2 Launch and config

```bash
text-generation-launcher \
    --model-id meta-llama/Llama-3-8B-Instruct \
    --num-shard 1 \
    --port 8080 \
    --max-input-length 4096 \
    --max-total-tokens 8192 \
    --max-batch-prefill-tokens 16384 \
    --max-batch-total-tokens 32768 \
    --quantize bitsandbytes-nf4 \
    --trust-remote-code
```

Docker image is the standard deployment:

```bash
docker run --gpus all -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:2.4.0 \
    --model-id meta-llama/Llama-3-8B-Instruct
```

### 2.3 Strengths and limits

**Strengths.** Best ecosystem fit if you already use HuggingFace Hub + safetensors + transformers. First-class Docker image. Production-quality metrics, health, OpenAPI. Hugging Face Inference Endpoints (managed) is TGI under the hood.

**Limits.** Generally 10–20 % slower than vLLM on the same workload (router overhead + Python shard runtime). Scheduler internals less central as a learning artifact. Advanced research features (e.g. SGLang's `fork()`, vLLM's speculative-decoding flags) tend to lag.

---

## 3. LightLLM — Python-first, Token Attention, transparent scheduler

[[lightllm]] (ModelTC + community, 2023–present) is a Python-first serving framework focused on **TokenAttention** — accurate token-level KV cache accounting — and an **efficient Router** that prevents OOM under burst load.

### 3.1 The TokenAttention idea

Where vLLM allocates KV in 16-token blocks (16-token granularity wastes ≤15 tokens at the tail of each request), LightLLM accounts KV at **per-token granularity** with explicit free-list management. Lower waste; simpler memory pool; more conservative admission so OOM never happens.

### 3.2 Architecture

```
lightllm/server/api_openai.py           # OpenAI-compatible HTTP
lightllm/server/api_server.py           # native HTTP
lightllm/server/router/                 # request scheduling / token accounting
lightllm/common/kv_cache_mem_manager/   # specialized cache managers
    • mem_manager.py                    # default token-level
    • int8_kv_cache_mem_manager.py      # INT8 KV
    • int8_quant_mem_manager.py         # int8 weight + KV
    • cpu_cache_manager.py              # CPU offload
lightllm/models/                        # per-architecture model code
```

### 3.3 Launch

```bash
python -m lightllm.server.api_server \
    --model_dir meta-llama/Llama-3-8B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --tp 1 \
    --max_total_token_num 240000 \
    --batch_max_tokens 16384 \
    --running_max_req_size 256 \
    --mode triton_int8kv
```

### 3.4 Strengths and limits

**Strengths.** Python codebase is approachable (good if you want to read router/cache behavior). Specialized cache managers for INT8 KV, CPU offload, model-specific layouts. Useful comparison point for token-vs-block-vs-radix cache designs.

**Limits.** Smaller ecosystem and adoption than vLLM/SGLang/TGI. Less feature breadth (no speculative decoding parity, fewer attention backend choices). Production suitability depends on model/hardware specifics.

---

## 4. LMDeploy — TurboMind persistent batch + PyTorch backend

[[lmdeploy]] (InternLM / OpenMMLab, 2023–present) is a serving toolkit with **two backends**: TurboMind (C++/CUDA, FasterTransformer lineage) and a PyTorch backend with explicit paging scheduler.

### 4.1 TurboMind — the FasterTransformer descendant

TurboMind preallocates a fixed-size batch ("persistent batch") at server startup. Requests fill slots; finished slots become available for new requests. This is a stricter model than vLLM's dynamic continuous batching, but it avoids per-iteration scheduling overhead and matches the FasterTransformer execution model many production teams already operate.

Key components:

- **Persistent batch executor** — fixed batch size for the server lifetime
- **KV cache manager** — memory pool + LRU cache for reusable conversation KV
- **LLaMA / InternLM / Qwen architectures** — explicitly tuned per-model kernels

### 4.2 PyTorch backend

For models where TurboMind doesn't have a hand-tuned implementation:

```
lmdeploy/pytorch/paging/scheduler.py        # paged scheduler
lmdeploy/pytorch/engine/cache_engine.py     # cache engine
lmdeploy/pytorch/kernels/cuda/pagedattention.py  # paged attention kernel
```

### 4.3 Launch

```bash
# TurboMind backend (default for supported architectures)
lmdeploy serve api_server meta-llama/Llama-3-8B-Instruct \
    --server-port 23333 \
    --tp 1 \
    --cache-max-entry-count 0.85 \
    --quant-policy 4

# PyTorch backend
lmdeploy serve api_server meta-llama/Llama-3-8B-Instruct \
    --backend pytorch \
    --server-port 23333
```

### 4.4 Strengths and limits

**Strengths.** Best performance on InternLM, Qwen, and other Chinese-frontier-lab models. FasterTransformer lineage gives mature C++ kernel infrastructure. Persistent-batch model is simpler operationally than continuous batching.

**Limits.** Narrower model coverage. TurboMind features depend on per-architecture implementation. Ecosystem smaller than vLLM/TGI for arbitrary HF checkpoints.

---

## 5. DeepSpeed-FastGen — Dynamic SplitFuse, now historical

[[deepspeed-fastgen]] (Microsoft DeepSpeed team, 2023–2024) introduced **Dynamic SplitFuse**: split long prompts into chunks and fuse them with decode tokens from other requests in the same forward pass.

This is the conceptual ancestor of **chunked prefill** (Sarathi-Serve, ch-05) and is now standard in vLLM, SGLang, TGI, LMDeploy, and TensorRT-LLM. FastGen itself is in maintenance mode but the design contribution is permanent.

### 5.1 The SplitFuse mechanism

```
Time t:    Batch = [decode_req1, decode_req2, prefill_req3_chunk_1_of_4]
Time t+1:  Batch = [decode_req1, decode_req2, prefill_req3_chunk_2_of_4]
Time t+2:  Batch = [decode_req1, decode_req2, prefill_req3_chunk_3_of_4, decode_req4]
...
```

Long prefills don't monopolize the GPU; decode latency stays bounded. This is the entire point of chunked prefill.

### 5.2 Why FastGen matters historically

DeepSpeed-FastGen + Sarathi-Serve (parallel works in 2023) showed that **prefill bursts are the dominant cause of decode latency tail** even with continuous batching. The fix — interleave at sub-prefill granularity — is now default in every serious serving stack.

### 5.3 Practical note

Don't deploy FastGen as a primary serving stack in 2026. Use vLLM/SGLang/TRT-LLM, which all have chunked prefill built-in. Treat FastGen as a scheduling-design reference + a fallback for orgs already invested in DeepSpeed training that want compatible inference.

---

## 6. llama.cpp + llama-server — the CPU + Apple Silicon stack

[[llama-cpp-server]] is the most-deployed LLM inference engine *in the world by user count* — every Ollama, LM Studio, GPT4All, and koboldcpp user is running llama.cpp underneath. Architectural bet: **C/C++ portability, gguf model format, slot-based batching, every CPU/GPU/NPU backend that exists**.

### 6.1 Backends

```
GGML compute graph  →  one of:
  • CPU (AVX2, AVX-512, NEON)
  • CUDA / cuBLAS (NVIDIA)
  • Metal (Apple Silicon)
  • HIP / ROCm (AMD)
  • SYCL (Intel Arc, Intel GPU)
  • Vulkan (cross-vendor)
  • OpenCL (legacy)
  • RPC (distribute to remote backend)
```

A single `.gguf` file runs unchanged across every backend the runtime is compiled for.

### 6.2 Server slots — the batching model

`llama-server` exposes N "slots" (parallel sequences), each owning a portion of the KV cache:

```bash
llama-server -m Llama-3-8B-Instruct-Q4_K_M.gguf \
    --port 8080 \
    --ctx-size 16384 \
    --parallel 8 \              # 8 slots
    --cont-batching \           # continuous batching across slots
    --metrics \
    --threads 8 \
    --n-gpu-layers 33           # offload first 33 layers to GPU
```

Each HTTP request gets assigned to a free slot; the server batches the per-slot decode work into one llama.cpp `llama_decode` call.

### 6.3 The OpenAI surface

`llama-server` is OpenAI-compatible:

```
/v1/chat/completions
/v1/completions
/v1/embeddings
/v1/models
```

Plus native llama.cpp extras:

```
/completion              # native generation control
/tokenize, /detokenize   # tokenizer access
/slots                   # introspect slot state
/health, /metrics        # ops
```

### 6.4 The gguf format

A single binary container holding model weights + architecture metadata + tokenizer + chat template + quantization scales. Self-describing: the runtime reads the gguf, infers the architecture, applies the right kernels. No per-model code needed if the architecture is supported.

See [[gguf-k-quants]] (ch-08 / model-quantization course) for the quant ladder (q2_K, q4_K_M, q8_0, IQ-quants).

### 6.5 Why it matters

For *anything not on a GPU cluster*, llama.cpp is the answer:

- MacBook (M1/M2/M3/M4) — Metal backend; Llama-3-8B Q4_K_M at ~25–40 tokens/sec
- Linux laptop with iGPU — Vulkan backend
- Raspberry Pi / Jetson — CPU + maybe CUDA
- Mixed CPU+GPU offload — first N layers on CPU, rest on GPU (`--n-gpu-layers`)
- Embedded / IoT — CPU-only with int4 quant

No other framework competes on portability + ease of compilation.

### 6.6 Limits

Throughput per request and concurrent-request count are nowhere near vLLM. llama.cpp wins on coverage, not on raw throughput. For multi-tenant high-QPS serving, use vLLM/SGLang on actual GPUs.

---

## 7. Decision rubric — who picks what

| Workload + ops constraint | Pick | Why |
|---------------------------|------|-----|
| Max throughput on NVIDIA H100/B100, willing to do engine builds | **TensorRT-LLM** + Triton | FP8/NVFP4 path + Transformer Engine; MLPerf-grade |
| OSS flexibility, willing to read Python source | **vLLM** | Largest ecosystem, fastest catch-up to new models |
| Shared-prefix / agent / RAG workloads | **SGLang** | RadixAttention + `fork()` (see ch-17) |
| HuggingFace-shaped org, Docker ops | **TGI** | Zero-friction Hub integration; Rust router; managed Inference Endpoints |
| Edge / consumer / Apple Silicon / CPU | **llama.cpp llama-server** | Only portable option; gguf + slot batching |
| InternLM / Qwen / persistent-batch ops | **LMDeploy** | TurboMind kernels tuned for these architectures |
| Research scheduler experimentation in Python | **LightLLM** | Token-level cache accounting; readable codebase |
| DeepSpeed-trained model, kernel injection wanted | **DeepSpeed-FastGen** | Compatibility with DeepSpeed-Inference (note: maintenance mode) |

### 7.1 The anti-recommendations

- **Don't pick by single-shape benchmark.** vLLM beats SGLang on no-overlap workloads, SGLang beats vLLM on shared-prefix workloads. Match to *your* traffic.
- **Don't pick TRT-LLM if your team can't operate engine-build pipelines.** The performance win is real but the operational cost is real too.
- **Don't pick TGI just because you use HF Transformers.** vLLM accepts HF safetensors checkpoints too, often faster.
- **Don't pick llama.cpp for multi-tenant GPU serving.** It's the wrong tool — use it for edge.
- **Don't pick FastGen for greenfield deployment.** Use vLLM/SGLang which absorbed the Dynamic SplitFuse idea.

---

## 8. Cross-framework feature matrix

| Feature | vLLM | SGLang | TRT-LLM | TGI | LightLLM | LMDeploy | llama.cpp |
|---------|:----:|:------:|:-------:|:---:|:--------:|:--------:|:---------:|
| Continuous batching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (slots) |
| Chunked prefill | ✓ | ✓ | ✓ | partial | partial | ✓ | — |
| Paged KV | ✓ | ✓ | ✓ | ✓ (via vLLM kernels) | ✓ (token-level) | ✓ | — (slot-based) |
| Prefix cache | ✓ (APC) | ✓ (RadixAttn) | ✓ (reuse) | partial | partial | ✓ (LRU) | — |
| FP8 weights | ✓ | ✓ | ✓ | ✓ | partial | ✓ | via gguf q8 |
| FP8 KV cache | ✓ | ✓ | ✓ | partial | ✓ (INT8) | ✓ | — |
| NVFP4 (Blackwell) | partial | partial | ✓ | — | — | — | — |
| Speculative decoding | ✓ | ✓ | ✓ | partial | — | ✓ | ✓ |
| Structured output | ✓ (xgrammar) | ✓ (xgrammar/outlines/llguidance) | partial | partial | — | partial | ✓ (gbnf) |
| LoRA serving | ✓ | ✓ | ✓ | ✓ | partial | ✓ | partial |
| TP + PP + EP | ✓ | ✓ | ✓ | TP only | TP | ✓ | TP only |
| Multi-LoRA hot-swap | ✓ | ✓ | partial | ✓ | partial | ✓ | — |
| CPU offload | ✓ | ✓ (HiCache L2) | ✓ | partial | ✓ | partial | ✓ (native CPU) |
| OpenAI API | ✓ | ✓ | via Triton | ✓ | ✓ | ✓ | ✓ |

`partial` = supported but with feature gaps or behind a flag.

---

## 9. Practitioner's cheat sheet

```bash
# vLLM (already covered in ch-16)
vllm serve meta-llama/Llama-3-8B-Instruct --enable-prefix-caching --max-num-seqs 256

# SGLang (covered in ch-17)
python -m sglang.launch_server --model meta-llama/Llama-3-8B-Instruct --schedule-policy lpm

# TensorRT-LLM — three-stage: convert + build + run
python convert_checkpoint.py --model_dir Meta-Llama-3-8B --output_dir converted --dtype bfloat16
trtllm-build --checkpoint_dir converted --output_dir engine \
             --gemm_plugin bfloat16 --max_batch_size 64 --max_input_len 4096
mpirun -n 1 python run.py --engine_dir engine --max_output_len 256 --tokenizer_dir Meta-Llama-3-8B

# TGI (Docker)
docker run --gpus all -p 8080:80 ghcr.io/huggingface/text-generation-inference:2.4.0 \
    --model-id meta-llama/Llama-3-8B-Instruct --max-batch-prefill-tokens 16384

# LightLLM
python -m lightllm.server.api_server \
    --model_dir meta-llama/Llama-3-8B-Instruct --tp 1 \
    --max_total_token_num 240000 --batch_max_tokens 16384

# LMDeploy TurboMind
lmdeploy serve api_server meta-llama/Llama-3-8B-Instruct --tp 1 --server-port 23333

# DeepSpeed-FastGen
python -m mii.server --model meta-llama/Llama-3-8B-Instruct --tensor-parallel 1

# llama.cpp
./llama-server -m Llama-3-8B-Instruct-Q4_K_M.gguf --ctx-size 16384 \
    --parallel 8 --cont-batching --n-gpu-layers 33 --port 8080
```

---

## Common pitfalls

- **Switching frameworks expecting linear feature parity.** Each framework names things differently: vLLM's "max-num-seqs" ≈ TGI's "max-concurrent-requests" ≈ TRT-LLM's `max_batch_size`. Read the docs; defaults are not portable.
- **TRT-LLM engine built for wrong SKU.** An H100 engine ≠ an H200 engine; rebuild per target hardware.
- **TGI without `--max-batch-prefill-tokens` tuned.** Default may be too low; prefill latency spikes hide behind it.
- **LightLLM without `--max_total_token_num` raised.** Default is conservative; throughput leaves a lot on the table.
- **LMDeploy TurboMind on an unsupported architecture.** Falls back to PyTorch silently; you pay TurboMind's startup cost for none of the kernel benefit.
- **llama.cpp `--parallel` set without `--cont-batching`.** Slots exist but no batching across them; you get per-slot throughput, not aggregated.
- **Comparing benchmarks across frameworks without matching SLOs.** Throughput numbers without latency targets are not comparable — see ch-19.

---

## Connections and what's next

- **Back: [[cuda-graphs-inference]] / ch-12** — all GPU frameworks use CUDA graphs differently; TRT-LLM is engine-level, vLLM uses piecewise capture, SGLang uses CUDA-graph capture per discrete batch size.
- **Back: [[vllm]] / ch-16** — direct comparison; vLLM is the OSS reference, TRT-LLM the NVIDIA reference.
- **Back: [[sglang]] / ch-17** — the prefix-cache specialist; complementary to TRT-LLM's raw throughput.
- **Back: [[continuous-batching]] / ch-04** + **[[sarathi-serve]] / ch-05** — the scheduling primitives every framework here implements.
- **Back: model-quantization course ch-19** — quantization-kernel choice intersects framework choice (Marlin/Machete in vLLM; FP8 in TRT-LLM; gguf in llama.cpp).
- **Forward: ch-19 (benchmarks)** — the only way to actually compare these frameworks for your workload.
- **Forward: ch-20 (production model reports)** — model cards specify recommended frameworks (Llama 3 → vLLM/TRT-LLM; DeepSeek → SGLang/vLLM; Mixtral → TRT-LLM/vLLM).
- **Forward: ch-21 (lab)** — vLLM vs SGLang head-to-head benchmark.

## Further reading

- [[tensorrt-llm]] / [[tensorrt-llm-paged-kv]] — NVIDIA's stack + paged KV details.
- [[hf-tgi]] / [[huggingface-inference]] — TGI + the HF inference ecosystem.
- [[lightllm]] — TokenAttention + transparent Python scheduler.
- [[lmdeploy]] — TurboMind persistent batch + FasterTransformer lineage.
- [[deepspeed-fastgen]] — Dynamic SplitFuse, the chunked-prefill ancestor.
- [[llama-cpp-server]] — gguf + slot batching for edge.
- [[nvidia-inference]] — NVIDIA ecosystem overview (TRT-LLM + Triton + NIM + GenAI-Perf).

## Companion visualization

**[figures/framework-decision-matrix.html](figures/framework-decision-matrix.html)** — interactive matrix: pick (workload type, hardware, ops constraint) and see which framework is recommended + why, with the rationale highlighted. Sliders for batch size, shared-prefix ratio, latency SLO, and GPU SKU; each combination lights up the matching cell in the decision rubric.
