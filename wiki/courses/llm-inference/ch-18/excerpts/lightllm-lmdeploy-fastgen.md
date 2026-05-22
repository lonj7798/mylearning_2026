---
chapter: ch-18
course: llm-inference
phase: read
excerpt_of: "LightLLM + LMDeploy + DeepSpeed-FastGen — three more production serving stacks"
source_url: https://github.com/ModelTC/lightllm + https://github.com/InternLM/lmdeploy + https://github.com/deepspeedai/DeepSpeed
created_at: "2026-05-21"
---

# Excerpt: LightLLM / LMDeploy / DeepSpeed-FastGen — the second-tier production stacks

**Authors:** ModelTC (LightLLM) / InternLM (LMDeploy) / Microsoft DeepSpeed (FastGen)
**Year:** 2023–2024
**Raw-data sources:** [[raw-data/lightllm]] + [[raw-data/lmdeploy]] + [[raw-data/deepspeed-fastgen]]

---

## LightLLM — TokenAttention + readable Python

**Architectural bet:** token-granularity KV accounting + Python-first scheduler that prevents OOM by construction.

```
lightllm/server/api_openai.py           # OpenAI-compatible HTTP
lightllm/server/api_server.py           # native HTTP
lightllm/server/router/                 # request scheduling, token accounting
lightllm/common/kv_cache_mem_manager/   # specialized cache managers
   • mem_manager.py                     # default token-level
   • int8_kv_cache_mem_manager.py       # INT8 KV
   • int8_quant_mem_manager.py          # int8 weight + KV
   • cpu_cache_manager.py               # CPU offload
   • per_model_mem_manager.py           # architecture-specific layouts
lightllm/models/                        # per-architecture model code
```

**TokenAttention.** Where vLLM allocates KV in 16-token blocks (wasting ≤15 tokens at the tail of each request), LightLLM tracks KV at per-token granularity. Lower waste + more conservative admission ⇒ OOM by construction never happens.

**Launch:**

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

**Strengths:** approachable Python codebase for router/cache reading; specialized cache managers for INT8 KV + CPU offload; useful comparison to block/page/radix designs.

**Limits:** smaller ecosystem and adoption than vLLM/SGLang/TGI; feature breadth narrower; smaller community.

---

## LMDeploy — TurboMind + PyTorch backends

**Architectural bet:** two backends — TurboMind (C++/CUDA, FasterTransformer descendant, persistent batch) and PyTorch (paged scheduler + paged attention).

### TurboMind backend

Preallocates a fixed-size batch at server startup. Requests fill slots; finished slots become available for new requests. Stricter than vLLM's continuous batching but avoids per-iteration scheduling overhead.

Components:
- Persistent batch executor — fixed batch size for server lifetime
- KV cache manager — memory pool + LRU cache for reusable conversation KV
- LLaMA / InternLM / Qwen architectures — explicitly tuned per-model kernels

### PyTorch backend

For models without TurboMind support:

```
lmdeploy/pytorch/paging/scheduler.py        # paged scheduler
lmdeploy/pytorch/engine/cache_engine.py     # cache engine
lmdeploy/pytorch/kernels/cuda/pagedattention.py  # paged attention kernel
```

**Launch:**

```bash
# TurboMind (default for supported architectures)
lmdeploy serve api_server meta-llama/Llama-3-8B-Instruct \
    --server-port 23333 --tp 1 \
    --cache-max-entry-count 0.85 \
    --quant-policy 4                # INT4 KV

# PyTorch backend
lmdeploy serve api_server meta-llama/Llama-3-8B-Instruct \
    --backend pytorch --server-port 23333
```

**Strengths:** best performance on InternLM, Qwen, and Chinese-frontier-lab models; FasterTransformer lineage = mature C++ kernel infrastructure; persistent-batch model is operationally simple.

**Limits:** narrower model coverage; TurboMind features depend on per-arch implementation; ecosystem smaller than vLLM/TGI for arbitrary HF checkpoints.

---

## DeepSpeed-FastGen — Dynamic SplitFuse, historical importance

**Architectural bet (2023):** split prompt prefills into chunks and fuse with decode tokens from other requests.

### The mechanism

```
Time t:    Batch = [decode_req1, decode_req2, prefill_req3_chunk_1_of_4]
Time t+1:  Batch = [decode_req1, decode_req2, prefill_req3_chunk_2_of_4]
Time t+2:  Batch = [decode_req1, decode_req2, prefill_req3_chunk_3_of_4, decode_req4]
```

Long prefills don't monopolize the GPU; decode latency stays bounded.

### Why it matters

DeepSpeed-FastGen + Sarathi-Serve (parallel 2023 works) showed prefill bursts dominate decode-latency tail even with continuous batching. The fix — interleave at sub-prefill granularity — is now default in vLLM, SGLang, TGI, LMDeploy, TensorRT-LLM. This is "chunked prefill" in current terminology (ch-05).

### Current status

FastGen itself is in maintenance mode. Don't deploy as a primary stack in 2026. Treat as:
- Scheduling-design reference for understanding chunked prefill origins
- Fallback for orgs already invested in DeepSpeed training that want compatible inference

### Launch (MII server, FastGen path)

```bash
python -m mii.server \
    --model meta-llama/Llama-3-8B-Instruct \
    --tensor-parallel 1 \
    --port 28080
```

---

## Choosing between them

| Use case | Pick |
|----------|------|
| Research scheduler experimentation in Python | LightLLM |
| Token-level KV accounting + INT8 KV cache | LightLLM |
| InternLM / Qwen production deployment | LMDeploy TurboMind |
| FasterTransformer-style persistent batch ops | LMDeploy TurboMind |
| DeepSpeed-trained model with kernel injection | DeepSpeed-FastGen |
| Anything greenfield in 2026 | Probably vLLM or SGLang, not these |

---

## Connections

- [[excerpts/sarathi-serve]] (ch-05) — the chunked-prefill formalization of FastGen's SplitFuse idea.
- [[excerpts/vllm-kv-cache-manager]] (ch-16) — the block-based KV alternative to LightLLM's token-level.
- [[excerpts/continuous-batching]] (ch-04) — what all three implement variants of.
- [[ch-18]] — parent decision rubric.
