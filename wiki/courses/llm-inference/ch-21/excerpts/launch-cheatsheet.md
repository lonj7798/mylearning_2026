---
chapter: ch-21
course: llm-inference
phase: read
excerpt_of: "vLLM 0.6.x + SGLang 0.3.x launch commands for the lab"
source_url: https://docs.vllm.ai/en/stable/configuration/serve_args/
created_at: "2026-05-21"
---

# Excerpt: vLLM + SGLang launch cheatsheet

**Source:** vLLM serve args docs + SGLang launch_server docs
**vLLM version:** 0.6.4
**SGLang version:** 0.3.7
**Raw-data source:** [[raw-data/vllm]], [[raw-data/sglang]]

---

## Install (separate venvs)

Both stacks have aggressive CUDA-kernel deps; do not co-install in one venv unless you enjoy debugging cuDNN. The cleanest pattern is two venvs.

```bash
# vLLM venv
python -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install --upgrade pip
pip install "vllm==0.6.4"
deactivate

# SGLang venv
python -m venv .venv-sgl
source .venv-sgl/bin/activate
pip install --upgrade pip
pip install "sglang[all]==0.3.7"
deactivate
```

To run both at once, activate one venv, start vLLM in the background, deactivate, activate the other, start SGLang. Each server is a separate process holding its own CUDA context.

---

## Full-budget launch (H100, Llama-3-8B-Instruct)

### vLLM (port 8001)

```bash
source .venv-vllm/bin/activate
HUGGING_FACE_HUB_TOKEN=hf_...
nohup vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  > vllm.log 2>&1 &
```

Tuning knobs and what they do (see [[vllm-scheduler]] for internals):

| Flag | Default | What it controls |
|------|---------|------------------|
| `--gpu-memory-utilization` | 0.90 | Fraction of GPU mem allocated to vLLM (weights + KV pool) |
| `--max-model-len` | model-card max | Hard cap on context (`prompt + output`) |
| `--max-num-seqs` | 256 | Max concurrent requests in flight |
| `--max-num-batched-tokens` | 8192 | Per-step token budget (prefill chunks + decode tokens combined) |
| `--enable-chunked-prefill` | True (V1) | Slice long prefills into multi-step chunks |
| `--enable-prefix-caching` | False | Turn on Automatic Prefix Cache (block-hash matching) |
| `--quantization` | none | `awq_marlin`, `gptq_marlin`, `fp8`, `bitsandbytes`, ... |

### SGLang (port 8002)

```bash
source .venv-sgl/bin/activate
nohup python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8002 \
  --dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --context-length 8192 \
  --max-running-requests 256 \
  --chunked-prefill-size 8192 \
  > sgl.log 2>&1 &
```

| Flag | Default | What it controls |
|------|---------|------------------|
| `--mem-fraction-static` | 0.88 | Equivalent to vLLM's `--gpu-memory-utilization` |
| `--context-length` | model-card max | Hard cap on context |
| `--max-running-requests` | 256 | Equivalent to `--max-num-seqs` |
| `--chunked-prefill-size` | 8192 | Per-step token budget |
| `--disable-radix-cache` | (off) | Toggle RadixAttention prefix cache |
| `--enable-torch-compile` | (off) | torch.compile path for the model graph |
| `--quantization` | none | `awq`, `gptq`, `fp8`, ... |

---

## Resource-constrained launch (single 24 GB GPU, Qwen-1.8B)

### vLLM

```bash
vllm serve Qwen/Qwen1.5-1.8B-Chat \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

### SGLang

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen1.5-1.8B-Chat \
  --port 8002 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --context-length 4096 \
  --max-running-requests 128 \
  --chunked-prefill-size 4096
```

Note: Qwen1.5-1.8B is a small dense model with GQA — no MoE, no thinking-mode flag. If you instead use Qwen2.5-3B, leave `enable_thinking=False` (default for the non-Qwen3 family is no thinking).

---

## Sanity check on a fresh server

After launch, before benchmarking, verify each server with a single request:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Write one sentence."}],
    "max_tokens": 32,
    "temperature": 0.0
  }'
```

Replace port 8001 → 8002 for SGLang. Both should return JSON in <1 s. If either hangs > 30 s, the server is still loading weights — `tail -f vllm.log` / `tail -f sgl.log` to watch.

---

## Stopping cleanly

```bash
# Best-effort shutdown
pkill -f "vllm serve"
pkill -f "sglang.launch_server"

# Force, if the above hangs
fuser -k 8001/tcp
fuser -k 8002/tcp
```

---

## Connections

- [[excerpts/benchmark-driver]] — once servers are up, drive them with `benchmark_serving.py`.
- [[ch-16]] / [[ch-17]] — internals of each scheduler that the launch flags actually steer.
