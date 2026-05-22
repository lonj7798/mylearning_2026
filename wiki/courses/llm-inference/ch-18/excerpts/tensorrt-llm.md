---
chapter: ch-18
course: llm-inference
phase: read
excerpt_of: "TensorRT-LLM — NVIDIA's compiled-engine LLM inference stack"
source_url: https://nvidia.github.io/TensorRT-LLM/
created_at: "2026-05-21"
---

# Excerpt: TensorRT-LLM — ahead-of-time engine + Transformer Engine + Triton

**Authors:** NVIDIA
**Year:** 2023–present
**URLs:** https://nvidia.github.io/TensorRT-LLM/ / https://github.com/NVIDIA/TensorRT-LLM
**Raw-data source:** [[raw-data/tensorrt-llm]] + [[raw-data/tensorrt-llm-paged-kv]]

---

## The architectural bet

**Compile the model to a TensorRT engine ahead of time, sacrifice flexibility for kernel-level optimization JIT systems can't match.**

```
HF / Megatron / NeMo checkpoint
              ↓
         convert_checkpoint.py
              ↓
   TRT-LLM intermediate format (config + quantized weights)
              ↓
         trtllm-build
              ↓
   .engine file (SKU-specific, serialized TensorRT graph)
              ↓
   ModelRunner (Python) / executor (C++) / Triton backend
              ↓
              client
```

The compile takes 5–30 min for a 70B model. Rebuild required if model, max-batch, max-seq-len, precision, or GPU SKU changes.

---

## Key features

| Feature | What it enables |
|---------|-----------------|
| **In-flight batching (IFB)** | Continuous batching — requests join/leave each iteration |
| **Paged KV cache** | Block-based KV (16- or 64-token blocks); reuse + offload + prioritized eviction |
| **`gptAttentionPlugin`** | Fused MHA + paged KV access + KV-cache quant + rotary embedding in one CUDA kernel |
| **Transformer Engine FP8/NVFP4** | Tensor cores consume FP8 (Hopper) and NVFP4 (Blackwell) natively |
| **Multi-shot speculative decoding** | Medusa / EAGLE / lookahead supported at engine level |
| **Tensor + pipeline + expert parallel** | Multi-GPU + multi-node via NCCL |
| **KV cache events** | Observability hook for cache state (block birth/death) |

---

## `KvCacheConfig` — runtime config

```python
from tensorrt_llm.runtime import ModelRunner, KvCacheConfig

kv_config = KvCacheConfig(
    max_tokens=524288,                # total KV slot budget
    free_gpu_memory_fraction=0.85,    # alternative to max_tokens
    sink_token_length=4,              # attention-sink window (ch-08)
    enable_block_reuse=True,          # prefix cache
    onboard_blocks=True,              # async copy to GPU as needed
    cross_kv_cache_fraction=0.0,      # encoder-decoder cross-attn share
    kv_cache_dtype="fp8",             # KV quant
    secondary_offload_min_priority=50, # CPU offload threshold
)

runner = ModelRunner.from_dir("./engine", rank=0, kv_cache_config=kv_config)
```

---

## Engine build command — `trtllm-build`

```bash
trtllm-build \
    --checkpoint_dir ./converted_llama_3_8b_fp8 \
    --output_dir ./engine \
    --gemm_plugin fp8 \
    --gpt_attention_plugin fp8 \
    --use_paged_context_fmha enable \
    --use_fp8_context_fmha enable \
    --max_batch_size 64 \
    --max_input_len 8192 \
    --max_seq_len 16384 \
    --max_num_tokens 16384 \
    --tp_size 1 \
    --pp_size 1
```

`--max_num_tokens` is the chunked-prefill bound (ch-05 equivalent).

---

## Kernel selection by quant mode

| Mode | Backend kernel |
|------|----------------|
| FP16 / BF16 | cuBLASLt + custom attention |
| FP8 (E4M3) | Transformer Engine FP8 GEMM (WGMMA path on Hopper) |
| INT8 SmoothQuant | `smoothQuant.cu` plugin |
| INT4 AWQ | `awq_ext` AWQ GEMM |
| INT4 GPTQ | Marlin-style internal kernel |
| NVFP4 | Blackwell block-scaled tensor-core GEMM |
| KV cache (FP8 / INT8) | Fused into `gptAttentionPlugin` (no dequant before softmax) |

---

## The Triton serving layer

TRT-LLM as a Triton backend is the production deployment:

```bash
# Triton model repo layout
models/
└── llama_3_8b/
    ├── config.pbtxt            # Triton model config
    └── 1/
        └── engine/             # .engine + tokenizer files
            ├── rank0.engine
            ├── tokenizer.json
            └── ...

# Launch Triton
tritonserver --model-repository=models --backend-config=tensorrtllm,...
```

Client (gRPC or HTTP):

```python
import tritonclient.grpc as grpcclient
client = grpcclient.InferenceServerClient(url="localhost:8001")
inputs = [grpcclient.InferInput("text_input", [1], "BYTES")]
inputs[0].set_data_from_numpy(np.array(["Hello"], dtype=object))
response = client.infer("llama_3_8b", inputs)
```

---

## Strengths

- Highest throughput on NVIDIA hardware (MLPerf grade)
- FP8/NVFP4 done right via Transformer Engine
- Tight Triton integration → production observability, metrics, model versioning

## Limits

- Engine-build operational overhead
- SKU-specific binaries (H100 engine ≠ B100 engine)
- Custom model architectures require conversion + plugin work
- Less Python-introspectable than vLLM/SGLang for debugging

---

## Connections

- [[excerpts/tensorrt-llm-paged-kv]] — paged-KV + IFB details.
- [[excerpts/cuda-graphs-inference]] (ch-12) — TRT-LLM uses CUDA-Graph capture inside the compiled engine.
- [[excerpts/pagedattention]] (ch-06) — the design family TRT-LLM's paged KV belongs to.
- [[ch-18]] — parent synthesis of production serving stacks.
