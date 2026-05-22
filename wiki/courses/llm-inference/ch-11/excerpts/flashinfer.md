---
chapter: ch-11
course: llm-inference
phase: read
excerpt_of: "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving"
source_url: https://arxiv.org/abs/2501.01005
created_at: "2026-05-21"
---

# Excerpt: FlashInfer — serving-aware attention engine

**Authors:** Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, Luis Ceze
**Year:** 2025
**Venue:** MLSys 2025 (arXiv 2501.01005)
**URL:** https://arxiv.org/abs/2501.01005
**Raw-data source:** [[raw-data/flashinfer]]

---

## What FlashInfer is (and isn't)

FlashInfer is **not** a single attention kernel. It's a **kernel engine** that:

1. Provides operator APIs for prefill, decode, append/update over paged or ragged KV caches.
2. Dispatches at runtime to the right kernel variant (FA2 / FA3 / FlashDecoding / cascade / custom) based on the actual batch shape.
3. JIT-compiles specialized kernels per `(model architecture, attention config, KV layout)` at engine init.
4. Supports CUDA graph capture out of the box (static-shape design).

Built specifically for serving engines (vLLM, SGLang, MLC) rather than model code.

---

## The serving-attention variants FA2/FA3 don't cover

| Variant | What's special | Why FA2 alone is wrong |
|---|---|---|
| Paged KV | KV in 16-token blocks, addressed via block table | FA2 wants contiguous K, V |
| Ragged batches | Per-request kv_lengths differ | FA2 needs padding (wastes compute) |
| GQA / MQA | kv_heads < n_heads, one K serves many Q | FA2 has, but needs config wiring |
| Cascade attention | shared system prompt + per-user suffix | recompute waste without cascade |
| Score modifiers | ALiBi, attention sinks, query-aware pruning | not in FA2 |
| CUDA-graph-safe | static shape across iterations | dynamic batches break graph capture |

FlashInfer's contribution is exposing all six as composable operators.

---

## Core APIs

**Paged batch prefill:**

```python
import flashinfer

wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, kv_layout="NHD"
)
wrapper.plan(
    qo_indptr=...,         # which Q rows belong to which request
    paged_kv_indptr=...,   # per-request start offset in block table
    paged_kv_indices=...,  # block IDs for each request
    paged_kv_last_page_len=...,
    num_qo_heads=32, num_kv_heads=8, head_dim=128, page_size=16,
    causal=True, pos_encoding_mode="ROPE_LLAMA",
)
output = wrapper.run(q, paged_kv_data)
```

**Paged batch decode:**

```python
wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(...)
wrapper.plan(...)  # same shape inputs
output = wrapper.run(q, paged_kv_data)  # uses FlashDecoding under the hood
```

**Cascade attention** (shared prefix + per-request suffix):

```python
shared_state = flashinfer.prefill_with_kv_cache(q, k_shared, v_shared)
private_state = flashinfer.prefill_with_kv_cache(q, k_private, v_private)
out = flashinfer.cascade.merge_states([shared_state, private_state])
```

The shared-prefix attention is computed **once** per batch (not per request); merging is cheap.

---

## JIT specialization

FlashInfer doesn't ship a megakernel that handles every shape. At engine init, it generates the specific kernel variant your serving engine needs:

```text
flashinfer.jit.gen_batch_decode_module(
    dtype_q="float16", dtype_kv="float16", dtype_o="float16",
    head_dim_qk=128, head_dim_vo=128,
    pos_encoding_mode="ROPE_LLAMA", use_logits_soft_cap=False,
    use_fp16_qk_reduction=False
)
```

Each specialization compiles in ~5s and is cached. Result: a Llama-3-8B server only has the specific kernel for `(fp16, d=128, GQA-8, RoPE, no soft cap, ...)` — no dispatch overhead at runtime.

---

## CUDA-graph compatibility

The `plan()` call records all shape information; subsequent `run()` calls have stable shapes. A serving engine can wrap the entire decode loop in a CUDA graph (ch-12) because FlashInfer guarantees the kernel calls have identical shapes per iteration.

Without this, the engine would have to fall back to eager execution every time batch composition changes.

---

## Integration in production

| Framework | FlashInfer usage |
|---|---|
| vLLM | `VLLM_ATTENTION_BACKEND=FLASHINFER` |
| SGLang | default backend |
| MLC-Engine | default backend |
| TensorRT-LLM | competing internal kernels (FasterTransformer lineage) |

---

## Reported numbers

- 28–69% lower inter-token latency vs vLLM's FA2-only backend on long-context decode.
- Long-context prefill: similar speed to FA3 (FlashInfer dispatches to it on Hopper).
- Parallel generation (sampling N from same prompt): 2–3× faster via cascade attention.

---

## Connections

- [[ch-11]] — parent chapter; FlashInfer is the lineage's "serving entry point".
- [[excerpts/flashattention-2]] / [[excerpts/flashattention-3]] / [[excerpts/flashdecoding]] — dispatch targets.
- [[pagedattention]] (ch-06) — the KV layout FlashInfer was built around.
- [[ch-07]] / [[sglang-radixattention]] — cascade attention is the kernel that makes RadixAttention efficient.
- [[ch-12]] / [[cuda-graphs-inference]] — FlashInfer's static-shape design is a precondition for graph capture.
- [[ch-16]] / [[ch-17]] — vLLM and SGLang both use FlashInfer.
