---
chapter: ch-08
course: llm-inference
phase: read
excerpt_of: "InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory (Xiao et al. 2024)"
source_url: https://arxiv.org/abs/2402.04617
created_at: "2026-05-21"
---

# Excerpt: InfLLM — external memory for million-token inference

**Authors:** Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, Maosong Sun
**Year:** 2024
**URL:** https://arxiv.org/abs/2402.04617
**Raw-data source:** [[raw-data/infllm]]

---

## The thesis

> "We propose a training-free memory-based method, InfLLM, to unveil LLMs' intrinsic ability to process streaming long sequences. InfLLM stores distant contexts into additional memory units and employs an efficient mechanism to look up token-relevant units for attention." (§1)

InfLLM is the most ambitious method in this chapter: it extends a model trained on 4k–8k context to handle ~128k–1M tokens at inference, *without fine-tuning*. The trick is to make distant context a retrievable memory rather than something attended-to directly.

---

## The architecture

The full input stream is partitioned into three regions:

```
┌────────────────────┬────────────────────────────┬────────────┬────────────┐
│ initial sink (n_i) │     past memory units      │  recent    │  current   │
│                    │  (chunked, retrievable)    │  local (n_l│ generation │
│                    │                            │            │            │
└────────────────────┴────────────────────────────┴────────────┴────────────┘
```

- **Initial sink** (`n_i` tokens, typically 4–32) — kept always, per StreamingLLM logic.
- **Past memory units** — older context chunked into fixed-size units (typically 128 tokens each). Each unit gets a *representative key* (a pooled summary of its keys).
- **Recent local window** (`n_l` tokens, typically 1024–4096) — full attention here.
- **Current generation** — being decoded.

At every layer of every decode step, the query attends to:

- The initial sink (always).
- The recent local window (always).
- The **top-`k` memory units** (selected per-query by similarity between query and unit representative key).

So per-step attention compute is bounded: `O(n_i + n_l + k × unit_size)` — independent of total context length.

---

## The unit-selection rule

For each query `q` at layer `ℓ`:

```python
# rep_keys: [num_units, n_heads, d_head]   pooled representative keys per unit
relevance = q @ rep_keys.transpose(-2, -1)        # per-head dot products
top_k_units = relevance.topk(k, dim=-1).indices   # per-head top-k

# Load only those units' KV
selected_kv = gather_kv(memory_units, top_k_units)
attn_output = attention(q, sink_kv + selected_kv + local_kv)
```

The selection is per-head and per-layer. Different heads at the same layer retrieve different units. Different layers similarly. The effective attention pattern can span the entire input even though only a small fraction of memory is loaded per step.

---

## Why it doesn't need fine-tuning

The model was trained on contexts of length `≤ training_ctx`. Per step, InfLLM never asks it to attend over more positions than `training_ctx` — the active context (sink + recent + selected units' tokens) is sized to fit. The model's softmax operates over a familiar number of keys. The *content* of those keys spans the full long context, retrieved by InfLLM's selection layer.

Compare this with naive context extrapolation (RoPE-NTK, YaRN, etc.), which asks the model to attend over many more positions than training and degrades quality without long-context fine-tuning.

---

## Empirical effect (paper §4)

| Model | Training context | Inference context | InfiniteBench accuracy |
|---|---|---|---|
| LLaMA-3-8B (8k train) | 8k | 8k (no extension) | 22 % |
| LLaMA-3-8B + naive sliding window | 8k | 128k | 5 % |
| LLaMA-3-8B + NTK-aware scaling | 8k | 128k | 18 % |
| **LLaMA-3-8B + InfLLM** | 8k | 128k | **42 %** |
| LLaMA-3-70B (8k train) + InfLLM | 8k | 1024k | 35 % on ∞-Bench retrieval |

The fact that an 8k-train model can hit reasonable retrieval accuracy at 1M tokens of inference context — *without fine-tuning* — is the headline. Other long-context approaches either require expensive fine-tuning (LongLoRA, position-extrapolation pretraining) or degrade sharply.

---

## Compute and memory cost

Per step, InfLLM's cost is bounded:

```
attention_compute  = O(n_i + n_l + k * unit_size)            ≈ constant in total context
memory_resident    = O(initial_sink + recent_window + total_units)
                   = O(total_context / unit_size + window)
```

For 1M tokens of context with `unit_size = 128`: ~8000 unit summaries, ~1 MB of representative keys. Per step we load ~8 units' worth of KV (`k=8`, ~1k tokens). Compare with full attention's 1M-token load: ~100,000× bandwidth savings at the cost of one-time unit summarization during prefill.

The unit summaries can be kept in HBM (small), but the full unit KV can be demoted to CPU or distributed storage and loaded on demand — exactly the HiCache (ch-07) tiering pattern.

---

## Where InfLLM fits and doesn't

**Fit:**

- Long-document Q&A where the question is a small slice of the document at a time.
- Codebase navigation — only the relevant files / functions need loading per query.
- Multi-document RAG where each query touches a subset.
- Streaming workloads where past context is referenced occasionally.

**Doesn't fit:**

- Tasks requiring *dense* use of the full context. Summarize-every-section needs every section loaded, defeating the retrieval premise.
- Workloads where retrieval misses are unacceptable — InfLLM is approximate.
- Cases where unit-summary quality is poor for the model (some models' early-layer summaries are weak).

---

## How InfLLM relates to others in this chapter

| Method | What it does about distant tokens |
|---|---|
| H2O | Evict the unimportant ones (gone forever) |
| SnapKV | Compress to top-k per head (gone forever) |
| Attention Sinks | Drop the middle, keep edges (gone forever) |
| Quest | Keep all in HBM, load only relevant per query (still in HBM) |
| **InfLLM** | Demote to memory tier, retrieve on demand (still accessible) |

InfLLM is the only one in this list with no permanent loss of information. The cost is a more complex pipeline (chunking, summarization, retrieval kernel) and approximate quality on dense-use tasks.

---

## Connections

- [[excerpts/attention-sinks]] — InfLLM uses the sink+window baseline, then adds the memory tier.
- [[excerpts/quest-kv]] — both make per-query selections; Quest at page granularity in HBM, InfLLM at unit granularity across tiers.
- [[excerpts/snapkv]] — different timing model; SnapKV one-shot, InfLLM continuous retrieval.
- [[excerpts/sglang-hicache]] (ch-07) — the production deployment substrate for InfLLM-style external memory.
- [[ch-08]] — parent synthesis.
- Forward to long-context-inference material — MLA, QJL quantization, and learned compression continue the "fit million-token contexts in real GPUs" thread.
