# The KV Cache Memory Formula — Exact Bytes, Worked Numbers, and Where the Factor 2 Comes From
<!-- slug: kv-cache-memory-formula · type: doc · source: wiki:course/llm-inference:wiki/raw-data/llm-inference/classics/kv-cache-memory-formula.md -->

**Core Insight.** One equation sizes every autoregressive serving deployment:
`KV bytes = 2 · B · s · L · n_kv_heads · d_head · bytes_per_element`. The leading **2 is K and V, nothing else**
— not "two bytes", not a safety factor. Every later optimisation (GQA, MQA, MLA, KV quantisation, PagedAttention,
KV eviction) is a targeted attack on exactly one term of this product.

**Guideline.** Before deploying, compute per-token KV bytes from the model config, multiply by max context to get
per-request KV, subtract weights + activation overhead from HBM to get the KV budget, and divide. Always read
`num_key_value_heads` from the config — using `num_attention_heads` overstates nothing but *understates* your
capacity by 4×–64× on any modern GQA/MQA model.

## Technical Details

- **The formula, per sequence and per batch:**
  ```
  KV bytes (one sequence)  = 2 · L · n_kv_heads · d_head · s · bytes_per_element
  KV bytes (batch of B)    = 2 · B · s · L · n_kv_heads · d_head · bytes_per_element
  ```
  | Symbol | Meaning | Config field |
  |---|---|---|
  | `2` | **one K tensor + one V tensor** | — |
  | `L` | transformer layers | `num_hidden_layers` |
  | `n_kv_heads` | **KV** heads, not query heads | `num_key_value_heads` |
  | `d_head` | head dimension = `hidden_size / num_attention_heads` | — |
  | `s` | cached tokens = **prompt + generated so far** | — |
  | `B` | concurrent sequences | — |
  | `bytes_per_element` | 2 (bf16/fp16), 1 (fp8/int8), 0.5 (int4) | — |
  Note the formula is **independent of `n_heads` and of `d_model`** — a 405B model caches only 1.58× what an
  8B model caches, because only `L` grew.
- **Per-token KV cost (`2 · L · n_kv_heads · d_head · b`), bf16 — verified exactly:**
  | Model | `L` | `n_kv_heads` | `d_head` | bytes/token | KiB/token |
  |---|---:|---:|---:|---:|---:|
  | Llama-2-7B (MHA) | 32 | 32 | 128 | **524,288** | 512.0 |
  | Llama-3-8B (GQA-8) | 32 | 8 | 128 | **131,072** | 128.0 |
  | Llama-3-70B (GQA-8) | 80 | 8 | 128 | **327,680** | 320.0 |
  | Llama-3-70B *if MHA-64* | 80 | 64 | 128 | **2,621,440** | 2,560.0 (2.5 MiB) |
  | Llama-3-405B (GQA-8) | 126 | 8 | 128 | **516,096** | 504.0 |
  | PaLM-540B (MQA) | 118 | 1 | 256 | **120,832** | 118.0 |
- **Worked example — Llama-3-70B, the canonical reference:** `2 · 80 · 8 · 128 · 2 = 327,680 B/token = 320 KiB`.
  | Context `s` | per request |
  |---:|---:|
  | 8,192 | 2,684,354,560 B = **2.50 GiB** |
  | 32,768 | 10,737,418,240 B = **10.00 GiB** |
  | 131,072 (128k) | 42,949,672,960 B = **40.00 GiB** |
  On 8×H100 (640 GB HBM), weights 140 GB bf16 + ~80 GB overhead ⇒ KV budget ≈ 420 GB ⇒ **168 concurrent 8k
  requests**, but only **10.5 concurrent 128k requests**. 16× context → ~16× fewer concurrent requests, linearly.
- **Worked example — batched, from the classic card:** `L=32, B=16, s=4096, n_kv_heads=8, d_head=128, bf16`
  → `2·32·16·4096·8·128·2 = 8,589,934,592 B = 8.00 GiB` (8.59 GB decimal).
- **What the formula omits.** It is a lower bound. Add: block/page tables ~1–2% under PagedAttention;
  fragmentation 5–15% in naive contiguous allocators, <5% paged; pre-allocated swap headroom; CUDA-graph
  batch-bucket reserves. **Rule of thumb: multiply by 1.15 for real-world consumption.**
- **Common pitfalls (verbatim from the source card).** Using `n_heads` instead of `n_kv_heads` (off by 4×–32×);
  forgetting `s` includes the prompt (a 4k-prompt + 4k-output request has `s = 8192`, not 4096); ignoring dtype;
  treating the KV budget as fixed across a batch (it *grows* one token per sequence per decode step, so schedulers
  must reserve for max length); forgetting that under tensor parallelism KV shards along `n_kv_heads`, so each
  GPU holds `n_kv_heads / TP` heads.
- **Training-memory angle:** The identical algebraic expression appears in a training budget — but it names a
  *different object*. `2 · B · s · L · n_kv_heads · d_head · b` is also the exact size of the **K and V activations
  saved for backward** across all layers of one micro-batch. Example: Llama-3-8B geometry at `B=1, s=8192, bf16`
  gives `1,073,741,824 B = 1.00 GiB` — numerically the same as the inference KV cache for one 8k request. The
  difference is lifetime and removability: the training tensors live for one forward+backward and are then freed,
  and gradient checkpointing can eliminate them entirely by recomputing ([[gradient-checkpointing-chen]],
  [[selective-recompute-korthikanti]]); the inference cache persists for the whole request and can only be
  *compressed*, never recomputed away. Budget them in different buckets — see [[train-vs-infer-kv-boundary]].

## Citation
Derived from Vaswani et al. 2017 (arXiv:1706.03762), Shazeer 2019 MQA (arXiv:1911.02150), Ainslie et al. 2023 GQA
(arXiv:2305.13245), and Kwon et al. 2023 PagedAttention/vLLM, SOSP (arXiv:2309.06180). Numbers re-derived and
verified from `course/llm-inference:wiki/raw-data/llm-inference/classics/kv-cache-memory-formula.md`,
`.../ch-03/excerpts/kv-cache-formula.md`, and `.../ch-20/excerpts/llama-3-gqa-and-kv.md`.
