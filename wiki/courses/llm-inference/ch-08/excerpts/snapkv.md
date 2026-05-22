---
chapter: ch-08
course: llm-inference
phase: read
excerpt_of: "SnapKV: LLM Knows What You Are Looking for Before Generation (Li et al. 2024)"
source_url: https://arxiv.org/abs/2404.14469
created_at: "2026-05-21"
---

# Excerpt: SnapKV — observation-window prompt-side compression

**Authors:** Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, Deming Chen
**Year:** 2024
**URL:** https://arxiv.org/abs/2404.14469
**Raw-data source:** [[raw-data/snapkv]]

---

## The thesis

> "We notice that during generation, each attention head focuses on a stable subset of prompt tokens, and this subset can be reliably predicted from the attention pattern of the last few tokens of the prompt. We use this observation to compress the prompt KV cache before generation starts." (§1, paraphrased)

The mechanism is timing-shifted from H2O: H2O decides at every decode step which tokens to evict; SnapKV decides **once**, at the end of prefill, what to keep — and then runs decode normally against the compressed cache.

---

## The observation that justifies the method

A two-part empirical claim (§3):

1. **Per-head importance is stable across the generation.** The set of prompt positions a given attention head looks at during decode is largely the same set it was looking at during the last tokens of prefill.
2. **The observation window is a small slice.** Looking at attention from the last 32–64 prompt tokens is sufficient to identify which prompt tokens each head will need.

Both claims are supported by attention-stability measurements across LLaMA-2, Mistral, and LongChat (paper Figures 3–5).

---

## The algorithm

```python
def snapkv_compress(prompt_kv, attn_weights_last_W, keep_k, kernel_window=15):
    """
    prompt_kv:           [L, 2, H_kv, T_prompt, d_head]   full prefill cache
    attn_weights_last_W: [L, H_kv, W, T_prompt]           attention from the last W
                                                          observation tokens to all prompt
                                                          tokens, per layer per head.
    keep_k:              tokens to retain per head per layer
    kernel_window:       width of the smoothing kernel for selection
    """
    importance = attn_weights_last_W.sum(dim=2)           # [L, H_kv, T_prompt]
    importance = pool1d(importance, kernel_window)        # smooth → clusters

    keep_indices_per_head = importance.topk(keep_k, dim=-1).indices  # [L, H_kv, keep_k]
    compressed_kv = gather_kv(prompt_kv, keep_indices_per_head)
    return compressed_kv, keep_indices_per_head
```

Three pieces worth flagging:

- **Per-head, per-layer selection.** Different heads at the same layer keep different positions. This is essential — heads specialize, and a global keep-set would be either too small (some heads miss critical positions) or too large.
- **1D-pool smoothing.** Selecting raw top-k gives isolated positions; the model usually attends to small *contiguous neighborhoods* (a phrase, an entity span). Smoothing with a small window picks the centers of clusters, then the gather collects those neighborhoods.
- **Compression once, then forget.** After prefill, the engine throws away non-selected blocks. Decode runs against the compressed cache; per-step cost matches the compressed size.

---

## Memory savings (paper §5)

| Model | Context | Compressed | Ratio | LongBench Δ |
|---|---|---|---|---|
| Mistral-7B-32k | 32k | 3.2k | 10× | -0.6 |
| LWM-Text-Chat-128k | 64k | 4.0k | 16× | -1.4 |
| LLaMA-2-7B-80k | 80k | 4.0k | 20× | -1.1 |
| LongChat-7B-32k | 32k | 1.6k | 20× | -0.9 |

The compression ratio gain at 80k context is striking: a 20× reduction with ~1 point of LongBench score loss. Below 20× ratio the quality cost is essentially measurement noise.

---

## Decode-time speedup

SnapKV is the simplest of the KV compression methods to deploy because **no runtime logic changes**. After end-of-prefill compression, the KV cache is just *smaller*. Standard paged decode, FlashAttention, and continuous batching all work unchanged.

Decode latency scales sub-linearly with the compression ratio (FFN cost stays the same), so 10× KV compression yields ~6–8× decode speedup on H100 + Mistral-7B at 32k context.

---

## Where SnapKV fits and doesn't

**Where it wins:**

- Long-prompt, short-generation workloads (RAG, summarization, long Q&A).
- One-shot tasks where the prompt is the binding constraint.
- Workloads with simple deployment requirements (no kernel changes, no eviction logic).

**Where it doesn't:**

- Long *generation* workloads (chain-of-thought, reasoning chains). SnapKV compresses the prompt; the decode-side KV still accumulates linearly.
- Multi-turn dialogue where the "prompt" includes previous-turn outputs that are themselves long.
- Tasks requiring near-perfect recall of all prompt details (e.g., exact-text needle-in-haystack).

For multi-turn, SnapKV can be combined with **incremental re-compression** at each turn boundary, but this is an extension not in the original paper.

---

## Implementation footprint

SnapKV requires:

- One instrumentation hook at end of prefill to capture `attn_weights_last_W`. This is a normal attention output; in FlashAttention, exposing it requires a small kernel option.
- One CPU-side selection pass (top-k after smoothing).
- One gather op to produce the compressed cache (the rest of the cache blocks are freed back to the pool).

Total added complexity: ~200 LOC in vLLM or SGLang. Most other compression methods are 10× larger.

---

## Connections

- [[excerpts/h2o]] — runtime-eviction alternative; comparable savings, different timing.
- [[excerpts/quest-kv]] — keeps the full cache but selects pages per-query at decode; complementary to SnapKV's one-shot compression.
- [[excerpts/attention-sinks]] — discusses why sink tokens must be preserved; SnapKV's top-k usually picks them automatically, but a safety-explicit policy is sometimes added.
- [[ch-08]] — parent synthesis.
