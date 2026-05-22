---
chapter: ch-08
course: llm-inference
phase: read
excerpt_of: "Efficient Streaming Language Models with Attention Sinks (Xiao et al. 2023, StreamingLLM)"
source_url: https://arxiv.org/abs/2309.17453
created_at: "2026-05-21"
---

# Excerpt: Attention Sinks (StreamingLLM) — the sink+window cache

**Authors:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
**Year:** 2023 (ICLR 2024)
**URL:** https://arxiv.org/abs/2309.17453
**Raw-data source:** [[raw-data/attention-sinks]]

---

## The phenomenon

> "When applied to multi-round dialogue, the model degenerates significantly when the conversation length exceeds the cache size. We discovered that this collapse happens because the initial tokens contain large amounts of attention. We coined these tokens 'attention sinks'." (§1)

The empirical observation, which the paper validates across LLaMA-2, MPT, Falcon, and Pythia:

- The *first few tokens* of any input receive a disproportionate share of attention mass — often 40–80 % at deep layers — regardless of their semantic content.
- This is true even for the BOS token, or for a token that happens to be a comma.
- Evicting these tokens (as a naive sliding-window cache would) causes catastrophic perplexity collapse.

---

## Why sinks emerge

The mechanistic explanation (§3.2):

The attention softmax forces weights to sum to 1. If no key matches the query strongly, the model still must put weight *somewhere*. Pretrained decoder-only models learn to deposit this residual mass on the **earliest positions** — they are the only positions guaranteed to be present in every input.

> "We attribute the emergence of attention sinks to the SoftMax operation that requires attention scores to sum up to one for all contextual tokens. Even when the current query does not have a strong match in earlier tokens, the model still needs to allocate these unneeded attention values somewhere so it sums up to one." (§3.2)

This is a **structural** consequence of softmax + causal masking + autoregressive training, not a learned semantic behavior. It's why sinks exist in essentially every pretrained decoder LLM.

---

## The fix — sink + window cache

The deployment recipe is one paragraph:

```python
def streamingllm_cache(seq_kv, sink_N=4, window_W=4096):
    """Keep first sink_N positions and most recent window_W positions; drop the rest."""
    sinks  = seq_kv[:sink_N]
    window = seq_kv[-window_W:] if len(seq_kv) > sink_N + window_W else seq_kv[sink_N:]
    return sinks + window
```

Two parameters:

- **`sink_N` (typically 4).** The number of initial tokens always retained. The paper shows quality is mostly insensitive to `sink_N ≥ 4`; even `sink_N = 1` (just keep BOS) recovers most of the quality, but 4 is safer.
- **`window_W` (typically 1024–4096).** The size of the recent-token sliding window. Larger means more local context but more memory; this is the standard speed/quality knob for streaming.

The cache uses bounded memory `sink_N + window_W` regardless of stream length. The middle of the conversation is discarded.

---

## Empirical effect (paper §4)

For LLaMA-2-7B (4k training context) on a streaming PG19 evaluation:

| Cache strategy | Cache size | Perplexity at 4M tokens streamed |
|---|---|---|
| Full attention (impossible past 4k) | unbounded | NaN |
| Dense up to 4k, then sliding window (W=2048) | 2048 | >10⁶ (collapse) |
| Sliding window only (W=2048) | 2048 | >10⁶ (collapse) |
| Sliding window + position re-indexing | 2048 | ~6.4 |
| **StreamingLLM (sink=4, W=2048)** | 2052 | **5.71** |

The headline: a tiny constant-cost change (always keep 4 tokens) turns a stream that was collapsing past the training context into one that's *stable for millions of tokens*.

---

## Pretraining sink tokens

A second contribution of the paper (§4.4): adding a single dedicated **learnable sink token** during pretraining makes the deployment-time sink+window pattern even more robust. The model learns to dump residual attention mass on the explicit sink instead of on arbitrary early tokens, freeing the early tokens to carry real content.

This recipe has been adopted by some recent pretraining stacks (Qwen 3 variants, OpenAI-style "system" position markers). For models that did not pretrain a sink, deployment-time `sink_N = 4` is still sufficient.

---

## When to use Attention Sinks

**The fit:**

- True streaming workloads — chatbots, log analysis, voice transcription.
- Inference at context lengths *beyond* the training context.
- Bounded-HBM deployments where unbounded conversations must be supported.

**The limit:**

- Tasks that need recall of the middle of the discarded portion. The middle is gone; no retrieval mechanism.
- Tasks where the prompt has critical content past position `sink_N` but before `len − window_W`. These tokens are silently dropped.

For the recall failure mode, layer InfLLM or RAG-style retrieval on top: keep the sink+window cache for local coherence, retrieve memory units for distant recall.

---

## Implementation footprint

Smallest of any method in this chapter:

- One line change to the cache eviction policy in the KV cache manager.
- Sink positions are tagged with `ref_cnt = infinity` (never evictable).
- Window is a normal LRU on the rest of the cache.

vLLM has a `sliding_window` parameter that, when paired with `attention_sinks` flag, gives this behavior natively. SGLang likewise.

---

## Connections

- [[excerpts/h2o]] — the heavy-hitter view. Attention Sinks is the special case where positions 0..N are always heavy.
- [[excerpts/snapkv]] — one-shot compression including sink protection.
- [[excerpts/infllm]] — adds retrieval-style memory on top of bounded local attention; the natural complement when recall matters.
- [[ch-08]] — parent synthesis.
