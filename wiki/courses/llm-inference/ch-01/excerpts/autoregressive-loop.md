---
chapter: ch-01
course: llm-inference
phase: read
excerpt_of: "Attention Is All You Need (Vaswani et al. 2017) + Language Models are Unsupervised Multitask Learners (Radford et al. 2019)"
source_url: https://arxiv.org/abs/1706.03762
created_at: "2026-05-21"
---

# Excerpt: The autoregressive decoder loop

**Authors:** Ashish Vaswani et al. (Transformer, 2017); Alec Radford et al. (GPT-2, 2019)
**Year:** 2017 / 2019
**Venue:** NeurIPS 2017 (Transformer); OpenAI technical report (GPT-2)
**URLs:** https://arxiv.org/abs/1706.03762 ; https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
**Raw-data sources:** [[raw-data/attention-is-all-you-need]], [[raw-data/language-models-are-unsupervised-multitask-learners]]

---

## The scaled dot-product attention block

The single computation underneath the autoregressive loop:

```math
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
```

For a causal decoder generating token `t`, the query is the projection of the current token's hidden state, and `K, V` are the projections of *all* prior tokens. Without caching, every decode step would recompute `K, V` for the entire history — `O(t)` work per step that throws away the previous step's result.

The mask `M` (added before softmax) sets `M[i,j] = -∞` for `j > i`, enforcing the autoregressive constraint that token `i` cannot see future tokens. During incremental decoding the mask is implicit: you simply compute attention against only the cached `t` keys.

---

## The two phases formalized

GPT-2 ([[raw-data/language-models-are-unsupervised-multitask-learners]]) made decoder-only generation the universal interface. The runtime decomposition that every modern engine implements:

```
PREFILL  (one call)
    input:  prompt tokens [x_1, ..., x_S]
    work:   compute Q, K, V for all S positions in parallel
            compute full S×S attention matrix per layer
            FFN + residuals + layernorms for all S positions
    output: logits[S-1, :] over vocabulary
    side effect: K, V tensors for all S positions cached per layer

DECODE   (one call per output token, repeated until stop)
    input:  one new token x_t + cached K, V for positions 1..t-1
    work:   compute Q, K_t, V_t for position t
            attention is now [1 × t] per layer (not t×t)
            FFN + residuals + layernorms for one position
    output: logits[0, :] over vocabulary
    side effect: K_t, V_t appended to cache
```

This is the bedrock asymmetry that ch-03's prefill-vs-decode analysis builds on. Prefill is **compute-bound** (lots of matmul on the prompt); decode is **memory-bandwidth-bound** (one query, but reads all prior KV).

---

## Why caching is mandatory

Without the KV cache, decode for output token `t` would cost roughly `O(t·d²)` per layer to recompute the prompt history, summed over all output tokens giving `O(T²·d²·L)` total work. With the cache it's `O(T·d²·L)` plus `O(T·t·d·L)` attention reads. For a 4k-token output the difference is roughly 4096× speedup in the recompute term.

The Transformer paper does not discuss caching — it only trains. Caching is an inference-time engineering decision that fell out of standard practice in 2018–2019 with GPT-1/2 deployment. By GPT-3 ([[raw-data/gpt-3-language-models-are-few-shot-learners]]) it was so universal that no inference codebase ships without it.

---

## What GPT-2 cemented as the universal interface

Before GPT-2, NLP systems used task-specific heads — classification heads for sentiment, span heads for QA, sequence-to-sequence heads for translation. GPT-2 demonstrated that all of these can be elicited from one causal LM by varying the *prompt*. The implication for inference is profound:

- One model architecture serves every task.
- The "task" is encoded as the prompt prefix.
- The same decode loop generates classification labels, translations, JSON, and chat — only the stop conditions and sampler differ.

This is why modern serving stacks (vLLM, SGLang, TGI) implement exactly one inference path. The OpenAI Chat Completions API is a thin wrapper around this loop; the Anthropic Messages API is another; both produce SSE streams of one-token deltas with one final `usage` object.

---

## Common pitfalls

- **Forgetting positions when extending context.** The cached KV at position `t` was computed with the position embedding for `t` (or, for [[rope]], rotated by the angle for `t`). Reusing the cache for a different position will silently produce nonsense. Always track absolute positions explicitly.
- **Recomputing instead of caching.** A "stateless" inference function that re-tokenizes and re-prefills the entire conversation on every turn is functionally correct but ~50× slower than a cached implementation. Production code never does this.
- **One forward pass per token at scale.** Decoding 1024 tokens means 1024 launches of the same decoder stack. The per-step latency floor is roughly `(num_layers · num_kernels_per_layer · launch_overhead)`, ~5–10 ms on a 70B model — which is why CUDA Graphs (ch-12) exist.

---

## Connections

- [[excerpts/sampling-strategies]] — once you have logits, the sampler converts them to one token id; the second half of the loop.
- [[raw-data/prefill-vs-decode]] — the SARATHI / DistServe formalization of the two-phase split.
- [[raw-data/kv-cache-memory-formula]] — what the cache costs in GB.
- [[ch-02]] — the attention complexity that makes prefill costly and decode cheap-per-step.
- [[ch-04]] — when many requests share the loop, batches are formed at the decode-step granularity.
