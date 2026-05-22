<!-- chapter: ch-08
     track: kv-cache
     title: KV Cache Compression + Eviction (H2O / SnapKV / Attention Sinks / Quest / InfLLM)
     sources: [[h2o]], [[snapkv]], [[attention-sinks]], [[quest-kv]], [[cachegen]], [[infllm]]
     back: [[kv-cache-memory-formula]] (ch-03), [[pagedattention]] (ch-06)
     forward: long-context-inference (analogous to [[qjl]] in adjacent course)
     figures: figures/h2o-eviction-policy.html
-->

# Chapter 8 — KV Cache Compression + Eviction

> **Core insight.** Paged allocation (ch-06) and prefix caching (ch-07) reduce *waste* in the KV cache; they do not reduce the per-token bytes. For long-context generation, the per-token bytes themselves become the bottleneck — a 128k-context Llama-3-70B request costs ~160 GB of KV, far more than any single GPU has. The 2023–2024 literature attacks this with a single empirical claim: **attention is sparse**. Only a small fraction of past tokens carry most of the attention mass for any given decode step. If you can identify which ones cheaply, you can evict or skip the rest. **H2O** uses accumulated attention scores to find heavy-hitters. **SnapKV** observes the last few tokens of prefill to predict per-head important positions for the whole generation. **StreamingLLM / Attention Sinks** keeps the first N tokens always, exploiting the model's strong positional bias toward them. **Quest** loads only query-relevant pages at decode time. **InfLLM** layers external memory retrieval on top of bounded local attention. Each makes a different trade between memory savings, quality, and implementation complexity.
>
> **Guideline.** For short-to-medium contexts (≤ 32k), don't enable any of these — paged + prefix is enough and these methods cost quality. For 32k–128k contexts, **SnapKV** (prompt-side compression, runs once at prefill end) or **StreamingLLM** sink+window (for genuine streaming) are safe wins. For >128k contexts where decode is bandwidth-bound, **Quest** (query-aware sparsity) is the strongest *runtime* method — it changes which blocks load, not which blocks exist. **InfLLM** is the production answer when the workload is "million-token document Q&A" rather than "extend a single conversation."

---

## Why this chapter exists

The ch-03 KV-cache formula `bytes = 2 · L · H_kv · d_head · T · b` has one term — `T`, the context length — that grows monotonically as the model generates. For the deployments dominating the 2024–2026 production landscape, that growth is the binding constraint:

- **Claude / Gemini 1M-token contexts** — pure compute scales acceptably (FlashAttention etc.), but KV bandwidth at decode does not. Each decode step must stream all of the cache from HBM.
- **DeepSeek R1 long-CoT generation** — generates 8k–32k tokens per response; KV cache after generation is comparable to a long prefill.
- **Agent loops** — accumulate KV across many turns; without compression, a long-running agent crashes the server on KV exhaustion.

Two distinct sub-problems:

1. **Decode bandwidth bottleneck.** Even if the KV fits in HBM, every decode step reads all of it. For Llama-3-70B at 100k context: ~160 GB of KV cache × ~3 TB/s HBM = ~50 ms of pure KV reads per decode step. Adding more decoding requests doesn't help — they're all reading their own KV.
2. **HBM capacity ceiling.** A 70B-class model at 1M tokens is ~1.5 TB of KV. No GPU exists with that HBM. Either compress, offload, or admit far fewer requests.

The five methods covered here address one or both:

- **H2O** — runtime eviction during decode. Reduces HBM use and bandwidth.
- **SnapKV** — one-shot compression after prefill. Reduces both, mostly the bandwidth.
- **StreamingLLM (Attention Sinks)** — sliding window + permanent sink. Bounded HBM for unbounded streams.
- **Quest** — query-aware sparse loading at decode. Reduces bandwidth only; HBM unchanged.
- **InfLLM** — external memory + retrieval. Bounds HBM with a memory tier.
- **CacheGen** (briefly) — compressed transport for cross-machine KV reuse.

Together they form the toolbox 2026 serving systems reach for when paging and prefix caching are not enough.

---

## 1. H2O — the heavy-hitter oracle

[[h2o]] (Zhang et al. 2023) starts from a clean empirical observation: during generation, attention mass is *concentrated*. Most past tokens contribute negligibly to the softmax; a small "heavy-hitter" set carries the rest.

**The observation, quantified.** Across many layers and heads of OPT-6.7B, ~5 % of tokens carry ~95 % of cumulative attention mass over a typical generation. The distribution is heavy-tailed: a tiny fraction of past tokens are looked at again and again across decode steps.

**The eviction policy.** Maintain a cache of bounded size `B`. At each decode step:

1. Compute attention scores for the current query.
2. Update each cached token's **accumulated attention score**: `score[i] += attn_weight[i]`.
3. Always keep the most recent `r` tokens (the "recent set").
4. Among older cached tokens, keep the top-`(B - r)` by accumulated score; evict the rest.

This is the "heavy-hitter + recent-window" formula. The paper's theoretical framing is *submodular*: heavy hitters maximize a coverage objective that bounds the deviation from the full-attention output.

**Pseudocode** (simplified):

```python
def h2o_step(query, cache, B, r):
    # cache: list of (key, value, score)
    attn = softmax(query @ stack(k for k,_,_ in cache).T)   # per-token weight
    for i, (k, v, s) in enumerate(cache):
        cache[i] = (k, v, s + attn[i])                       # accumulate
    if len(cache) > B:
        recent = cache[-r:]
        old    = cache[:-r]
        old    = sorted(old, key=lambda t: -t[2])[:B - r]    # top by score
        cache  = old + recent
    return attention_output(query, cache), cache
```

**Empirical effect.** On OPT-6.7B with 20 % of full cache: H2O matches full-cache perplexity within 0.1 on WikiText; throughput on a single A100 rises 3× because batch size grows (less KV per request → more requests fit).

**The honest caveats:**

- **It's lossy.** Evicted tokens are gone; if a later query needed them, the model effectively didn't see them. Quality regression is small for many tasks but real for ones requiring distant recall.
- **Score accumulation is per-decode-step bookkeeping.** Modest CPU/GPU overhead.
- **Sensitive to the `r/B` ratio.** Too-small recent window breaks local coherence; too-large kills the heavy-hitter benefit.

H2O is the canonical reference for *attention-derived eviction* — and the baseline every subsequent KV-compression paper compares against.

---

## 2. SnapKV — the observation-window mechanism

[[snapkv]] (Li et al. 2024) attacks the same problem with a different timing: instead of evicting during decode, compress **once, at the end of prefill**, and use the compressed cache for the entire generation.

**The mechanism.**

1. **Observation window.** Run the full prefill normally, but instrument the *last* `W` tokens (typically `W = 32–64`) of the prompt to record their attention patterns over all earlier prompt tokens.
2. **Per-head importance vote.** For each attention head, sum the attention weights it gave to each earlier position across the observation window. The result: a per-head importance vector over the prompt's positions.
3. **Cluster + select.** For each head, keep the top-k positions (typically k = 256–1024) plus a small contiguous "neighborhood" around each, to preserve local coherence. Pool selections across the head to get a per-layer keep-set.
4. **Drop the rest.** Decode runs against the compressed cache.

The justification rests on an attention-stability observation: the positions a head attends to during *generation* are very close to what it attended to during the *end of prefill*. The observation window predicts the future.

**Empirical effect (paper §5).**

| Model | Full cache | SnapKV (compressed) | Memory ratio | LongBench score Δ |
|---|---|---|---|---|
| Mistral-7B-32k | 32k tokens | ~3k tokens (10×) | 0.094 | -0.5 |
| LLaMA-2-7B-80k | 80k tokens | ~4k tokens (20×) | 0.05 | -1.1 |

Decode speed roughly tracks the memory ratio — at 10× compression, decode is ~7–8× faster (the missing factor is FFN cost, which doesn't shrink).

**Where SnapKV is the right tool:**

- Long *prompts* with short generations (long-context Q&A, RAG, summarization). The compression happens once and you reap the savings the whole decode.
- Workloads where prompt content is the binding constraint, not generation length.

**Where it isn't:**

- Long generations where decode itself accumulates significant new KV (SnapKV compresses only the prefill cache; decode KV grows normally).
- Tasks requiring full prompt recall (needle-in-a-haystack with retrieval far from the observation window).

SnapKV is the simplest of these methods to deploy: one extra pass at end-of-prefill, then standard paged decode. No runtime eviction logic.

---

## 3. Attention Sinks — keep the first N tokens

[[attention-sinks]] (Xiao et al. 2023; published as StreamingLLM) is the smallest result on this list, and arguably the most surprising.

**The phenomenon.** Pretrained decoder-only LLMs assign disproportionately large attention weight to the *first few tokens of the context*, regardless of content. Even when those tokens are semantically meaningless (e.g., a BOS marker), they receive ~20–80 % of attention mass on many layers.

**Why.** The softmax in attention requires its weights to sum to 1. If no key matches the query well, the model still must put weight *somewhere*. Pretrained models learn to dump that residual mass on early tokens — the only positions guaranteed to exist in every input. These positions become **attention sinks**.

**The deployment consequence.** A naive sliding-window cache that evicts the oldest tokens (keeping only the most recent `W`) destroys the sinks → softmax distributions shift dramatically → perplexity explodes.

The fix: a **sink + window** cache. Always keep the first `N` tokens (typically `N = 4`), plus a sliding recent window of size `W` (typically `W = 1024` or `W = 4096`). Throw away the middle.

**Empirical effect (paper §5).**

- LLaMA-2-7B with naive sliding-window cache: perplexity explodes (>1000) past the training context length.
- LLaMA-2-7B with sink (N=4) + window (W=4096) cache: stable perplexity for >4M tokens of streaming generation. The model never sees the discontinuity.

**When to use it.**

- True streaming workloads — chatbots that never terminate, infinite scratchpads, log analysis.
- Inference past the training context length without fine-tuning.

The sink-token trick is so cheap (it's literally just "don't evict positions 0..3") that some pretraining recipes now reserve a special learnable sink token explicitly, exactly to make this deployment pattern robust.

---

## 4. Quest — query-aware sparsity at decode time

[[quest-kv]] (Tang et al. 2024) makes a different bet from H2O / SnapKV: don't *evict* anything; instead, at decode time, **load only the blocks the current query needs**.

The observation: token criticality is **query-dependent**, not query-agnostic. The tokens that mattered for question A are not the tokens that matter for question B. Static eviction (H2O, SnapKV) bakes in one importance ordering; Quest re-decides per decode step.

**The mechanism.** Quest builds on paged KV (ch-06). For each KV page (16-token block), it maintains compact summary statistics — the element-wise min and max of the keys in the block:

```python
class PageMeta:
    k_min: Tensor[d_head]   # min over the page's keys, per dimension
    k_max: Tensor[d_head]   # max over the page's keys, per dimension
```

At decode time for a query `q`:

1. **Estimate** an upper bound on the maximum attention score this page could produce: `score_upper(q, page) = max(q · k_min, q · k_max)` (element-wise, sum over `d_head`).
2. **Rank** all pages by their upper-bound score.
3. **Load** only the top-`K` pages and run attention against just those.

The upper-bound estimation is `O(d_head)` per page (one cheap inner product). The actual attention runs only over the selected pages — bandwidth savings proportional to `1 − K / total_pages`.

**The kernel.** Quest plugs into PagedAttention's block-table indirection (ch-06). The change to the attention kernel is one line — skip blocks not in the selected set. Implementation lives in custom paged kernels (FlashInfer's `BatchDecodeWithPagedKVCache`, vLLM's experimental `quest_attention`).

**Empirical effect.** On LLaMA-3-8B at 128k context with `K = top-1024` blocks (out of ~8000):

- Decode latency drops 3–5× (bandwidth dominant).
- Accuracy: within 1 % of full attention on LongBench.
- Memory: **unchanged** — Quest does not evict; it just skips at runtime.

**Why this matters in serving.** Eviction methods (H2O, SnapKV) make different importance bets per request. Quest makes a runtime bet per *decode step*. For mixed-question workloads against the same long context, Quest is the only one of these methods that doesn't fix-and-forget importance.

The trade is implementation complexity: Quest requires kernel-level integration with the paged cache, while SnapKV is a pre-decode pass that any engine can run.

---

## 5. InfLLM — external memory for million-token streams

[[infllm]] (Xiao et al. 2024) is the most ambitious of the five: extend a pretrained LLM to handle context far beyond its training length, *without fine-tuning*, by treating distant context as a retrievable external memory.

**The architecture.**

- **Local attention window** — the most recent `W` tokens are full-attention context, like a normal LLM.
- **Memory units** — older context is chunked into fixed-size units (typically 128 or 256 tokens). Each unit is summarized by a representative key.
- **Retrieval at decode** — for each layer, the query selects the top-`k` most relevant memory units; their KV is loaded and attended to alongside the local window.

The model's actual attention compute per step is bounded: `O(W + k × unit_size)` instead of `O(total context)`. Memory units not selected this step are not loaded.

**Why this works without fine-tuning.** The model's attention head architecture already does softmax-weighted retrieval; InfLLM externalizes the same mechanism. The model is never asked to attend to more positions than it was trained on; older positions are pre-filtered to the most relevant `k`.

**Empirical effect.** On a 1024-token training context model extended to 128k inference context: InfLLM passes ∞-Bench retrieval benchmarks at >85 % accuracy, where naive context extension scores <20 %. Memory cost per request is roughly `O(W + total_units × small_summary)` — bounded by the number of units, not the context length.

**When to use it.** Workloads where the input is so long that even compression methods (SnapKV, Quest) don't bring it into HBM range — long-form document Q&A, codebase navigation, multi-document RAG with 100k+ token context. InfLLM is the "retrieve, don't carry" answer.

**The trade.** InfLLM changes the model's effective attention behavior. Quality on tasks requiring dense use of the full context (e.g., summarizing every section of a 1M-token document) is worse than full attention. The right framing is "selective recall," not "full long-context."

---

## 6. CacheGen — compressed KV transport (cross-cutting)

[[cachegen]] is adjacent to this chapter: it doesn't reduce the active KV cache, but it makes *transferring* cached KV across machines feasible. For a deployment where one node prefills a long shared context and many other nodes serve queries against it, naively shipping the full KV cache over the network costs more than re-prefilling. CacheGen compresses the KV tensors using their distributional properties (heavy-tailed, group-structured) and streams the compressed bitstream, adapting compression level to bandwidth.

It's the natural complement to HiCache's L3 tier (ch-07) — HiCache decides *what* to store distributed; CacheGen decides *how* to encode the bytes on the wire.

For this chapter the key point is: KV compression is not only about saving GPU memory. Once distributed serving (ch-09) becomes the dominant pattern, compressing KV for *transport* matters as much as compressing it for *storage*.

---

## 7. Comparison table

The five methods, side by side:

| Method | What it reduces | When applied | Quality cost | Memory savings | Bandwidth savings | Kernel changes |
|---|---|---|---|---|---|---|
| **H2O** | HBM use + bandwidth | Every decode step | Small-moderate (lossy eviction) | 3–5× | 3–5× | Eviction logic |
| **SnapKV** | HBM use + bandwidth | Once at end of prefill | Small (-0.5 to -1.5 LongBench) | 10–20× | 10–20× | None (just allocate less) |
| **Attention Sinks (StreamingLLM)** | HBM use (bounded) | Cache eviction during stream | Negligible if N+W tuned | Unbounded streams in bounded HBM | proportional | None (cache mgr change only) |
| **Quest** | Bandwidth only | Each decode step | Small (-0.5 to -1 LongBench) | None | 3–8× | Paged attention + page selection |
| **InfLLM** | HBM use + compute | Decode step retrieval | Moderate (changes attention semantics) | Bounded for any context | Significant | Memory units + retrieval kernel |
| **CacheGen** | Network bytes (off-GPU) | KV transport | Small if quality target met | N/A (transport only) | Network-bound | None on GPU |

**Decision flow** for picking a default:

1. Context ≤ 32k → none; paged + prefix caching is enough.
2. Long prompts, short generations, simple deployment → **SnapKV**.
3. True streaming workloads → **Attention Sinks (StreamingLLM)**.
4. Long context, decode latency dominates, you control the kernel → **Quest**.
5. Context > 256k, retrieval-style usage → **InfLLM**.
6. Multi-node KV reuse over the network → **CacheGen** (on top of any of the above).

Most production deployments in 2026 combine: paged KV + prefix cache + SnapKV-for-long-prompts + StreamingLLM-for-chat. Quest and InfLLM are more recent and require kernel-level integration; they appear in research-grade serving (SGLang experimental builds) and in the long-context inference work covered in adjacent material on QJL-style compression.

---

## 8. Why this chapter is the bridge to long-context inference

Ch-06 (paging) and ch-07 (prefix caching) made the KV cache more *efficient*. Ch-08 (compression and eviction) makes it *smaller* — at a quality cost. The next conceptual step (covered in long-context-inference material in adjacent courses) is making the cache **structurally smaller** — fewer bytes per token via quantization (QJL), low-rank decomposition (MLA in DeepSeek V3), or learned compression. Those methods change the per-token KV footprint itself; this chapter's methods keep the per-token footprint but reduce the number of tokens you carry.

For LLM inference research the two threads are complementary. A frontier serving deployment for 1M+ token Gemini-style contexts in 2026 is some combination of:

- MLA-style attention (cuts per-token KV ~10×)
- Paged + radix prefix caching (eliminates waste, shares prefixes)
- SnapKV at end of prefill (cuts 5–10× more)
- Quest at decode (cuts bandwidth another 3–5×)

The aggregate is the only way to make a 1M-token cache fit in the bandwidth of a single H100, let alone in its HBM.

---

## Connections and what's next

- **Back to [[kv-cache-memory-formula]] (ch-03)** — the formula this chapter attacks; all five methods reduce its `T` or `T × bandwidth` product.
- **Back to [[pagedattention]] (ch-06)** — H2O, SnapKV, Quest all assume paged allocation; eviction = freeing blocks, sparsity = skipping blocks.
- **Forward to [[distserve]] (ch-09)** — disaggregated prefill/decode often *needs* compressed KV transport (CacheGen) because the cross-pool KV transfer is the bottleneck.
- **Forward to long-context-inference material** — KV quantization (QJL), MLA (DeepSeek V3), and learned compression continue this thread but operate at the *bytes-per-token* level, not the *number-of-tokens* level.
- **Forward to [[sglang-hicache]] (ch-07 / ch-17)** — HiCache's tiering composes with these methods: compressed L1, compressed L2, with different compression budgets per tier.

## Further reading

- [[h2o]] — Zhang et al. 2023, heavy-hitter eviction.
- [[snapkv]] — Li et al. 2024, observation-window compression.
- [[attention-sinks]] — Xiao et al. 2023 (StreamingLLM), sink + window cache.
- [[quest-kv]] — Tang et al. 2024, query-aware page sparsity.
- [[infllm]] — Xiao et al. 2024, external memory for long context.
- [[cachegen]] — Liu et al. 2023, compressed KV transport.

## Companion visualization

**[figures/h2o-eviction-policy.html](figures/h2o-eviction-policy.html)** — interactive eviction simulator: watch the cache evolve over a streaming generation under (a) naive sliding window, (b) H2O heavy-hitter + recent, (c) StreamingLLM sink + window. Heatmap shows per-position attention; eviction events flash; perplexity tracks live.
