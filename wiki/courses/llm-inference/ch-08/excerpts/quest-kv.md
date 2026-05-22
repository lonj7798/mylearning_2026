---
chapter: ch-08
course: llm-inference
phase: read
excerpt_of: "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference (Tang et al. 2024)"
source_url: https://arxiv.org/abs/2406.10774
created_at: "2026-05-21"
---

# Excerpt: Quest — query-aware page sparsity at decode time

**Authors:** Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han
**Year:** 2024 (ICML)
**URL:** https://arxiv.org/abs/2406.10774
**Raw-data source:** [[raw-data/quest-kv]]

---

## The thesis

> "Although prior work studies KV cache eviction, we observe that token criticality is highly dependent on the current query. In long-context inference, what tokens matter depends on what is being asked — and the answer changes every decode step. Quest exploits this query-dependence by dynamically selecting which KV pages to load, page by page, query by query." (§1, paraphrased)

H2O and SnapKV bake in one importance ordering — either accumulated during decode (H2O) or estimated at end-of-prefill (SnapKV) — and then live with it. Quest re-decides per decode step.

---

## The mechanism

Quest builds on PagedAttention (ch-06): the KV cache is already in blocks of 16 tokens. Quest adds per-block metadata:

```python
class PageMetadata:
    k_min: Tensor[d_head]   # element-wise min over the page's keys
    k_max: Tensor[d_head]   # element-wise max over the page's keys
```

Both vectors are `d_head`-dimensional; total metadata per 16-token page is `~512 bytes` for Llama-3 (head_dim=128). For 100k tokens of KV (6250 pages), metadata totals ~3 MB — negligible.

---

## The selection rule

At decode time, the query vector `q ∈ ℝ^d_head` is known. For each page, Quest computes an upper bound on the maximum attention weight any key in that page could produce:

```math
\text{score\_upper}(q, \text{page}) \;=\; \sum_{d=1}^{d_\text{head}} \max(q_d \cdot k_{\min, d}, \; q_d \cdot k_{\max, d})
```

This is the per-dimension maximum of two element-wise products. Geometrically: it's the highest possible dot product between `q` and any key whose components lie in the box `[k_min, k_max]`. The cost is `O(d_head)` per page — a single small inner product.

Then:

1. Compute `score_upper` for every page.
2. Take the top-K pages by `score_upper`.
3. Run attention only over those pages.

---

## The kernel integration

Quest needs two changes to the paged attention kernel:

- A page-scoring pre-pass that produces a sorted page list.
- A modified attention kernel that consumes a *selected page list* rather than the full block table.

The second is a near-trivial change in PagedAttention: pass `selected_block_table[batch, K]` instead of `block_table[batch, max_blocks]`, and iterate `K` blocks per query instead of `context_len / 16`.

Implementations exist in:

- The Quest authors' reference repo (https://github.com/mit-han-lab/Quest).
- FlashInfer's `BatchDecodeWithPagedKVCacheWrapper` with sparse-block extension.
- SGLang's experimental `quest_attention` mode.

---

## Empirical effect (paper §5)

| Model | Context | Full pages | Quest top-K | Latency speedup | LongBench Δ |
|---|---|---|---|---|---|
| LLaMA-3-8B | 32k | 2048 | 1024 | 2.2× | -0.4 |
| LLaMA-3-8B | 128k | 8192 | 1024 | 5.0× | -0.7 |
| Mistral-7B (LongChat) | 32k | 2048 | 1024 | 2.1× | -0.6 |
| Yi-6B-200k | 100k | 6250 | 2048 | 3.0× | -1.0 |

Three things to note:

- **Speedup is bandwidth-bound.** At 128k context, attention is ~90 % of decode latency; loading 12.5 % of pages gives ~5× decode speedup. FFN cost is unchanged.
- **Quality cost stays small at moderate K.** Top-1024 out of 8192 is a strong heuristic across LongBench tasks.
- **Memory savings: zero.** Quest does not evict; the full KV stays in HBM. This is a *bandwidth* optimization, not a *capacity* optimization.

---

## When to use Quest

**The fit:**

- Long-context workloads where decode latency dominates and HBM is not the binding constraint.
- Mixed-query workloads where the same long context is queried with different questions — Quest reselects per query.
- Deployments where the kernel can be modified (vLLM custom build, SGLang experimental).

**The limit:**

- HBM-bound deployments — Quest doesn't free any memory.
- Short contexts — page selection overhead exceeds savings below ~16k tokens.
- Workloads requiring guaranteed full-attention semantics — Quest is approximate.

---

## Why query-aware matters

Static eviction (H2O, SnapKV) implicitly assumes one importance ordering per request. For a *single-question* RAG against a long document, this is fine. For a long agent loop with many sub-questions against the same scratchpad, it's wrong — different sub-questions need different parts of the scratchpad.

Quest's per-decode-step re-selection is the answer: same KV stays in HBM, only the *loaded subset* changes. This is conceptually similar to how a database with a fixed table answers different SELECT queries by reading different rows.

---

## Composition with other methods

Quest composes cleanly with:

- **Paged KV (ch-06)** — Quest is built on top of the block table.
- **Prefix caching (ch-07)** — cached prefix blocks are subject to Quest's page selection just like fresh blocks.
- **SnapKV** — SnapKV compresses *which tokens exist*; Quest selects *which tokens load*. Composable: compress to 10k tokens with SnapKV, then load top-1024 of those with Quest.

It does *not* compose well with:

- **Eviction methods that change semantics each step (vanilla H2O)** — score accumulation under Quest's query-dependent loading would be biased.

---

## Connections

- [[excerpts/h2o]] — query-agnostic eviction; the alternative philosophy.
- [[excerpts/snapkv]] — one-shot compression; complements Quest.
- [[excerpts/attention-sinks]] — explains why sink pages will usually appear in Quest's top-K (their k_min/k_max boxes happen to overlap most queries).
- [[ch-08]] — parent synthesis.
