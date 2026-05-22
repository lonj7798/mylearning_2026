---
chapter: ch-08
course: llm-inference
phase: read
excerpt_of: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models (Zhang et al. 2023)"
source_url: https://arxiv.org/abs/2306.14048
created_at: "2026-05-21"
---

# Excerpt: H2O — heavy-hitter KV eviction

**Authors:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, Zhangyang Wang, Beidi Chen
**Year:** 2023 (NeurIPS)
**URL:** https://arxiv.org/abs/2306.14048
**Raw-data source:** [[raw-data/h2o]]

---

## The thesis

> "We observe that a small portion of tokens contributes most of the value when computing attention scores. We call these tokens Heavy Hitters (H₂). Through a comprehensive investigation, we find that (i) the emergence of H₂ is natural and strongly correlates with the frequent co-occurrence of tokens in the text, and (ii) removing them completely results in significant performance degradation." (§1)

The paper's core empirical claim: at any moment during generation, ~5 % of past tokens account for ~95 % of attention mass at most layer/head pairs (their Figures 2 and 3 on OPT-6.7B). If you can identify those tokens cheaply at runtime, you can evict the other 95 % and keep nearly the same generations.

---

## The accumulated-attention scoring rule

Each cached token `i` has a running score `s_i`. After each decode step's softmax produces attention weights `a_i`:

```math
s_i \;\leftarrow\; s_i + a_i
```

`s_i` is the total mass token `i` has ever received from any decode step in this sequence. The intuition: heavy hitters are tokens that receive non-trivial attention *often*, not just once.

---

## The eviction policy

Given:
- Cache budget `B` (max tokens to retain).
- Recent window `r` (always-kept most-recent tokens).

After each step:

```python
def h2o_evict(cache, B, r):
    if len(cache) <= B: return cache
    recent = cache[-r:]                          # always keep last r
    old    = cache[:-r]
    old    = sorted(old, key=lambda t: -t.score)[:B - r]  # top by accumulated score
    return old + recent
```

Two design choices the paper validates with ablations (Tables 3, 4):

- **Recent window is critical.** Without it (`r=0`), local coherence breaks — the cache becomes only globally-important tokens, losing recency. Quality drops sharply.
- **The score is *accumulated*, not *current*.** Per-step top-k would oscillate; accumulation gives a stable importance signal.

---

## Theoretical framing

The paper proves a *submodular* coverage bound: the heavy-hitter selection approximately maximizes a function whose value bounds the deviation of the truncated-cache output from the full-cache output (§3.2, Theorem 1). The bound holds under a "sparsity assumption" — there exists a small set `S` such that attention restricted to `S` differs from full attention by ≤ ε. Empirically this assumption holds across many layers / heads of OPT, LLaMA, and GPT-NeoX.

The submodularity matters because it justifies the greedy top-k selection — greedy on a submodular objective is (1−1/e)-approximate optimal.

---

## Empirical results (paper §5)

| Model | Cache budget (of full) | Task quality | Throughput vs full |
|---|---|---|---|
| OPT-6.7B | 20 % | -0.1 ppl on WikiText | 3× on A100 |
| OPT-30B | 20 % | -0.2 ppl on WikiText | 3.6× on 4×A100 |
| LLaMA-7B (paper appendix) | 20 % | -0.4 ppl | 2.8× |
| GPT-NeoX-20B | 20 % | within 1 % MMLU | 3.2× |

Throughput gains come from two sources:

1. **Lower per-step KV bandwidth.** Reading 20 % of the KV is faster.
2. **Larger batch size.** Less KV per request → more concurrent requests fit in HBM.

The paper reports the second effect dominates above batch size ~16.

---

## Where H2O fits in production

H2O is the canonical *runtime-eviction* reference, and the baseline every later KV-compression paper benchmarks against. Production deployments rarely use vanilla H2O because:

- The per-step bookkeeping (score updates, sort, evict) adds CPU overhead.
- The eviction decision changes per layer / per head ideally, but vanilla H2O uses one budget across the layer (simpler, less accurate).
- Newer methods (SnapKV, Quest) compress differently with similar or better quality.

But its *idea* — attention is sparse, evict by accumulated importance — underlies all of them. Reading H2O is the cheapest way to internalize why KV compression works at all.

---

## Connections

- [[excerpts/snapkv]] — moves the importance estimation to the *end of prefill* rather than during decode; simpler runtime, comparable quality.
- [[excerpts/attention-sinks]] — argues that some positions (early ones) should always be kept regardless of accumulated score; explains an H2O failure mode.
- [[excerpts/quest-kv]] — avoids eviction entirely by making the importance decision per-query at decode.
- [[ch-08]] — parent synthesis.
- Forward to long-context-inference material — KV quantization / MLA / learned compression operate at a different layer than H2O's "which tokens to keep."
