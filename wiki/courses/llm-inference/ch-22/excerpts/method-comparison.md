---
chapter: ch-22
course: llm-inference
phase: read
excerpt_of: "Side-by-side comparison of the five capstone method options"
source_url: synthesis across raw-data papers
created_at: "2026-05-21"
---

# Excerpt: Capstone method comparison + reproduction-risk scoring

**Authors:** synthesis across [[pagedattention]] / [[sglang-radixattention]] / [[speculative-decoding]] / [[eagle]] / [[distserve]]
**Raw-data sources:** referenced inline below

---

## At-a-glance

| | Surface | Where the work is | Code volume | Headline metric | Hardware floor | Risk |
|---|---|---|---|---|---|---|
| **PagedAttention** | KV-cache layout | Cache + attention kernel | 500–1500 LOC | 96 % mem utilisation | 1 × H100 | Low (memory metric only) |
| **RadixAttention** | KV-cache reuse | Tree DS + scheduler hook | 800–2000 LOC | 5× shared-prefix tput | 1 × H100 | Med (DS correctness) |
| **Speculative Decoding** | Decoding loop | Generation wrapper | 200–500 LOC | 2–3× TPOT | 1 × H100 (target + draft) | Low-med (acceptance test math) |
| **EAGLE** | Decoding loop + small training | Draft head + tree verify | 1000–2500 LOC | 3× TPOT, 80% accept | 1 × H100 (+ train compute) | High (training needed) |
| **Disaggregated P/D** | Serving architecture | Two-process orchestration | 1500–4000 LOC | Goodput@SLO delta | **2 × H100** | High (systems plumbing) |

---

## Picking by what you want to learn

**If your background is systems and you want to do KV-cache research:**
Pick **PagedAttention** or **RadixAttention**. Both are data-structures + cache-coherence problems wearing GPU-serving clothes. PagedAttention is simpler (memory metric only, no kernel-perf needed); RadixAttention has the more interesting algorithmic core (radix tree with reference counting + LRU + split correctness).

**If your background is ML / decoding and you want to do speculative-decoding research:**
Pick **Speculative Decoding** first, **EAGLE** second. SpecDec lets you internalise the lossless-sampling math without the distraction of training a draft. EAGLE adds the training piece — and the training is the *interesting* part because it's where the draft quality (acceptance rate) actually comes from.

**If your background is distributed systems and you want to do serving-architecture research:**
Pick **Disaggregated Prefill/Decode**. This is the option that exercises NCCL, multi-process orchestration, KV-transport overlap with compute, and SLO-aware goodput measurement.

---

## Risk-vs-reward profile per option

### PagedAttention — low risk, well-trodden

**Why low risk.** The paper's headline metric (96 % memory utilisation) is reproducible without writing a fast kernel. The reference PyTorch attention with block-table indirection is correct (just slow), and the memory accounting is independent of kernel speed. You can write a 1000-line reproduction in 2 days that *matches the headline*.

**Why limited reward.** If you only report memory utilisation, the result is uninteresting (everyone knows paging works). The interesting deepening is the *throughput* claim, which requires a Triton/CUDA kernel. Budget another 3–5 days for that.

### RadixAttention — medium risk, big reward on shared-prefix workloads

**Why medium risk.** The radix-tree data structure is a known algorithm but has nasty edge cases: split-on-divergence, eviction-while-locked, prefix-of-cached-prefix. Each is one bug.

**Why big reward.** The 5× speedup on shared-prefix workloads is real and measurable; the reproduction lets you also explore *why* it falls back to ~1× on no-shared-prefix workloads. The cross-workload comparison is excellent memo material.

### Speculative Decoding — low risk, modest reward

**Why low risk.** Pure software, no kernels, no training. The whole reference implementation is ~300 lines of clean PyTorch.

**Why modest reward.** The 2–3× speedup is achievable but workload-sensitive: chat (lots of stop-word agreement) speeds up well; code (token-level divergence) less so. Your memo will benchmark across both and report the spread.

### EAGLE — high risk, high reward

**Why high risk.** The draft head must be *trained*. Without training, acceptance ≈ 0. Training adds another moving part (data, loss, learning rate) that the paper doesn't dwell on. Plan for two days of training-loop debugging.

**Why high reward.** Reproducing EAGLE means you've reproduced training-side and inference-side both. The cross-architecture generalisation question (does EAGLE's 3× hold on GQA models?) is a publishable workshop topic.

### Disaggregated Prefill/Decode — high risk, high systems reward

**Why high risk.** Two-process orchestration. NCCL setup. KV transport scheduling. SLO-met goodput measurement. Each is one weekend of setup pain.

**Why high reward.** This is the most production-relevant option. The reproduction directly informs whether your team should build a disaggregated serving stack. The memo's "where my numbers diverge" section will probably name interconnect bandwidth as the key axis — which is exactly the practical answer the field needs.

---

## What each option does NOT teach

- **PagedAttention** does not teach you about scheduling. The paged cache is a layer below the scheduler.
- **RadixAttention** does not teach you about decoding acceleration. Prefix reuse only helps the prefill phase.
- **SpecDec / EAGLE** do not teach you about memory. Both methods slot into existing cache management; the cache is unchanged.
- **DistServe** does not teach you about kernels. The win comes from architecture, not from any kernel optimization.

The optimal pick depends on which gap in your skill set the capstone should fill.

---

## What to read before starting

| If you picked... | Read first |
|------------------|------------|
| PagedAttention | [[pagedattention]] §3-4 + the vLLM `kv_cache_manager.py` source as a sanity check |
| RadixAttention | [[sglang-radixattention]] + the SGLang `radix_cache.py` source |
| Speculative Decoding | [[speculative-decoding]] + [[fast-inference-from-transformers-via-speculative-decoding]] (read both — the second has the lossless proof) |
| EAGLE | [[eagle]] + [[fast-inference-from-transformers-via-speculative-decoding]] (foundation) |
| DistServe | [[distserve]] + [[mooncake]] (production extension) |

**Read the paper, NOT the source first.** The point of the capstone is the paper-to-pseudocode translation. Reading the source first ruins the exercise.

---

## Connections

- [[ch-06]] / [[ch-07]] / [[ch-09]] / [[ch-14]] / [[ch-15]] — the chapter each option's background was introduced in.
- [[excerpts/pagedattention-reference]] — a concrete PagedAttention reference skeleton.
- [[excerpts/specdec-acceptance]] — the acceptance test worked through for SpecDec.
- [[excerpts/debugging-tree]] — diagnostic tree for "my numbers don't match the paper."
