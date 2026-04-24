---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Ding et al. — "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens"
source_url: https://arxiv.org/abs/2402.13753
created_at: "2026-04-23"
---

# Excerpt: LongRoPE — per-dimension RoPE search and the <1B-token fine-tune budget

**Source:** `wiki/raw-data/llm-training/papers/longrope-data.md`
**Paper:** Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, Mao Yang (Microsoft Research Asia), 2024
**arXiv:** https://arxiv.org/abs/2402.13753

---

## Bibliographic header

> *"Context extension is not purely a data problem — the right RoPE-extension scheme (non-uniform per-dimension base rescaling found via evolutionary search) is just as important; LongRoPE demonstrates 2M+ token context on LLaMA-2 with only a small fine-tuning corpus (< 1B tokens) by decoupling the RoPE math from the data volume."*

This is the paper that makes position-encoding its own lane. Every prior recipe (ProLong, Llama-3, Fu 2024) tuned the RoPE base as a scalar; LongRoPE generalises to a 128-dimensional search.

---

## The core equation

From the raw-data:

> *"Standard RoPE applies rotation at frequency θ_i = θ^(2i/d) for dimension i. NTK-aware scales θ → θ · s^(d/(d-2)). LongRoPE generalizes: θ_i' = θ_i / λ_i where λ_i is a per-dimension rescaling factor learned via evolutionary search. Uniform YaRN is the special case λ_i = constant."*

Written cleanly:

```math
θ_i' = θ_i / λ_i,    i = 0, 1, ..., d/2 - 1
```

- Uniform NTK-aware: `λ_i = λ` (one scalar).
- YaRN: piecewise-constant `λ_i` (low-freq dims scaled, high-freq dims unchanged).
- LongRoPE: **every `λ_i` learned independently**, 64 values for a `d = 128` head.

**Notice:** this is the natural generalisation that YaRN hinted at. YaRN correctly recognised that high-freq and low-freq dims should be rescaled differently; LongRoPE extends this from two regions to 64.

---

## Why per-dimension wins

Intuition. RoPE's frequency decays as `θ^(-2i/d)`. At the high-frequency end (small `i`), one period of rotation covers only a few tokens; those dims already *saturate* within the original training range. Rescaling them aggressively would destroy the information they encode. At the low-frequency end (large `i`), one period covers thousands of tokens; those dims *don't saturate* until far into the target context, and rescaling them harder pushes aliasing past the target length.

A uniform scalar `λ` is a compromise: either it's large enough to fix the low-freq dims (and it ruins the high-freq dims) or it leaves the high-freq dims alone (and it fails at the target length). Per-dimension `λ_i` resolves the tradeoff.

---

## The evolutionary search

> *"Search space: per-dimension rescaling factor λ_i for each of the 128 RoPE dimensions. Fitness function: perplexity on a held-out long-context corpus + NIAH retrieval accuracy. Algorithm: evolutionary strategy with population size 64, 40 generations, mutation rate 0.3. Initial population: NTK-aware, YaRN, and uniform rescaling schemes seeded."*

Walk the protocol:

1. **Population** = 64 candidate `λ_i` vectors.
2. **Seed** = NTK-aware, YaRN, uniform — a reasonable prior so the first generations aren't random noise.
3. **Fitness** = weighted combination of long-ppl and NIAH accuracy. Each evaluation is a long-context forward pass — expensive but discrete.
4. **Generations** = 40.
5. **Mutation rate** = 0.3 per-dim.

Total fitness evaluations in the worst case: `64 × 40 = 2560`. In practice, with elite retention, effective evaluations are closer to `64 + 63 × 40 ≈ 2600`. Each evaluation on a 256K-context window is non-trivial — tens of minutes of GPU time — so the full search is **~1000 GPU-hours order of magnitude**.

That's expensive, but it amortises: the `λ_i` vector is computed **once per base model**, not per training run. And the payoff is large: **10× reduction in fine-tune data**.

---

## The two-stage progressive extension

> *"Two-stage fine-tuning: Stage 1 — 256K extension (~300M tokens, long docs from Books / ArXiv / long web); Stage 2 — 2048K extension (~600M tokens, concatenated multi-document long sequences)."*

Compare to the data-lane recipes:

| Recipe | Fine-tune tokens | Final context |
|---|---|---|
| Fu 2024 | 5B | 128K |
| ProLong | 20B + 5B | 512K |
| Llama 3 | ~800B (CPT) | 128K |
| **LongRoPE** | **< 1B** | **2M** |

The orders-of-magnitude gap is real — but the caveat is that LongRoPE's evaluation is heavy on NIAH and passkey retrieval, lighter on BABILong-style reasoning. The paper explicitly notes:

> *"Real reasoning at 2M is weak: NIAH passes, but complex multi-hop over 2M context is not reliably solved."*

Position-encoding extension and data coverage are *complementary*, not substitutes. LongRoPE gets you to 2M on retrieval; to get 2M on reasoning you still need long-coherent-doc training data, which doesn't exist at 2M-token-per-document scale.

---

## Search cost as a one-time amortised price

The usual counterargument: "2560 fitness evaluations is a lot." But the `λ_i` vector is model-and-target-context-specific, not training-run-specific. Search once for your base model at your target context; reuse the `λ_i` across many fine-tuning experiments. That changes the cost accounting.

Compared to, say, Llama-3's 800B-token continued pretraining, a 2560-evaluation search is sub-0.1% of the compute — even though it feels expensive locally.

---

## What doesn't generalise

- **Non-uniform λ may generalize poorly** to very different base models. The search must be repeated when you swap base models (the per-dim profile depends on the head dim, the pretraining base, and the pretraining length distribution).
- **Fitness function is NIAH-heavy.** A fitness that weighted BABILong reasoning more might produce a different `λ_i` — one the paper didn't explore.
- **Search is sequential.** Parallelising the 64-member population helps, but 40 generations are serial.

---

## Connections

- Chapter synthesis: [[ch-28]]
- Data-lane counterparts: [[excerpts/prolong-coherence]], [[excerpts/llama3-staged-schedule]]
- PoSE position-skip alternative (much cheaper, weaker): see ch-28 §5
- Used inside Qwen 1M pipeline: [[excerpts/qwen-1m-pipeline]]
