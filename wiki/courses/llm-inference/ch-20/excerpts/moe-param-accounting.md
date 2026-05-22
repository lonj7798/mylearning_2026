---
chapter: ch-20
course: llm-inference
phase: read
excerpt_of: "Cross-model MoE parameter accounting rules (Mixtral / Qwen-3 / DeepSeek-V3 / GPT-OSS)"
source_url: https://mistral.ai/news/mixtral-of-experts/
created_at: "2026-05-21"
---

# Excerpt: MoE parameter accounting — `8×7B ≠ 56B`

**Authors:** synthesis across model cards
**Year:** 2023–2025
**Raw-data source:** [[raw-data/mixtral-inference]], [[raw-data/qwen-3-inference]], [[raw-data/deepseek-v3-inference]], [[raw-data/gpt-oss-inference]]

---

## The rule

For every open MoE model, you must read **three** numbers:

- **Total params** = bytes of weights you must keep on GPUs/storage.
- **Active params per token** = bytes of weights touched per forward step (drives compute + bandwidth).
- **Per-expert FFN params** = building-block size (rarely advertised; usually inferred).

The names `8×7B`, `30B-A3B`, `671B / 37B active`, `117B / 5.1B active` are all this same triplet in different encodings.

---

## Mixtral 8×7B decomposed

| Component | Params | Notes |
|-----------|-------:|-------|
| Shared attention (QKV + O), 32 layers, GQA-8 | ≈ 4.2 B | one set, used by every expert |
| Embedding (32 k vocab × 4096) + LM head | ≈ 0.5 B | shared |
| LayerNorm + biases | ~0.1 B | shared |
| FFN per expert (3 matrices, intermediate 14336) | ≈ 4.2 B each | 8 experts → 8 × 4.2 = 33.6 B |
| **Total** | **≈ 46.7 B** | Mistral's reported number |
| Active per token (attention + 2 experts) | ≈ 4.2 + 0.5 + 2 × 4.2 = 12.9 B | |

The naive `8 × 7 = 56 B` overestimates because attention is shared. The correct mental model: **MoE replicates only the FFN sublayer per expert; attention is shared.**

---

## Qwen 3 30B-A3B decomposed

| | Value |
|---|---:|
| Layers | 48 |
| Hidden | 2048 |
| Q heads | 32 |
| KV heads | 4 (GQA-8) |
| FFN intermediate | 6144 |
| **Experts** | **128** |
| Experts active per token | 8 |
| Total params | ~30 B |
| **Active per token** | **~3 B** |

Note: 128 experts × 8 active is much more granular routing than Mixtral's 8×2. This finer routing improves specialisation but increases the all-to-all communication cost (more, smaller dispatches per layer). Serving stacks must use grouped/fused expert kernels (DeepEP, MegaBlocks) to hit decent throughput.

---

## DeepSeek V3 decomposed

| | Value |
|---|---:|
| Layers | 61 |
| Hidden | 7168 |
| Attention | MLA, latent 512 |
| Routed experts | 256 |
| Shared experts | 1 |
| Experts active per token | 8 (of 256) + 1 shared |
| Total params | 671 B |
| **Active per token** | **37 B** |

Two MoE design quirks worth knowing:
- **One shared expert always active**, in addition to 8 routed. The shared expert captures common-knowledge patterns; routed experts specialise. This is a DeepSeek-specific design pattern (also used in DeepSeek-V2).
- **Auxiliary-loss-free load balancing** — V3 dropped the standard auxiliary balancing loss in favour of a bias-shift trick. From a serving perspective this changes routing dynamics but not the per-token cost.

---

## GPT-OSS-120B decomposed

| | Value |
|---|---:|
| Layers | 36 |
| Experts | 128 |
| Experts active per token | 4 |
| Total params | 117 B |
| Active per token | 5.1 B |

GPT-OSS-20B uses the same shape with 32 experts (4 active) and proportionally smaller hidden/intermediate dims. The 4-of-N active rate is the same across both sizes.

---

## Side-by-side

| Model | Total | Active | Active / Total | Experts active / total | Attention |
|-------|------:|-------:|---------------:|-----------------------:|-----------|
| Mixtral 8×7B | 46.7 B | 12.9 B | 27.6 % | 2/8 | GQA-8 |
| Mixtral 8×22B | 141 B | 39 B | 27.7 % | 2/8 | GQA-8 |
| Qwen-3 30B-A3B | 30 B | 3 B | 10.0 % | 8/128 | GQA-8 |
| Qwen-3 235B-A22B | 235 B | 22 B | 9.4 % | 8/128 | GQA-8 |
| DeepSeek V3 | 671 B | 37 B | 5.5 % | 8/256 + 1 shared | MLA |
| GPT-OSS 20B | 21 B | 3.6 B | 17.1 % | 4/32 | GQA + banded |
| GPT-OSS 120B | 117 B | 5.1 B | 4.4 % | 4/128 | GQA + banded |

Two trends visible in this table:
1. **Active fraction shrinks as models scale.** Mixtral's 28 % was high (only 8 experts to choose from). The 2024–2025 generation moved to 10 % or less, getting capacity from many fine-grained experts.
2. **MLA appears at the largest scale.** DeepSeek V3 is the only entry where attention KV is also compressed — the natural next step when MoE has already cut weight cost as far as it goes.

---

## Why this matters for serving math

For a fixed GPU budget, **total params bound** what you can load; **active params bound** the compute per token. The throughput-per-dollar pitch of MoE is that you pay for total at deployment time (storage) and active at runtime (FLOPs + bandwidth).

But the math has a catch: **all-to-all communication for expert routing** scales with the number of expert dispatches per layer, not with active params. Qwen-3-30B-A3B has 8 dispatches per layer; Mixtral has 2. So Qwen-3's per-token compute (3 B active) is cheaper, but its per-layer routing overhead is 4× higher. On small batches the routing overhead dominates; on large batches the active-params compute dominates and Qwen-3's sparsity wins. This crossover happens around batch ≥ 16 on a typical H100 node.

---

## Connections

- [[excerpts/deepseek-mla-compression]] — V3's MLA is the attention-side complement to MoE's FFN sparsity.
- [[excerpts/gpt-oss-mxfp4-deployment]] — MXFP4 is the per-weight memory complement to MoE's per-token compute savings.
- [[ch-13]] — TP / PP / EP combinations for serving each of these MoE topologies.
