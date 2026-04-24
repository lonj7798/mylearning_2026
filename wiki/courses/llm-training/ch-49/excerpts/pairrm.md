---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/pairrm.md
source_url: https://arxiv.org/abs/2306.02561
created_at: "2026-04-23"
---

# Excerpt: PairRM — joint-input encoding + swap-augmentation as standard practice

**Source library:** `wiki/raw-data/llm-training/papers/pairrm.md`
**Authors:** Dongfu Jiang, Xiang Ren, Bill Yuchen Lin (LLM-Blender, ACL 2023)
**Year:** 2023

---

## Why this source matters for ch-49

PairRM is the cheapest calibrated judge on the market. 0.4B params, encoder-only, matches scalar RMs at 7B. For ch-49 it plays two roles: (a) it is where **swap-augmentation** becomes a named standard, and (b) it is the reference "fast per-comparison" judge to contrast with generative judges in §6 (RL-time RM vs eval-time judge split).

---

## Swap-augmentation as the standard protocol

Source §Key Contributions:

> "Swap-augmentation: always evaluate `(y_A, y_B)` and `(y_B, y_A)`, average logits -- cancels position bias at train and inference time."

And §Technical Details:

> "Failure modes: inherits verbosity and position bias to some extent; swap-augmentation handles position but not verbosity -- explicitly length-balance training pairs."

Ch-49 §4's "swap protocol" artifact is cited from this paper. The fact that swap-augmentation is *standard practice* rather than an add-on comes from PairRM normalizing it; later GenRM pipelines inherit the convention.

---

## Joint-input vs scalar construction

Source §Key Contributions:

> "Joint encoding: `f(x, y_A, y_B) -> logit`; cross-attention sees both responses at once."

Ch-49 §5 uses this distinction: "Judge-RM vs generative-RM" is partly the joint-input-vs-scalar distinction. Joint encoding lets self-attention compare both responses at once, catching subtle differences that a per-response score would miss. GenRMs inherit this by *concatenating* both responses into the critique context.

---

## The size-efficiency claim

Source §Abstract:

> "A 0.4B DeBERTa-based PairRM beats scalar RMs based on Llama-2-7B on MixInstruct and MT-Bench reranking."

This is why PairRM is the RL-time reward model of choice in many open stacks: per-comparison cost is 20× lower than a 7B GenRM. Ch-49 §6 uses this to motivate why RL-time RM need not be the same as eval-time judge — they have different compute constraints.

---

## Tournament Best-of-N

Source §Key Contributions:

> "Tournament Best-of-N: for N candidates run O(N log N) pairwise comparisons, advance winners -- avoids O(N^2) full pairwise pass, retains near-optimal selection."

Not directly used in ch-49 but structurally important: Best-of-N with a pairwise judge is log-time in N. Ch-49 §6's runtime-reward use case inherits this scaling.

---

## DPO pair filtering

Source §Technical Details:

> "DPO pair filtering usage: keep `(y_w, y_l)` pairs where `PairRM(y_w, y_l) > tau`; this is a simple quality gate that has been shown to lift DPO performance ~2 pp on held-out evals."

Ch-49 does not dwell on this because it is a training-time use, but it anchors the "different judges at different phases" argument — the PairRM that filters training pairs is not the same judge as the GenRM that scores checkpoints, even if both are calibrated.

---

## What swap does *not* fix

Source §Technical Details:

> "Failure modes: inherits verbosity and position bias to some extent; swap-augmentation handles position but not verbosity -- explicitly length-balance training pairs."

Ch-49 §3 applies this: swap fixes *position*, length-balanced pairs fix *verbosity*, and they are different interventions. Neither fixes the other. The §3 table keeps them as separate rows for this reason.

---

## Comparison with GPT-4

Source §Key Figures/Tables:

> "Table comparing PairRM vs GPT-4 as judge -- PairRM-0.4B is within a few points of GPT-4 on tight pairs, at a fraction of cost."

Ch-49 §5 cost argument cites this: a 0.4B joint encoder closes most of the gap to GPT-4 at a fraction of the API bill, which means GPT-4-as-judge's cost premium is hard to justify for many use cases even without the leakage argument.

---

## Connections

- `read.md` §4 swap protocol: named and standardized by this paper.
- `read.md` §5 method-contrast table: PairRM row.
- `read.md` §6 RL-time RM vs eval-time judge split: PairRM is the canonical fast-RL-time judge.
- `read.md` §3 position-bias row: swap-augmentation correction quoted from here.
- [[generative-reward-models]]: inherits joint-input structure and swap-augmentation convention.
