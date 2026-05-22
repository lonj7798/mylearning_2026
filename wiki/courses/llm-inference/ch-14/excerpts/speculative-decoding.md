---
chapter: ch-14
course: llm-inference
phase: read
excerpt_of: "Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation (Xia et al. 2022)"
source_url: https://arxiv.org/abs/2203.16487
created_at: "2026-05-21"
---

# Excerpt: SpecDec — The Original Draft-and-Verify

**Authors:** Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, Zhifang Sui
**Year:** 2022
**Venue:** EMNLP 2023 (originally arXiv 2022)
**URL:** https://arxiv.org/abs/2203.16487
**Raw-data source:** [[raw-data/speculative-decoding]]

---

## The framing

SpecDec is the **first formal application of speculative execution to sequence generation**. The framing is exactly the one CPU branch predictors use: a cheap, possibly-wrong path runs ahead, and a expensive, correct path verifies and corrects in batch.

The control flow:

```
1. Spec-Drafter: a cheap (non-autoregressive or small AR) model proposes K future tokens.
2. Spec-Verification: the target AR model evaluates the drafted positions in one parallel
                      forward pass (because transformers naturally produce next-token
                      distributions at every input position).
3. Accept the longest prefix that matches the target model's decisions; on first mismatch,
   replace with the target's own token and restart drafting.
```

Key insight: **a transformer forward pass over `n` tokens produces `n` next-token predictions** — usually only the last is consumed. SpecDec turns the other `n-1` into free verification.

---

## What's new vs Xia's predecessors

Speculative decoding for sequence generation had been informally explored, but Xia formalizes:

- **Draft-then-verify as a generic control flow** for autoregressive generation, not tied to a specific drafter architecture.
- **Spec-Drafter design** — a non-AR transformer trained to predict K tokens at once from the encoder representation (in the NMT setting where SpecDec originated).
- **Spec-Verification** as a parallel target forward pass with a prefix-matching acceptance rule.
- **Latency tables** demonstrating real speedups (e.g. ~5× on NMT vs greedy AR baseline).

The framing — drafter / verifier / accepted prefix — is now the standard vocabulary used by Medusa, EAGLE, Lookahead, and every modern speculative method.

---

## The greedy acceptance rule (SpecDec original)

For greedy decoding:

```
For each drafted token x'_i, i = 1..K:
    if target_argmax_at_position(i) == x'_i:
        accept x'_i
    else:
        commit target_argmax_at_position(i) instead
        discard x'_{i+1..K}
        restart drafting
```

This trivially preserves greedy outputs — accepted tokens are exactly what the target would have argmax'd anyway.

The sampling generalization (with `min(1, p/q)` and residual) came one year later in Leviathan-Kalman-Matias 2023 ([[excerpts/leviathan-2023]]).

---

## Why "speculative execution" is the right metaphor

CPU branch predictors run instructions speculatively past a branch; if the branch resolves to a different target, the speculative work is squashed. The cost of a misprediction is the squashed pipeline depth.

In SpecDec: the drafter "predicts the branch" (next K tokens), the target verifies, and a mismatch squashes the drafter's remaining K-i tokens. The cost of a misprediction is the wasted drafter work for tokens beyond the first miss — but the target's verification cost is **paid once regardless**, because the target forward pass is parallel over all K positions.

The asymmetry — verification is fixed-cost-per-round, drafter is variable-cost-per-token — is what makes the math favorable when `α` is high.

---

## Why it works for autoregressive transformers specifically

Two architectural properties:

1. **Causal mask makes verification position-local.** The target's distribution at position `i` depends only on tokens at positions ≤ `i`. So computing all K positions in parallel is just one masked forward pass, no extra dependencies.
2. **Memory-bandwidth-bound decode** ([[kv-cache-memory-formula]] from ch-03): the per-token forward cost is dominated by loading the model's weights from HBM. K-token verification loads the weights *once*, so its cost is ≈ 1× single-token decode, not K×. This is the source of the speedup.

Without property 2, spec-dec would be a wash. Because decode is bandwidth-bound, K verifications cost roughly as much as 1 — and that's the source of the speedup.

---

## Connections

- [[excerpts/leviathan-2023]] — lossless sampling generalization (the version everyone cites).
- [[excerpts/hf-assisted-generation]] — the library API exposing this method.
- [[ch-15]] — modern variants (Medusa, EAGLE, Lookahead, PLD, MTP) all extend this skeleton.
- [[ch-14]] — parent chapter.
