---
chapter: ch-10
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/scaling-laws-data-quality.md
source_url: https://arxiv.org/abs/2510.03313
created_at: "2026-04-23"
---

# Excerpt: Scaling laws revisited — data quality as a scaling variable

**Source library:** `wiki/raw-data/llm-training/papers/scaling-laws-data-quality.md`
**Paper:** Subramanyam, Chen, Grossman 2025, "Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining."

---

## Why this source anchors ch-10

This paper is the **theoretical counterpart** to the empirical filter-and-ablate work of CCNet, C4, Dolma, and FineWeb. Its central move is to extend Chinchilla-style scaling laws with an explicit **data-quality term**, arguing that two corpora with the same token count can sit on different loss curves if quality differs enough. Ch-10 §6 uses this as the frame for "quality is a scaling variable" — a filter that helps at 1B parameters may plateau at 70B, or vice versa, and you cannot rank pipelines without pinning both.

From the source (lines 6–7):

> **Core Insight:** Data quality can be treated as an explicit scaling variable, not just an anecdotal curation benefit.
>
> **Guideline:** When comparing data pipelines, model effective sample size and noise/deficiency explicitly instead of treating all tokens as equal.

The guideline is what makes this excerpt load-bearing for ch-10: "all tokens are equal" is the implicit assumption of a Chinchilla-style token-count comparison, and every pipeline in this chapter provides evidence that the assumption is false.

---

## The extension — quality enters the scaling law

From the source (lines 13–14):

> This paper extends standard language-model scaling-law thinking by adding a formal data-quality term. The central claim is that model loss should be understood as a function of model size, token count, and data quality jointly, with quality affecting the effective value of the data budget.

The formal move: Chinchilla's loss law is `L(N, D) = A/Nᵅ + B/Dᵝ + E`, where `N` is parameters and `D` is tokens. Subramanyam et al. introduce a quality term `Q` (a per-corpus scalar) and replace the token term with an **effective-token** term `D_eff = f(D, Q)` where `f` is monotone in both arguments but sub-linear in `D` when `Q` is low.

The practical consequence is that a corpus of `D` tokens at quality `Q` trains like a corpus of `D_eff < D` tokens at some reference quality. A pipeline that improves `Q` by a factor of 2× may be worth more than a pipeline that doubles `D` — and the trade-off depends on where on the scaling curve you are.

---

## The mechanism — effective sample size and deficiency

From the source (lines 21–23):

> - Models quality through effective sample size / deficiency-style terms.
> - Evaluates how corruption or redundancy changes the useful training signal.
> - Practical lesson: two corpora with the same token count can sit on different scaling curves if quality differs enough.

The "effective sample size" framing is borrowed from classical statistics — it is the sample size a noise-free corpus would need to convey equivalent information. Corrupted tokens (OCR noise, broken encoding, repeated boilerplate) have near-zero or negative effective sample contribution; high-quality tokens (Wikipedia, peS2o, FineWeb-Edu score-5 documents) have effective contribution approaching 1:1.

The deficiency term captures what dedup and quality filtering remove. Lee 2021 ([[deduplicating-training-data]]) showed that removing exact duplicates reduces memorization and improves eval — in Subramanyam's framing, the duplicate tokens had high `D` contribution but very low `D_eff` contribution, because they conveyed information the model had already extracted.

---

## Why this matters for ch-10's four pipelines

The paper makes explicit what ch-10's pipelines demonstrate empirically:

- **CCNet's head partition** is higher-`Q` than the middle or tail; using the head alone is a `D`-for-`Q` trade (fewer tokens, higher per-token signal).
- **C4's heuristic rules** reduce `D` (each rule drops documents) in exchange for an increase in `Q` (the survivors are more prose-like).
- **Dolma's ablation table** is a direct empirical measurement of each stage's `Q` contribution at a fixed 1B model size.
- **FineWeb-Edu at threshold 3** is the most aggressive `D`-for-`Q` trade in the chapter: 15T → 1.3T tokens (11×) for a measured MMLU gain. The paper's claim that this trade is positive is the claim that `D_eff(1.3T, Q_edu) > D_eff(15T, Q_base)` at the relevant model size.

---

## Why pipeline comparisons are scale-dependent

From the source (line 24):

> Practical lesson: two corpora with the same token count can sit on different scaling curves if quality differs enough.

The flip side: two **pipelines** can produce corpora on different curves at 1B and on the *same* curve at 70B, or vice versa. A filter that improves `Q` on narrow-knowledge benchmarks may help small models (which have little parametric knowledge to start) and plateau in large models (which have already absorbed the knowledge). A filter that tightens a corpus toward a specific distribution may help at large scale (where the model has capacity to exploit the distribution) and hurt at small scale (where the model needs breadth).

This is why ch-10 §6's checklist demands "ablations at the right scale" — a filter ablation at 150M parameters is not a filter ablation at 7B, and a ranking at 7B is not a ranking at 70B. The scaling-laws paper gives the theoretical reason the ranking is not scale-invariant.

---

## Why this paper is the right complement to FineWeb

[[excerpts/fineweb]] is the most aggressive empirical demonstration of quality-vs-quantity trade. Scaling-laws-data-quality is the theoretical framing that explains why 1.3T tokens at high `Q` can beat 15T tokens at lower `Q` on specific benchmarks at specific model sizes. The two papers are best read together:

- FineWeb: "here is the number — 1.3T at score ≥ 3 beats 15T."
- Scaling laws: "here is the reason — `D_eff(1.3T, Q_edu)` > `D_eff(15T, Q_base)` at the 1B–8B scales we can afford to test."

Without the scaling-laws frame, the FineWeb result reads as one data point. With it, it is an instance of a general relationship that can be extrapolated cautiously to other pipelines and other model sizes.

---

## What the paper does not do

- **It does not give a closed-form `Q`.** Quality is modeled as an abstract scalar, not as a prescription for how to measure it on a new corpus.
- **It does not replace ablations.** The scaling-law framework tells you the *shape* of the quality-vs-quantity trade, not the actual numbers on your corpus. Dolma's and FineWeb's ablation tables are still necessary.
- **It does not resolve the synthetic-data question.** [[excerpts/rephrasing-the-web]] raises the question of whether rephrased text has higher or lower `Q` than its source; the scaling-laws paper does not answer it.

---

## What to take from this paper for ch-10

1. **Token count is not enough.** A pipeline comparison that pins only `D` is under-specified; you need to pin `Q` (or its proxy — the downstream eval gap).
2. **`D_eff` explains why classifier-based pipelines beat heuristic-based pipelines on some benchmarks at some scales.** The classifier increases `Q` enough that the resulting `D_eff` rises despite the aggressive token discard.
3. **Rankings are scale-dependent.** Do not extrapolate a 1B ablation to a 70B model without a second ablation at intermediate scale.
4. **The framework is descriptive, not prescriptive.** It organizes the empirical findings in this chapter; it does not generate new pipelines.

---

## Connections

- [[excerpts/ccnet]] — head/middle/tail partition is a primitive `D`-for-`Q` lever.
- [[excerpts/c4]] — heuristic rules trade `D` for `Q` with no ablation to tell you the trade is positive.
- [[excerpts/dolma]] — per-stage ablation table is the empirical populator of the quality-scaling frame.
- [[excerpts/fineweb]] — the most aggressive `D`-for-`Q` trade in the chapter.
- [[excerpts/rephrasing-the-web]] — raises the question of whether synthetic rephrasing changes `Q` at fixed `D`.
- [[ch-10]] §6 (quality-vs-quantity at scale), §5 (comparison table).
