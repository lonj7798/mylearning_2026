---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: Subramanyam, Chen, Grossman 2025 — "Scaling Laws Revisited: Modeling the Role of Data Quality"
source_url: https://arxiv.org/abs/2510.03313
created_at: "2026-04-23"
---

# Excerpt: Quality as a Formal Scaling-Law Variable

**Source:** `wiki/raw-data/llm-training/papers/scaling-laws-data-quality.md`
**Primary paper:** Anirudh Subramanyam, Yuxin Chen, Robert L. Grossman, "Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining", 2025
**arXiv:** https://arxiv.org/abs/2510.03313

---

## Bibliographic header

Chinchilla's `L(N, D) = E + A/N^α + B/D^β` treats all tokens as equivalent. Every data-curation practitioner knows this is false — a FineWeb-Edu token is worth more than a raw CommonCrawl token — but the scaling-law literature did not formalise the difference until 2025. This paper does.

From the raw-data notes:

> *"Data quality can be treated as an explicit scaling variable, not just an anecdotal curation benefit. When comparing data pipelines, model effective sample size and noise/deficiency explicitly instead of treating all tokens as equal."*

The value of the paper is practical: it lets you compare two pretraining runs on different corpora with a shared scaling-law fit, rather than arguing by benchmark anecdote.

---

## The core extension

Standard Chinchilla:

```math
L(N, D) = E + A / N^\alpha + B / D^\beta
```

Subramanyam's quality-aware extension:

```math
L(N, D, q) = E(q) + A / N^\alpha + B / (\psi(q) \cdot D)^\beta
```

Two quality-dependent terms:

1. **`E(q)` — irreducible-loss term.** Depends on the inherent noise floor of the corpus. Junk-heavy corpora have higher `E(q)` — no amount of `N` or `D` drives the loss below this.
2. **`ψ(q)` — effective-sample-size multiplier.** A cleaner corpus acts like more tokens at the same raw count. A filtered-to-2T FineWeb-Edu corpus might have `ψ(q_edu) ≈ 1.5`, meaning it trains like a 3T unfiltered corpus would.

---

## The quality axis is not a multiplier — it is an asymptote

From the raw-data notes:

> *"Models quality through effective sample size / deficiency-style terms. Evaluates how corruption or redundancy changes the useful training signal."*

The subtlety: if quality only entered via `ψ(q)`, it would be a multiplier on `D` — you could compensate by scraping more low-quality tokens. It does not work that way. The `E(q)` term is an **asymptote**. Two corpora with different quality floors sit on *different scaling curves* that never cross. You cannot scale your way out of a bad corpus with more data or more parameters.

```
   loss
    |
    |                   E(low-quality)  ←---  asymptote you can never beat
    |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    |
    |        E(high-quality)
    |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    |
    +----------------------------→ log(N or D)
```

This is the 2025 statement of why FineWeb-Edu beats FineWeb at every scale, not just at small scale: they live on different asymptotes.

---

## Measuring ψ(q) in practice

The paper proposes several proxies for the quality factor `ψ(q)`:

1. **Held-out perplexity ratio.** Train small proxy models on candidate corpora; measure perplexity on a fixed held-out real-text test set. `ψ_proxy(q) = PPL_ref / PPL_candidate` normalised to a reference corpus. Cheap but fragile — susceptible to overlap artifacts.
2. **Downstream task scaling-law fit.** Fit `L(N, D, q)` to a sweep of small models; read off `ψ(q)` from the fit. Expensive but load-bearing — this is what the paper uses for its main claims.
3. **Classifier-score distribution.** For classifier-filtered corpora (FineWeb-Edu style), the classifier score distribution is itself a proxy for quality; corpora with higher mean/lower variance in educational-content scores have higher `ψ`.

The paper's empirical measurements: filtered web corpora (FineWeb-Edu vs FineWeb) give `ψ(q_edu) / ψ(q_plain) ≈ 1.3–1.5`; heavily-deduplicated corpora over raw scrapes give `ψ ≈ 1.1–1.2`; de-contaminated corpora over contaminated corpora show a smaller `ψ` gain (~1.02–1.05) but a large `E(q)` improvement at benchmark-specific losses.

---

## Composition with the data-constrained law

Stacking with Muennighoff's `D' = U (1 − exp(−R/R_T))`:

```math
D_{\text{eff}} = \psi(q) \cdot U \cdot (1 - \exp(-R / R_T))
```

```math
L(N, D_{\text{eff}}, q) = E(q) + A/N^\alpha + B/D_{\text{eff}}^\beta
```

**Key consequence:** high-quality repeats are worth more than low-quality novel tokens until the repetition-saturation dominates. Concretely: if `q_old/q_new = 2`, one repeat of `q_old` data beats one fresh `q_new` token up to about epoch `R_T · ln(2) ≈ 2.8`. Past that, the Muennighoff decay kicks in and fresh noisy tokens become preferable.

This is the mathematical justification for the OLMo 2 / OLMo 3 cooldown design: a small high-quality corpus (Dolmino) at many epochs beats a large low-quality corpus at 1 epoch for the final-quality cooldown stage.

---

## What the paper does not resolve

From the raw-data notes:

> *"Practical lesson: two corpora with the same token count can sit on different scaling curves if quality differs enough."*

The paper gives a framework but leaves three open problems:

1. **No universal `q` scalar.** `q` is really multi-dimensional (factual density, linguistic diversity, domain coverage, error rate). The paper lumps them into a scalar proxy. Task-specific applications may need per-dimension fits.
2. **No recipe for what quality operations give the largest ψ lift.** The fit is descriptive, not prescriptive — it tells you two corpora differ but not which curation step was responsible.
3. **Contamination is treated as a subcase of quality.** But as Dohmatob 2024 shows ([[strong-model-collapse]]), synthetic contamination *flatlines* scaling laws rather than shifting them. The two failure modes are not interchangeable.

---

## Connections

- The original data-constrained paper: [[excerpts/data-constrained-scaling]]
- Empirical quality curation: [[fineweb]] (classifier approach), [[dolma]] (cascade approach)
- The contamination side of the quality question: [[excerpts/model-collapse]]
- Chapter synthesis: [[ch-14]]
