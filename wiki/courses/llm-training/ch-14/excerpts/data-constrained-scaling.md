---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: Muennighoff et al. 2023 — "Scaling Data-Constrained Language Models"
source_url: https://arxiv.org/abs/2305.16264
created_at: "2026-04-23"
---

# Excerpt: Muennighoff's Data-Constrained Scaling Law

**Source:** `wiki/raw-data/llm-training/papers/data-constrained-scaling.md`
**Primary paper:** Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, Colin Raffel, "Scaling Data-Constrained Language Models", 2023
**arXiv:** https://arxiv.org/abs/2305.16264

---

## Bibliographic header

The paper trains 400 language models ranging from 10M to 9B parameters with token budgets of up to 900B tokens drawn from up to 178B unique tokens. The contribution is a *modified* Chinchilla scaling law that replaces token count `D` with an **effective-token count `D'`** that saturates with repetition.

From the raw-data notes:

> *"Data-constrained scaling studies how language-model training changes when the amount of unique high-quality data becomes a bottleneck. The core finding is that the value of repeating existing data versus adding lower-quality new data depends on where the model sits in the compute-data regime."*

This is the single paper that underlies every "how many epochs should I train?" decision in 2024–2025 frontier pretraining.

---

## The core formula

The standard Chinchilla law:

```math
L(N, D) = E + A / N^\alpha + B / D^\beta
```

Muennighoff's data-constrained extension replaces the total-token count `D` with an **effective count** that captures the diminishing returns of repetition:

```math
D' = U \cdot \left(1 - \exp\left(-\frac{R}{R_T}\right)\right)
```

where:
- `U` = unique token count in the training corpus
- `R = D / U` = number of epochs (may be fractional)
- `R_T` = fitted "token half-life" in epochs; empirically `R_T ≈ 4`

Equivalently, the marginal value of the `k`-th repetition of a token is:

```math
w(k) = \exp\left(-\frac{k-1}{R_T}\right)
```

So the effective-token count is the sum of geometrically-decaying repeat values:

```math
D' = U \cdot \sum_{k=1}^{R} w(k) \;\approx\; U \cdot \int_0^R e^{-r/R_T}\,dr = U \cdot R_T \cdot (1 - e^{-R/R_T})
```

Note the left-hand side uses `R_T` as the scale; the integral form shows why `R_T` is called a "half-life" even though the decay is not binary (it is the characteristic scale of an exponential).

---

## Reading the decay constant

With `R_T = 4`:

| `k` (epoch) | `w(k)` | Cumulative `D'(R) / U` |
|---|---|---|
| 1 | 1.000 | 0.221 |
| 2 | 0.779 | 0.393 |
| 3 | 0.607 | 0.528 |
| 4 | 0.472 | 0.632 |
| 5 | 0.368 | 0.713 |
| 8 | 0.174 | 0.865 |
| 12 | 0.050 | 0.950 |
| 20 | 0.008 | 0.993 |

**Three inflection points to memorise:**
- `R = R_T = 4` → 63% of the asymptote absorbed. This is the "knee" of the curve — repetition past this point is deliberately cheap.
- `R = 3 · R_T = 12` → 95% absorbed. Past here, fresh tokens (even noisy) dominate repetition.
- `R = 5 · R_T = 20` → 99.3% absorbed. This is the wasted-compute regime.

**Notice:** the formula does not claim that epoch 5 is "free" — it claims epoch 5 buys you 37% of what epoch 1 did. The integral is not zero; it is just diminishing.

---

## The data-vs-parameter tradeoff under repetition

Substituting `D'` back into the scaling law gives a *modified* compute-optimal boundary. Under pure Chinchilla, compute-optimal says `D_opt / N_opt ≈ 20`. Under data-constrained scaling with `U` fixed, beyond `R = R_T`:

```
dL/dN  ∝  −A α / N^(α+1)
dL/dD' ∝  −B β / D'^(β+1)
dD'/dD ∝   exp(−D/(U · R_T)) / (U · R_T)       # chain rule
```

As `D → ∞`, `dD'/dD → 0` — additional raw tokens stop buying effective tokens. So the compute-optimal model shifts: you want a *larger* model, not more repetitions. The paper states this explicitly:

> *"Practical implication: 'more tokens' is ambiguous if many are repeats."*

The planning consequence: if your `U` is 1T and you want to do a compute-optimal run at 3e25 FLOPs, you spend compute on a larger `N` rather than pushing `R` past 4–8.

---

## Empirical measurements supporting R_T ≈ 4

The paper fits `R_T` from a grid of (N, D, U) sweeps at 400 training runs. The fitted value varies slightly with model size — `R_T ≈ 3.5–4.5` across the 10M–9B range — with the midpoint estimate of 4 used as the canonical constant. The fit is remarkably stable: within the measurement range, no setting shows `R_T` dropping below 2 or exceeding 6.

The paper also measures the alternative — *adding noisier new data* — and fits a quality-equivalence function. The summary:

> *"Compares regimes with more unique data versus more repeated passes over fixed corpora."*

Rough equivalence: 1 fresh token at filtering quality `q_new` is worth approximately `q_new / q_old` repeated tokens at the old corpus's quality. If your new scrape is 30% as clean as your old corpus (`q_new/q_old = 0.3`), repeating existing data beats adding new data until `w(k) < 0.3`, i.e., until epoch `k ≈ 1 − R_T · ln(0.3) ≈ 5.8`.

---

## The corner that the paper warned about

From the raw-data notes:

> *"Important for frontier training because unique high-quality corpora are finite."*

At the time (May 2023), this was a theoretical warning. By 2024 it was the binding constraint for every frontier lab. Llama 3's 15.6T token budget is almost exactly the total amount of high-quality English text that exists publicly; beyond it, Meta reports diminishing returns per additional scraped token. The data-constrained regime is no longer a small-scale curiosity — it is the default.

---

## Where the formula breaks

The paper is explicit about the regime of validity:
1. **Fixed-quality corpus.** The formula assumes `U` tokens are at uniform quality. Mixing high- and low-quality tokens requires the quality-aware extension ([[scaling-laws-data-quality]] / this chapter §2).
2. **No retention bound.** The formula is a loss-scaling fit. It says nothing about whether the model *retains* the rare facts seen only a few times — that is Allen-Zhu's territory (see [[physics-of-lm-3]]).
3. **No contamination.** The fit assumes `U` unique tokens are *independent* samples of the target distribution. When the corpus is contaminated with synthetic data, the effective-token count drops further (see [[model-collapse]] / [[strong-model-collapse]]).

Engineers in 2025 typically cap `R` at 4–8 for bulk pretraining and push to higher `R` only on the curated-cooldown stage (~50–100B tokens at 8–30 epochs). That is the Muennighoff formula being applied consciously with quality awareness layered on top.

---

## Connections

- Companion paper on quality as an axis: [[excerpts/scaling-laws-data-quality]]
- Companion paper on retention: [[excerpts/physics-of-lm-3]]
- Frontier recipe using the formula implicitly: [[excerpts/olmo-3-decontamination]]
- Chapter synthesis: [[ch-14]]
