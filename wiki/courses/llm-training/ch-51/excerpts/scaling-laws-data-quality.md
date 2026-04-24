---
chapter: ch-51
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/scaling-laws-data-quality.md
source_url: https://arxiv.org/abs/2510.03313
created_at: "2026-04-23"
---

# Excerpt: Scaling Laws Revisited — effective sample size as the denominator of every CI

**Source library:** `wiki/raw-data/llm-training/papers/scaling-laws-data-quality.md`
**Artifact:** effective-sample-size framing; corpora with identical token count sit on different scaling curves.

---

## Why this source is ch-51's "units matter" anchor

The scaling-laws-data-quality paper is a pretraining paper, but its core move — treat data quality as an explicit scaling variable, measure *effective* sample size rather than nominal token count — is the same move ch-51 makes for evaluation: treat item count as the denominator of a CI, and treat noisy / correlated / contamination-shadowed items as reducing the *effective* N.

Source §Core Insight:

> Data quality can be treated as an explicit scaling variable, not just an anecdotal curation benefit.

The same logic applies in reverse to eval sets. An eval set has a *nominal* item count N and an *effective* item count N_eff that can be much smaller when items are correlated (e.g., the same topic sampled multiple times), contaminated (model has seen test items), or judged by a biased process ([[judge-llm-bias]]).

---

## The guideline ch-51 inherits

Source §Guideline:

> When comparing data pipelines, model effective sample size and noise/deficiency explicitly instead of treating all tokens as equal.

Ch-51 §1 budgets noise from four sources (decode / prompt permutation / dropout / judge) explicitly. The bootstrap CI in §3 uses nominal N; but if you suspect item correlation (e.g., MATH500 has duplicate-topic clusters), you compute a cluster-bootstrap instead — resample at the *cluster* level, which gives the correct N_eff-based halfwidth.

---

## What ch-51 explicitly does NOT borrow

The paper's quality-aware Chinchilla-style scaling law is about pretraining loss, not eval accuracy. Ch-51 keeps the *framing* (N_eff, not N) but uses empirical bootstrap instead of any parametric scaling law — an eval set is too small for a fit, too variable across tasks for a universal law, and too entangled with judge bias for a clean functional form.

---

## Connection to the ch-51 variance table

The variance-source table in §1 bounds *per-run* σ; the CI halfwidth in §3 bounds *item-count* σ. The paper's argument — quality is a scaling variable — motivates ch-51's rule: if N_eff is much smaller than N (due to correlation, contamination, or judge bias), the halfwidth is larger than the naive sqrt(p(1-p)/N) formula predicts, and reporting only the naive CI under-reports the true noise.

---

## Connections

- **[[faithful-synth-eval]]** — averaged metrics hide tail degradation; same reason eval needs per-slice CIs.
- **[[fineweb]], [[dolma]]** — empirical data-quality filters whose effect shows up as shifted scaling curves.
- **ch-50** — per-slice analysis defines the clusters over which to bootstrap.
- **ch-51 §3** — bootstrap derivation; cluster-bootstrap extension uses this paper's N_eff framing.
