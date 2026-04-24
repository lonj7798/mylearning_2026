---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: Shumailov et al. 2024 (Nature) — "AI Models Collapse When Trained on Recursively Generated Data" + Dohmatob et al. 2025 (ICLR Spotlight) — "Strong Model Collapse"
source_url: https://www.nature.com/articles/s41586-024-07566-y ; https://arxiv.org/abs/2410.04840
created_at: "2026-04-23"
---

# Excerpt: Model Collapse — From Curse-of-Recursion to 1% Contamination Flatlining

**Source:** `wiki/raw-data/llm-training/papers/model-collapse.md`, `wiki/raw-data/llm-training/papers/strong-model-collapse.md`
**Primary papers:**
- Ilia Shumailov et al., "AI Models Collapse When Trained on Recursively Generated Data", Nature 631, 755–759 (2024), arXiv:2305.17493
- Elvis Dohmatob, Yunzhen Feng, Arjun Subramonian, Julia Kempe, "Strong Model Collapse", ICLR 2025 Spotlight, arXiv:2410.04840

---

## Bibliographic header

Shumailov et al. (Nature 2024) is the canonical reference that synthetic-data risk crossed from theory to measured failure mode. Dohmatob et al. (ICLR 2025) tightens the bound to the *scaling-law* regime: even a fixed small fraction of synthetic contamination breaks scaling.

From Shumailov (raw-data notes):

> *"Each generation of sampling-then-refitting smooths the tails of the true distribution; iterated, the model's support contracts onto a degenerate near-Gaussian regardless of architecture. Never replace real data with synthetic; always accumulate synthetic on top of a persistent real-data anchor."*

From Dohmatob (raw-data notes):

> *"Within the neural-scaling-laws paradigm, even a fixed small fraction of synthetic contamination (≈1%) in the training pool eliminates the expected test-error reduction from larger data — scaling laws flatline; collapse is not limited to pure-recursive settings."*

These two papers together are why provenance-aware curation became mandatory in 2024–2025.

---

## Shumailov's curse of recursion

The setup: sample from model `p_n`, refit to get `p_{n+1}`, iterate. Three error sources compound:

1. **Statistical sampling error.** Finite-sample estimate of `p_n` has `O(1/√N)` noise; rare-tail events are missed first.
2. **Functional expressivity error.** The model family cannot represent the full distribution; approximation bias accumulates.
3. **Functional approximation error.** The learning algorithm's optimisation noise.

All three compound across generations. The core equation (condensed from the raw-data notes):

```math
\text{Var}[\mu_k^{(n)}] \approx n \cdot \sigma^2 / N + O(\text{model error})
```

Sampling variance grows *linearly in generation count `n`*. Since tails contribute most of the per-event variance (rare events are rarely sampled), they are erased first. The sequence of effects:

- Generations 1–3: distribution looks fine on average metrics; rare-token perplexity starts creeping up.
- Generations 3–5: rare-token PPL spikes visibly, but *average* PPL looks flat or even improves (because average is dominated by head mass, which is sharpening).
- Generations 5–9: model outputs become noticeably degenerate; coherence drops.

**This is the "warning signal hidden in averages" failure mode.** You cannot detect collapse with aggregate loss alone.

---

## The LLM experiment

From the raw-data notes:

> *"LLM experiments: fine-tune OPT-125M on wikitext2, generate text, retrain, iterate; by generation 9 the model emits nonsense while 'new-sample' perplexity appears to improve."*

The striking result: within ~5 generations, the degradation is visible; by ~9 generations, outputs are incoherent. This is a small model on a small corpus, so the mechanism is not architecture-dependent — the same process runs on LSTMs, VAEs, and transformers.

The paper rules out the obvious escape routes:
- Larger models don't prevent collapse (they delay it).
- Larger datasets per generation don't prevent collapse (they reduce per-generation variance but the linear-in-`n` term dominates).
- Temperature tuning doesn't prevent collapse (changes the tail shape but not the statistical mechanism).

**What does prevent collapse:** accumulation, not replacement. Keep a fixed real-data anchor across generations; collapse is bounded.

---

## Dohmatob's tightening — 1% is enough

The 2025 ICLR spotlight sharpens the bound. The key claim: even if you are *not* training recursively, even if you are training on a mix of real and synthetic at fixed proportion `p`, the scaling law flatlines.

From the raw-data notes:

> *"Any non-vanishing synthetic fraction introduces an irreducible error term that prevents test error from decreasing with data size, regardless of model size — scaling-law flattening."*

Core equation (schematic):

```math
\mathbb{E}[R_{\text{test}}] \approx f(N) + c(p) \cdot \sigma_{\text{synth}}^2
```

with `c(p) > 0` for any `p > 0`. The second term is independent of `N` (training set size) — it is an *asymptote*, not a decay term. As `N → ∞`, the test error plateaus at `c(p) · σ²_synth` rather than at the standard floor `f(∞)`.

Empirical reproduction: GPT-2-scale LMs with 1% synthetic injection show measurable scaling-law flattening. The 1% threshold is not a tail effect — it is the new asymptote.

---

## Model size modulates but does not eliminate

A subtle point from the Dohmatob paper:

> *"Below the interpolation threshold larger models amplify collapse; beyond it, larger models partially mitigate but never eliminate it."*

Small models fit the head of the distribution and ignore the tails; collapse affects them via accuracy on the head. Past interpolation, larger models start to recover some tail structure, mitigating (but not preventing) the scaling-law break. This means the "let's just scale up" defense is unavailable — there is no `N` past which contamination becomes irrelevant.

---

## The engineering consequence

Combined picture:

1. **Pure recursion** (generate → train → generate → train → ...): catastrophic. Unfiltered; collapse inside ~5 generations.
2. **Fixed-fraction contamination** (`p ≥ 1%`): scaling law flatlines; you can never out-scale the problem.
3. **Fixed-anchor accumulation** (`p_real ≥ 50%`, never shrinks): bounded error; scaling laws continue.
4. **Verifier-in-the-loop synthetic** (external judge filters synthetic before training): breaks the recursion; behaves like real data in the limit.

The 2025 open-web reality: `p` is already > 1% and climbing. Llama 3, OLMo 3, and DeepSeek V3 all explicitly anchor a persistent real-data slice (books, pre-2022 web) that never shrinks across releases.

---

## What this changes about decontamination

Traditional decontamination (ch-14 §4) filters *eval overlap*. Synthetic-contamination decontamination filters *machine-generated content*. The tools are different:

- **Eval overlap:** n-gram filter (K=8, τ=0.5) against known eval sets.
- **Synthetic detection:** machine-text classifier (similar to AI-detector deployments but at scale); score-distribution shift monitoring; domain-provenance filtering.

Both are required. Neither is sufficient alone. See `ch-14/read.md` §6 for the adversarial-contamination extension.

---

## The open research frontier

From the raw-data notes:

> *"He et al. 2025 / Garg et al. 2025: derive optimal real:synthetic mixing ratios. Escaping Model Collapse via Synthetic Data Verification (Zhang et al. 2025): external verifier breaks the loop; establishes convergence guarantees."*

The 2025 research direction: when synthetic *is* safe. Early findings:
- Fresh real data matters more than total data volume. Accumulation + filtering > replacement.
- External verifiers that are *stronger* than the generator break the recursion.
- Diversity constraints during generation (prompt diversity, decoding temperature, rejection sampling) substantially raise the collapse threshold.

These results underlie the synthetic-data-generation track starting at ch-18 — every 2025 synthetic pipeline is designed around avoiding collapse by construction.

---

## Connections

- Real-anchor practice in frontier pipelines: [[excerpts/llama-3-decontamination]]
- Provenance filtering connection: [[excerpts/olmo-3-decontamination]]
- Adversarial contamination next step: [[excerpts/anthropic-sleeper-agents-data]]
- Chapter synthesis: [[ch-14]]
