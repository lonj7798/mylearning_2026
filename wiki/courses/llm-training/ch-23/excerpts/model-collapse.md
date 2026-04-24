---
chapter: ch-23
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/model-collapse.md
source_url: https://www.nature.com/articles/s41586-024-07566-y
created_at: "2026-04-23"
---

# Excerpt: Shumailov et al. (Nature 2024) — the Gaussian-mixture collapse proof and the OPT-125M ladder

**Source library:** `wiki/raw-data/llm-training/papers/model-collapse.md`
**Authors:** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, Yarin Gal
**Venue:** Nature 631, 755–759 (2024); arXiv 2305.17493

---

## Why this source anchors ch-23

Ch-23's §1 is a compressed version of this paper's §2–§3. The three-error decomposition, the tail-variance formula, and the OPT-125M generation table all lift directly from the source. The chapter's rhetorical spine — "mean loss is a collapsing quantity" — is this paper's most-cited observation, repackaged as the operational takeaway.

---

## The three error sources — direct quote frame

```
# model-collapse.md, lines 17-22 + "Abstract"
They isolate three error sources (statistical sampling error,
functional expressivity error, functional approximation error)
that compound across generations.
```

Each source is a distinct failure mode with a distinct remedy:

- **Statistical sampling error** — finite-sample Monte Carlo loses rare mass in expectation. Remedy: larger `N`. But the paper shows sampling error does not vanish with model scale, it vanishes with sample scale, and `N` in the iterated loop is bounded by what the *generator* emits per generation.
- **Functional expressivity error** — the model class cannot represent every distribution. Remedy: richer class. But expressivity error is the *floor*; once you've saturated it, only the other two sources remain, and they still compound.
- **Functional approximation error** — optimization is inexact. Remedy: better training. But each generation's optimization error is stochastic; its tails add across generations.

The practical consequence is that you cannot fix iterated collapse by making the model bigger, better-optimized, or better-parameterized. You can only fix it by breaking the recursion — with fresh real data or an external verifier (§3 and §4 of ch-23, respectively).

---

## The Gaussian-mixture proof — why tails die first, permanently

The paper's analytical setup (simplified): real distribution `p_0 = Σ w_k N(μ_k, σ²)`. Sample `N` from `p_0`; refit; iterate. The refitted mixture weights `ŵ_k^{(n)}` follow (approximately)

```
Var[ŵ_k^{(n+1)}] ≈ Var[ŵ_k^{(n)}] + w_k(1 - w_k) / N
⇒ Var[ŵ_k^{(n)}] ≈ n · w_k(1 - w_k) / N
```

This is the critical formula. **Variance in the estimated weight grows linearly with generation count `n`.** Now the irrecoverability argument: once `ŵ_k^{(n)}` hits zero in any generation, subsequent generations sample zero points from component `k`, so its weight stays at zero forever. Low-weight components (rare modes) have the smallest safety margin — they hit zero first.

Quantitatively, a component with `w_k = 0.03` (the rarest mode in ch-23's HTML simulation) and sample size `N = 300` has variance `n · 0.03 · 0.97 / 300 ≈ n/10300` per generation. The probability of hitting zero by generation 10 is non-trivial; by generation 20 it is dominant. The simulation in the companion HTML reproduces this — the rightmost mode (w=0.03) is the first to vanish under pure recursion.

---

## The `k`-th-moment formula — ch-23's central equation

```
# model-collapse.md, lines 30-32
Var[μ_k^{(n)}] ≈ n · σ² / N + O(model error)
```

This is the quote-line. The variance of the `k`-th moment of the empirical distribution **accumulates linearly in `n`**. Tails are erased first because their *sampled* mass vanishes fastest — rare events have the smallest contribution to `N_k` and therefore the largest relative variance.

Two implications the paper underlines:

1. **Scaling `N` (per-generation sample size) postpones but does not prevent collapse.** Doubling `N` halves per-generation variance growth but still leaves variance growing linearly; you buy generations, not immunity.
2. **Scaling `n` (number of generations) amplifies collapse linearly.** Long pipelines — years of iterated post-training, successive RM-filtered generations — accumulate more collapse than short ones.

The only way to change the linear-in-`n` behavior is to add fresh real-data draws (which inject new variance *toward* truth, not away from it) or to reject off-truth samples before refit (the verifier).

---

## The OPT-125M experiment — what Shumailov actually ran

Concrete setup from the source:

- Base model: **OPT-125M**, pretrained weights from Meta.
- Real data: **wikitext2**.
- Generation: fine-tune on current data → temperature-1.0 sample 100k tokens → replace training data with samples (or mix with fraction of real) → re-fine-tune.
- Iterations: up to **10 generations**.
- Variants: (i) pure replacement `p_real = 0`, (ii) persistent 10% real, (iii) perplexity-filter ablation (which fails by the expressivity/approx sources even though it helps sampling).

The paper's progression numbers (ch-23 §1 lifts these):

| Generation | Average PPL | Rare-token PPL | Qualitative |
|---|---|---|---|
| 0 | 34.1 | 412 | coherent |
| 5 | 32.9 | 1,104 | topic compression |
| 9 | 31.4 | >10⁴ | incoherent |

**Average PPL drops monotonically** while rare-token PPL explodes. This is the single most important visual evidence in the paper: the collapse is *invisible* in the standard eval metric. A practitioner watching mean PPL would say "training is converging." They would be watching a model implode.

---

## Why it holds across architectures — the statistical mechanism

The source is explicit:

```
# model-collapse.md, lines 42-44 + Abstract
Holds for Gaussian mixtures, VAEs, and LMs — the mechanism
is statistical, not architecture-specific.
```

The paper runs three model-class experiments:

- **Gaussian mixtures** — closed-form; the proof's exact setting.
- **VAEs** on MNIST — image distributions; early digits degenerate, then later ones; by generation 10, most samples are blurry mode-collapsed blobs.
- **LLMs (OPT-125M)** — the table above.

The mechanism is **statistical sampling error compounding**, which is present in any finite-sample iterated loop. No architecture immunizes against it. This is why "use a better architecture" is not in the paper's list of remedies; only "inject external signal" is.

---

## The accumulation-vs-replacement distinction

```
# model-collapse.md, lines 45-48
Not a prediction that all synthetic data is bad — the pure-replacement
regime the paper studies is stricter than most real pipelines.
Accumulation + filtering avoids the worst case.
```

This is the hopeful corollary the paper itself points to. **Accumulation** — where each generation's training set is `real ∪ synthetic` with `real` never rotated out — bounds the error. The proof: the real-data contribution to the mixture estimator has variance that *does not* grow with `n` (it's re-drawn from the true distribution every generation); the synthetic contribution's variance does grow with `n`, but it's weighted down by the fixed real fraction.

Concretely, if `real_frac = 0.10` persistently, the variance at generation `n` is bounded by roughly `n · σ² / (N · 10) + σ² / (N · 0.1) ≈ O(1/N)` instead of `O(n/N)`. Collapse is turned from a linear-in-`n` divergence into a constant error floor. This is why ch-23 §3's mitigable regimes all share one property: a persistent real-data anchor or an external-signal gate.

---

## What the 2025 literature adds beyond this paper

Source file §"2025 follow-ups (load-bearing)":

- **[[strong-model-collapse]]** (Dohmatob 2024/2025): scaling-law-level statement; 1% contamination breaks scaling.
- **Gerstgrasser 2024:** explicit proof that accumulation bounds error.
- **Zhu 2025:** token-level re-sampling as a mid-step mitigator.
- **He 2025 / Garg 2025:** optimal real:synth mixing ratios.
- **Zhang 2025 (arxiv 2510.16657):** external verification → analytical convergence guarantee.

Ch-23 §3 structures all of these as responses to the core Shumailov result: accumulation and verification are not orthogonal fixes, they are two sides of the same "inject external signal" principle.

---

## Connections

- [[excerpts/strong-model-collapse]] — the scaling-law extension; 1% contamination flatlines the scaling benefit.
- [[excerpts/faithful-synth-eval]] — the verification-as-defense line Zhang et al. 2025 anchors.
- [[excerpts/prismatic-synthesis]] — G-Vendi as upstream coverage audit; gradient-targeted synthesis as an anti-collapse mechanism.
- [[excerpts/synthetic-data-scaling-laws]] — empirical evidence that rephrased synthetic evades the worst of this paper's predictions while pure-generated textbook synthetic does not.
- [[excerpts/nemotron-4-synthetic]] — production evidence that accumulation + RM-as-judge keeps a 98%-synthetic pipeline from collapsing.
- [[excerpts/apigen]] — rule-based verifier gate, 3-layer; the strongest practical anti-collapse signal for code / tool modalities.
- [[ch-23]] — §1 and §3 draw directly from this source.
