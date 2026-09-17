---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: primary source (no library card at the time of this revision)
source_url: https://arxiv.org/abs/2404.04475
created_at: "2026-09-15"
---

# Excerpt: Length-Controlled AlpacaEval — A Simple Way to Debias Automatic Evaluators

**Authors:** Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori B. Hashimoto (Stanford University; independent)
**Year:** 2024 (arXiv v1 2024-04; v2 2025-03-10)
**Source type:** paper
**Checked on:** 2026-09-15 against the arXiv v2 PDF text.

## Why ch-49 uses it

It is the reference method for removing a named judge bias by regression, and it supplies the measured size of the length bias on a widely used automatic benchmark.

## The regression (§3, Eq. 1)

```
q_{θ,φ,ψ}(y = 1 | z_m, z_b, m, b, x) =
    logistic( θ_m − θ_b  +  φ_{m,b} · tanh( (len(z_m) − len(z_b)) / std(len(z_m) − len(z_b)) )  +  (ψ_m − ψ_b) γ_x )
```

- `z_m`, `z_b` — outputs of the evaluated model `m` and baseline `b` on instruction `x`.
- `θ_m`, `θ_b` — model strength terms. `φ_{m,b}` — the pair's length sensitivity.
- `ψ_m`, `ψ_b`, `γ_x` — model sensitivity to instruction difficulty, and the difficulty of instruction `x`.
- The length difference is standardized to unit variance and passed through `tanh`, stated as modeling diminishing returns of length differences on the log odds.

Properties the paper derives (§3): identity, `q(y = 1 | z_b, z_b, b, b, x) = 0.5`, and symmetry, `q(y | z_m, z_b, b, m, x) = 1 − q(y | z_b, z_m, b, m, x)`. Both hold because `tanh` is odd and the model and instruction terms flip sign under a swap. Any antisymmetric term centered at 0 would preserve them.

## The length-controlled win rate (Eq. 2)

```
winrate_LC(m, b) = 100 · E_x [ logistic( θ_m − θ_b + (ψ_m − ψ_b) γ_x ) ]
```

Training is a standard GLM fit with cross-entropy loss. For `M` models and `N` instructions the GLM has `3M + N` parameters fitted from `M·N` examples; AlpacaEval has `M > 128` and `N = 805`.

## Gameability measurement (§4.1)

Models were prompted with "Answer with as much detail as possible." (verbose) or "Be as concise as possible while still providing all the necessary information to answer the question." (concise).

- Raw AlpacaEval: gpt4_1106_preview fluctuates from 22.9% to 64.3% across the verbosity prompts.
- Length-controlled AlpacaEval: the same model fluctuates from 41.9% to 51.6%.
- Normalized standard deviation across the three verbosity prompts: 25% → 10%.

## Correlation with human preference (§4.2)

Spearman correlation with Chatbot Arena rises from 0.94 to 0.98, computed over the 38 models present on both leaderboards (MT-Bench has 34); correlations are computed on benchmarks that evaluate at least 25 Arena models.

## Table 1 — comparison of length-control methods

| Metric | Chatbot Arena correlation (↑) | Gameability (↓) | Adversarial win-rate gain (↓) |
|---|---|---|---|
| Win rate | 0.94 | 26% | 0.0 |
| Length-controlled (GLM) | 0.98 | 10% | 8.5 |
| Length-normalized | 0.96 | 15% | 3.6 |
| Length-balanced | 0.95 | 15% | 40.8 |

## Limits relevant to ch-49

The method removes the variance explained by the features included in the regression. It is not a defense against manipulation aimed at the annotator itself: [[null-model-cheating-benchmarks]] reports an 86.5% length-controlled win rate for a constant response of about 205 tokens.

## Connections

[[judge-llm-bias]] (the verbosity bias this method controls), [[lmsys-style-control]] (the same idea applied to human votes, with a different feature normalization), [[arena-hard-benchbuilder]] (compares against LC-AlpacaEval and adopts style control), [[null-model-cheating-benchmarks]] (the adversarial limit).
