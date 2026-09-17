---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: primary source (no library card at the time of this revision)
source_url: https://lmsys.org/blog/2024-08-28-style-control/
created_at: "2026-09-15"
---

# Excerpt: Does style matter? Disentangling style and substance in Chatbot Arena

**Authors:** Tianle Li, Anastasios Angelopoulos, Wei-Lin Chiang (LMSYS Org)
**Year:** 2024 (blog post, 2024-08-29)
**Source type:** official blog (the organization that runs Chatbot Arena)
**Checked on:** 2026-09-15 against the fetched blog text.

## Why ch-49 uses it

It is the style-control method that Arena-Hard-Auto and later work reuse, and it gives fitted coefficients showing how much of a human preference score is attributable to length and markdown.

## Method

The Arena score is a Bradley–Terry logistic regression of the battle outcome `Y_i ∈ {0,1}` on a model-indicator vector `X_i ∈ R^M`, where `X_{i,m} = 1` if model `m` is shown on the left and `−1` if on the right:

```
β̂ = argmin_{β ∈ R^M}  (1/n) Σ_i BCELoss( sigmoid( X_iᵀ β ), Y_i )
```

Style control adds style features `Z_i ∈ R^S` with their own coefficients:

```
β̂, γ̂ = argmin_{β ∈ R^M, γ ∈ R^S}  (1/n) Σ_i BCELoss( sigmoid( X_iᵀ β + Z_iᵀ γ ), Y_i )
```

`β̂` are the model coefficients, now controlled for style; `γ̂` are the style coefficients. The post notes the objective is also reweighted in practice to handle non-uniform model sampling.

Each style feature is defined as

```
normalize( (feature_A − feature_B) / (feature_A + feature_B) )
```

The division by the sum makes the difference proportional to the pair's magnitudes: a 20-token difference between 500 and 520 tokens is treated differently from the same difference between 20 and 40 tokens. The post contrasts this with the AlpacaEval LC normalization, `tanh((feature_A − feature_B) / σ(feature_A − feature_B))`.

## Features and fitted coefficients

Four features: answer token length, number of markdown headers, number of markdown bold elements, number of markdown lists.

| Control setting | Length | Markdown list | Markdown header | Markdown bold |
|---|---|---|---|---|
| Control both | 0.249 | 0.031 | 0.024 | 0.019 |
| Control markdown only | — | 0.111 | 0.044 | 0.056 |
| Control length only | 0.267 | — | — | — |

The post states that length was the dominant style factor and that all other markdown effects are second order.

## Reported ranking effects

With length and markdown controlled, GPT-4o-mini and Grok-2-mini drop below most frontier models, while Claude 3.5 Sonnet, Claude 3 Opus, and Llama-3.1-405B rise. On the Hard Prompt subset, Claude 3.5 Sonnet ties for first with chatgpt-4o-latest and Llama-3.1-405B moves to third.

## Limits stated by the post

The post describes the feature set as a first version and calls the choice of what counts as a confounder an interpretive decision. Style dimensions not represented by these four features are not removed.

## Connections

[[chatbot-arena]] (the Bradley–Terry fit being extended), [[length-controlled-alpacaeval]] (the same regression idea for an automatic annotator), [[arena-hard-benchbuilder]] (reuses these four features), [[rm-bench]] (uses the same length-and-markdown style definition to build its style-controlled pairs).
