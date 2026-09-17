---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/toolformer.md
source_url: https://arxiv.org/abs/2302.04761
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against the arXiv PDF; the earlier version stated candidate counts and survival rates the paper does not report)"
---

# Excerpt: Toolformer — self-supervised API-call annotation with a loss filter

**Source library:** `wiki/raw-data/llm-training/papers/toolformer.md`
**Paper:** Schick, Dwivedi-Yu, Dessì, Raileanu, Lomeli, Zettlemoyer, Cancedda, Scialom, arXiv:2302.04761 (2023-02).

## Setup (§3, §4.1)

GPT-J (6.7B) as the model M; a subset of CCNet as the corpus C; five tools: question answering (Atlas), Wikipedia search (BM25 over KILT), calculator (four operations, two decimals), machine translation (NLLB 600M), calendar (current date).

## Sampling (§2)

`p_i = p_M(<API> | P(x), x_{1:i−1})`; keep positions with `p_i > τ_s`, at most the top k; sample up to m calls per position. Defaults: τ_s = 0.05, k = 5, m = 5. For calculator and machine translation: τ_s = 0.0, k = 20, m = 10, and τ_f = 0.5 (App. A).

## Filtering (§2)

```
L_i(z) = − Σ_{j=i..n} w_{j−i} · log p_M(x_j | z, x_{1:j−1})
L_i^+  = L_i(e(c_i, r_i))
L_i^−  = min( L_i(ε), L_i(e(c_i, ε)) )
keep the call if  L_i^− − L_i^+ ≥ τ_f      (default τ_f = 1.0)
```

`e(c, r)` is the call text with its result, `ε` is the empty sequence. The weights are `w_t = w̃_t / Σ_s w̃_s` with `w̃_t = max(0, 1 − 0.2·t)` (§4.1), so only the five tokens after the call position receive weight.

## Examples kept (Table 2, examples with API calls in C*)

| API | τ_f = 0.5 | τ_f = 1.0 | τ_f = 2.0 |
|---|---|---|---|
| Question Answering | 51,987 | 18,526 | 5,135 |
| Wikipedia Search | 207,241 | 60,974 | 13,944 |
| Calculator | 3,680 | 994 | 138 |
| Calendar | 61,811 | 20,587 | 3,007 |
| Machine Translation | 3,156 | 1,034 | 229 |

The number of candidate calls before filtering is not reported.

## Results

- Math (Table 4): ASDiv / SVAMP / MAWPS — GPT-J 7.5 / 5.2 / 9.9; Toolformer (API calls disabled) 14.8 / 6.3 / 15.0; Toolformer 40.4 / 29.4 / 44.0; GPT-3 (175B) 14.0 / 10.0 / 19.8. The calculator is called for 97.9% of examples.
- Language modeling retention (Table 8): perplexity WikiText / CCNet — GPT-J + CC 10.3 / 10.5; Toolformer (disabled) 10.3 / 10.5.
- Fine-tuning: batch size 128, learning rate 1e-5, linear warmup over the first 10% of training (§4.1).
