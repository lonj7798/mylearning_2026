---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: none (practitioner blog and released library code; no library card planned on 2026-09-15)
source_url: https://www.cs.columbia.edu/~johnhew/vocab-expansion.html
created_at: "2026-09-15"
---

# Excerpt: Initializing New Word Embeddings for Pretrained Language Models (Hewitt, 2021) and the transformers `mean_resizing` default

**Sources:** John Hewitt, blog post dated 06 Dec 2021 (practitioner-evidence: derivation plus a GPT-2 demonstration; no benchmark study). huggingface/transformers v4.46.0, `src/transformers/modeling_utils.py` (released code).
**Status:** read on 2026-09-15.

## Setup and derivation (blog, "Vocabulary Expansion Setup")
- Model: p(w_i | w_<i) = exp(h·e_{w_i}) / Σ_{j=1..n} exp(h·e_j), with h the last hidden state and e_j the output embedding of token j.
- After adding token n+1 with embedding e_{n+1}: p'(w_i) = p(w_i) · 1 / (1 + exp(h·e_{n+1}) / Z), where Z = Σ_j exp(h·e_j).
- KL(p ‖ p') = log(1 + exp(h·e_{n+1}) / Z).
- Zero initialization gives exp(h·0) = 1 and KL = log(1 + 1/Z). "If all h·e_i are large and negative", Z is small and the new token dominates.
- Mean initialization e_{n+1} = (1/n) Σ e_i gives, by Jensen's inequality, exp(h·e_{n+1}) ≤ Z/n and KL ≤ log(1 + 1/n) for any prefix. Optional noise: e_{n+1} ~ N(μ, Σ) with Σ = (E − μ)ᵀ(E − μ)/n; the bound does not hold with noise.

## Demonstration (blog)
GPT-2 small with three added tokens ('Aragorn', 'Frodo', 'Lothlorien') and the then-default initialization generated "Dogs are great because they are  Aragorn Aragorn Aragorn Aragorn Frodo Aragorn". The author states that this "can lead to worse domain adaptation performance – the first gradient steps just remove probability from the new words"; no measurement is given for that statement.

## transformers v4.46.0 (modeling_utils.py L2080-2116)
`def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing: bool = True)`. Docstring: "Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings' mean and covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`. Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models ... Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html"

## How ch-11 uses it
§5 (initializing added tokens), worked example, Negative samples, Common mistakes.
