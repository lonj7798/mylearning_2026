---
chapter: ch-01
course: llm-inference
phase: read
excerpt_of: "The Curious Case of Neural Text Degeneration (Holtzman et al. 2019) + HF Generation Strategies docs"
source_url: https://arxiv.org/abs/1904.09751
created_at: "2026-05-21"
---

# Excerpt: Sampling strategies — temperature, top-k, top-p, min-p

**Authors:** Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi (nucleus sampling, 2019); HF Transformers team (`GenerationConfig` docs)
**Year:** 2019 (Holtzman) / continuous (HF docs)
**Venue:** ICLR 2020 (Holtzman) ; HF documentation
**URLs:** https://arxiv.org/abs/1904.09751 ; https://huggingface.co/docs/transformers/generation_strategies
**Raw-data sources:** [[raw-data/neural-text-degeneration]], [[raw-data/hf-generation-strategies]]

---

## The four canonical knobs

Sampling acts on the `[vocab]` logit vector at each decode step. Modern stacks apply transformations in a fixed order:

```
1. Penalties     z_i ← z_i − α·count_in_context(i) − β·1[i appeared]
2. Temperature   z_i ← z_i / T                              (T ∈ (0, ∞))
3. Top-k mask    keep argmax_k(z), set rest to −∞           (k ∈ ℤ≥0)
4. Softmax       p_i = exp(z_i) / Σ exp(z_j)
5. Top-p mask    keep smallest V s.t. Σ_{i∈V} p_i ≥ p; renormalize  (p ∈ (0, 1])
6. Min-p mask    drop p_i < min_p · max(p); renormalize    (min_p ∈ [0, 1])
7. Sample        next_id ~ Multinomial(p)
```

`T = 0` (greedy) bypasses everything: return `argmax(z)` directly. Any combination of (3, 5, 6) can be active simultaneously.

---

## Holtzman's nucleus sampling definition

The smallest subset `V_p ⊆ V` such that:

```math
\sum_{x \in V_p} P(x \mid x_{1:t-1}) \geq p
```

Sampling renormalizes `P` over `V_p` and draws one token. Concretely:

```python
sorted_probs, sorted_idx = torch.sort(probs, descending=True)
cdf = torch.cumsum(sorted_probs, dim=-1)
nucleus_mask = cdf <= top_p
nucleus_mask[..., 0] = True                                 # always keep argmax
nucleus_probs = sorted_probs * nucleus_mask
nucleus_probs = nucleus_probs / nucleus_probs.sum()
sampled_pos = torch.multinomial(nucleus_probs, num_samples=1)
return sorted_idx[sampled_pos]
```

The behavioral property that fixed-k can't match: at a low-entropy step (e.g. after "The capital of France is "), `V_p` collapses to ~1 token; at a high-entropy step (e.g. after "She thought about "), `V_p` opens up to dozens. Top-k is rigid at every step; top-p is adaptive.

---

## Why beam search fails for chat

[[raw-data/neural-text-degeneration]] diagnosed the failure empirically. Beam search maximizes cumulative `Σ log P(token_i | context)`. For open-ended text the *globally* highest-likelihood continuation is repetitive: once "the" appears, `P("the" | "...the") > 0.5` keeps "the" as the top expansion, so the beam locks into "the the the the the". Holtzman's Figure 1 has the canonical examples — "I don't know. I don't know. I don't know. ..." with B=10.

The deeper reason: human text *does not* track maximum-per-step likelihood. The empirical distribution of human continuations has fat tails and high entropy. Beam search aggressively suppresses that variance. Sampling preserves it; nucleus sampling preserves it while truncating the unreliable far tail.

**Where beam still works.** Tasks where likelihood and quality align: translation (one correct meaning per source sentence), code completion under a known target (autoregressive decoding under a CFG), formal summarization. Modern hosted chat APIs (OpenAI, Anthropic) don't expose beam search at all.

---

## Min-p: the 2024 newcomer

Min-p ([[raw-data/hf-generation-strategies]] addition) thresholds probabilities relative to the maximum:

```python
threshold = min_p * probs.max()
probs = torch.where(probs < threshold, 0.0, probs)
probs = probs / probs.sum()
```

Compared to top-p, min-p is robust to vocab-size and temperature changes — `min_p = 0.1` means "any token at least 10% as likely as the top token". A common defaults: `top_p=1.0` (off) + `min_p=0.1` + `temperature=1.0`.

---

## Hugging Face `GenerationConfig` mapping

| Strategy | `do_sample` | `num_beams` | Other |
|---|---|---|---|
| Greedy | False | 1 | — |
| Beam | False | >1 | `early_stopping=True` |
| Multinomial sampling | True | 1 | `temperature=1.0` |
| Top-k sampling | True | 1 | `top_k=50` |
| Nucleus | True | 1 | `top_p=0.95` |
| Beam + sampling | True | >1 | `top_p`, `temperature` |
| Diverse beam | False | >1 | `num_beam_groups>1`, `diversity_penalty` |
| Contrastive | False | 1 | `penalty_alpha`, `top_k` |

The set of defaults that frontier-lab APIs use as of 2026: `temperature=0.7` to `1.0`, `top_p=0.9` to `0.95`, presence/frequency penalties at `0`. Code-model defaults: `temperature=0.2`, `top_p=0.95`.

---

## Common pitfalls

- **Applying `top_k` after `top_p`**: order-of-operations bug. The pipeline order is `temperature → top-k → softmax → top-p → min-p`. Reversing causes silent quality loss because you renormalize twice.
- **`temperature=0` with `top_p<1`**: meaningless — greedy bypasses the distribution. Most stacks short-circuit to `argmax` when `T=0`.
- **Beam search for chat**: produces "I don't know I don't know I don't know" patterns. See Holtzman Figure 1.
- **Forgetting `length_penalty` for beam**: beam scores cumulative log-prob, biasing toward shorter outputs (each token subtracts a negative). `length_penalty=1.0` is neutral; `>1` favors longer sequences.

---

## Connections

- [[excerpts/autoregressive-loop]] — sampling consumes the logits produced by one forward pass; together they form the decode iteration.
- [[excerpts/structured-decoding]] — constrained decoding sets `logits[invalid] = -∞` *before* the sampler runs.
- [[raw-data/beam-search]] — Freitag & Al-Onaizan 2017 on beam mechanics for MT.
- [[ch-14]] — speculative decoding uses the sampler's distribution for its acceptance rule.
