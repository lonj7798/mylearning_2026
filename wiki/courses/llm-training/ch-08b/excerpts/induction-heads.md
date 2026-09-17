---
chapter: ch-08b
course: llm-training
phase: read
excerpt_of: primary source arXiv:2209.11895v1 (Transformer Circuits Thread, published 2022-03-08; no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2209.11895
created_at: "2026-09-15"
---

# Excerpt: In-context Learning and Induction Heads

**Paper:** Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, et al.; correspondence Chris Olah (Anthropic). Published on the Transformer Circuits Thread 2022-03-08; arXiv v1 2022-09. Source type: paper.

## Claim (Abstract)
- Preliminary and indirect evidence that induction heads might constitute the mechanism for the majority of in-context learning in large transformers; strong, causal evidence for small attention-only models; correlational evidence for larger models with MLPs.

## Definitions (Key Concepts)
- In-context learning measured as decreasing loss at increasing token indices (the Kaplan et al. conception), not as few-shot accuracy on chosen tasks.
- In-context learning score: loss at the 500th token minus loss at the 50th token, averaged over dataset examples (Key Concepts wording); Argument 1 writes "the 50th token loss minus the 500th token loss" and reports positive values. Other index choices "do not change our conclusions".
- Induction head, defined empirically on repeated random token sequences: prefix matching (attends to earlier tokens that were followed by the current and/or recent tokens) and copying (raises the logit of the attended-to token). Completes [A][B] … [A] → [B].
- In small models: a previous-token head plus an induction head in a later layer; the rule applies regardless of what A and B are, not a fixed n-gram table.
- Per-token loss vectors: log-likelihoods on 10,000 random tokens from different sequences, analyzed with PCA across snapshots.

## Argument 1: co-occurrence
- ICL develops abruptly in a window of roughly 2.5 to 5 billion tokens and is then constant; before the window less than 0.15 nats, after it roughly 0.4 nats, constant across many model sizes (except one-layer models).
- In the same window: induction heads form; a bump appears on the training loss curve (the only place where the loss is not convex); per-token-loss PCA trajectories change direction.
- For large models the window is "maybe 1-2% of the way through training".
- Not aligned with a scheduled LR, warmup, or weight-decay change; also observed on a books-only dataset for small models.
- Over 75% of final in-context learning forms in this window.
- Caveats: 15 snapshots for large models; a shared latent cause (for example, learning layer composition) could drive both; constant ICL score after the change does not mean constant mechanisms.

## Argument 2: co-perturbation
- Smeared key: k_j^h = σ(α_h) k_j^h + (1 − σ(α_h)) k_{j−1}^h with trainable α_h per head.
- With it, ICL forms in one-layer models and earlier in two-layer models. Argument 2 says "two-layer and larger models", but Model Details presents smeared-key models only at one-layer and two-layer sizes.

## Argument 3: ablation
- Knocking out induction heads at test time: "almost all the in-context learning in small attention-only models appears to come from these induction heads". No ablations for full-scale models.

## Argument 4: examples in the 40-layer, 13B model
- Literal copying head: layer 21/40, copying score 0.89, prefix matching score 0.75.
- Translation head: layer 7/40, 0.20, 0.85.
- Pattern-matching head: layer 8/40, 0.69, 0.94.

## Models (Model Analysis Table, Model Details)
- 34 decoder-only models, one training run each, four series: small attention-only (1-6 layers), small with MLPs (1-6 layers), full-scale (4 layers / 13M to 40 layers / 13B non-embedding parameters), smeared-key.
- Data: earlier version of the Askell et al. dataset (filtered Common Crawl, internet books, other smaller distributions, about 10% Python code); full-scale models on an improved version; each dataset seen in the same order, no repeated data.
- Small models: context 8192, d_model 768, 12 heads, 10,000 steps (~10B tokens), 200 snapshots every 50 steps; phase change at about 1-3B tokens; LR warmup over the first 1.5e9 tokens; weight decay reduced at step 4750 (~5B tokens).
- Full-scale models: d_model = 128 · n_layer; dense and local attention heads; snapshots at steps 2^5 to 2^17 plus final saves (15 snapshots; 14 for 40L).

## Unexplained Curiosities
- "Seemingly Constant In-Context Learning Score": after the phase change the score is about the same for a two-layer model and the 13B model; large models gain their advantage over small models very early in the context, the majority of the difference in the first ten tokens; varying the index definition (including the final token 8192) changes the score only slightly.
- Derivative of loss with respect to log tokens: small models decrease faster than large before the phase change and slower after it.

## Verification
- Read on 2026-09-15 against arXiv PDF v1 (Abstract, Key Concepts, Arguments 1-4, Model Analysis Table, Model Details, Unexplained Curiosities).
