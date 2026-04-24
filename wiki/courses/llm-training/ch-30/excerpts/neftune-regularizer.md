---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/neftune.md
source_url: https://arxiv.org/abs/2310.05914
created_at: "2026-04-23"
---

# Excerpt: NEFTune — the one-line regulariser with an expiration date

**Source library:** `wiki/raw-data/llm-training/papers/neftune.md`
**Anchor paper:** Jain et al. 2023 — "NEFTune: Noisy Embeddings Improve Instruction Finetuning"
**Counterpart recipe:** [[allenai-tulu-sft-recipe]] (attested "NEFTune off at 939K — neutral")

---

## Why this source anchors ch-30

NEFTune is axis #5 of the five SFT design axes. It is the smallest axis in code (one tensor add) and one of the most dramatic in effect at small scales (+30 pts AlpacaEval on the flagship Alpaca run). It is also the axis most frequently mis-transplanted — practitioners copy it from the 50K Alpaca setting into a 1M-prompt production run where the attested result is that it does nothing. The source's value to ch-30 is that it names both the effect *and* its scale window.

---

## The one-line formula — quoted verbatim

From `neftune.md`, §Noise formula:

> At each training step, for an input embedding matrix `X ∈ R^{L × d}`:
> `X_tilde = X + ε`
> `ε_{i,j} ~ Uniform(−1, 1) · (α / √(L · d))`
> where
> - L = current sequence length (post-packing or pre-packing, per sub-sequence).
> - d = embedding dimension.
> - α = scale hyperparameter (paper's main: α = 5 on LLaMA, sweep {5, 10, 15}).
>
> Noise is resampled each forward pass; no noise at inference.

Ch-30's implementation lifts this verbatim:

```python
def neftune(embeds, alpha=5.0):
    L, d = embeds.shape[-2], embeds.shape[-1]
    eps = (torch.rand_like(embeds) * 2 - 1) * (alpha / (L * d) ** 0.5)
    return embeds + eps
```

## Notice: the `√(L · d)` normalisation is the reason it generalises across scales

From `neftune.md`, §Intuition:

> - √(L · d) normalization: noise magnitude in the embedding subspace is independent of model size / context length.
> - Uniform (not Gaussian) per-element: keeps tail behavior bounded.
> - Only applied to input embeddings; positional / token-type embeddings left untouched.

This is the quiet cleverness. Naive noise injection scales with `d` — a 4096-dim embedding gets four times more perturbation than a 1024-dim one, and you can't compare hyperparameters across model sizes. Dividing by `√(L · d)` makes the RMS of the noise vector independent of `L` and `d`. The same α = 5 works on 7B and 70B without re-tuning.

The uniform-not-Gaussian choice is pragmatic: Gaussian tails can occasionally produce very large perturbations that look like a gradient spike to the optimizer. Uniform is bounded in [−1, 1] so the perturbation is bounded in [−α/√(L·d), α/√(L·d)].

---

## The flagship attested result

From `neftune.md`, abstract:

> Standard finetuning of LLaMA-2-7B using Alpaca achieves 29.79% on AlpacaEval, which rises to 64.69% using noisy embeddings. NEFTune also improves over strong baselines on modern instruction datasets: Evol-Instruct (~10%), ShareGPT (~8%), and OpenPlatypus (~8%).

The Alpaca number (29.79 → 64.69) is the often-cited "+30 pts for free" headline. It is attested, but notice the dataset: Alpaca is 52K rows of Self-Instruct-generated pairs, known to overfit because of its low diversity. NEFTune's gain is largest exactly where overfitting is the dominant failure mode.

On Evol-Instruct / ShareGPT / OpenPlatypus the gains drop to ~8–10 pts — still real, still free, but not the same class of result.

---

## Where NEFTune stops helping — the scale-saturation rule

From `neftune.md`, §When it helps vs doesn't:

> - **Helps:** small/medium instruction datasets (Alpaca, Evol-Instruct, ShareGPT); early-stopped SFT; models that otherwise overfit.
> - **Less or no gain:** very large SFT sets (Tülu 3 full), already-RLHF'd models with tightly-controlled loss, continued pretraining.

And from [[allenai-tulu-sft-recipe]]:

> NEFTune gain saturates — no improvement at 939K; small gain ≤ 100K.

Reading these two together: NEFTune is an anti-overfitting regulariser. When the data is abundant enough that the model cannot memorise it, there is no overfitting to prevent, and the noise becomes dead weight at best. The attested rule in ch-30:

- `|D| ≤ 100K` → on, `α ∈ [5, 15]`.
- `|D| ≥ 500K` → off.
- `100K < |D| < 500K` → sweep `α ∈ {0, 5, 10, 15}` on a 500-prompt probe.

The HTML companion's NEFTune advice logic implements exactly this rule.

---

## Why NEFTune stacks cleanly with every other axis

From `neftune.md`, §Cost:

> Free: one tensor add, no extra forward/backward. Compatible with gradient checkpointing, packing, FSDP.

The compositional structure is important. NEFTune touches only the input embedding matrix, before the first transformer block. Packing operates on the attention masks and position IDs. Loss masking operates on the labels. These three axes never touch the same tensor — they compose by construction. That is why ch-30 can treat them as orthogonal axes of a 5-D design space rather than a set of interacting knobs requiring joint tuning.

---

## The regularisation signature — how to tell it is working

From `neftune.md`, §Key Figures/Tables to Study:

> **Figure 4:** Training loss drops slightly (regularizer), eval win rate rises sharply — classic regularization signature.

If you enable NEFTune and your train loss *decreases* relative to the baseline, something is wrong — the noise should make the per-step loss slightly *higher*. The classic signature is: train loss up ~2–5%, eval win-rate up 5–30 pts. Monitor both curves.

---

## What ch-30 keeps, changes, drops

| Source choice | Ch-30 position | Reason |
|---------------|----------------|--------|
| α = 5 default | Listed as default in HTML, sweep range [5, 15] | Attested across LLaMA-2, LLaMA-3, OPT |
| On for all SFT | Gated by dataset size | Tülu-3 saturation result makes the unconditional recommendation obsolete |
| No inference noise | Same | Attested; noise off at eval |
| Uniform sampling | Same | Bounded-tail argument holds |
| Input-embedding only | Same | Not injected into positional embeddings or hidden states |

---

## Connections

- [[excerpts/loss-masking-regimes]] — NEFTune and the loss mask are on different tensors; composition is trivial.
- [[excerpts/tulu-3-sft-recipe]] — the attested "NEFTune off at 939K" comes from Tülu-3's ablation.
- [[ch-30]] — §5 and the HTML companion's advice logic both depend on this source.
- [[ch-36]] (SFT lab) — NEFTune becomes one of the ablation dimensions.
