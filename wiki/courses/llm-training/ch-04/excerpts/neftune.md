---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Jain et al. 2023 — "NEFTune: Noisy Embeddings Improve Instruction Finetuning"
source_url: https://arxiv.org/abs/2310.05914
created_at: "2026-04-23"
---

# Excerpt: Jain et al. 2023 — NEFTune

**Source:** `wiki/raw-data/llm-training/papers/neftune.md`
**Paper:** Neel Jain, Ping-yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Aniruddha Saha, Micah Goldblum, Jonas Geiping, Tom Goldstein
**arXiv:** https://arxiv.org/abs/2310.05914

---

## Bibliographic header

NEFTune is the archetypal "free lunch" paper in SFT: one tensor addition, no new forward pass, no new hyperparameter beyond a scalar α. The headline result:

> *"Standard finetuning of LLaMA-2-7B using Alpaca achieves 29.79% on AlpacaEval, which rises to 64.69% using noisy embeddings."*

A 35-point AlpacaEval jump from a three-line code change is the kind of delta that forces every SFT stack to adopt the recipe by default. The deeper interest is *why* it works — it reveals that much of the 2023-era Alpaca-scale SFT loss was dominated by memorisation rather than generalisation.

---

## The core insight quote

From the raw-data notes:

> *"Adding bounded uniform noise to the input-embedding vectors at every SFT step acts as a regularizer that disrupts token-identity overfitting on small instruction sets — one line of code, often ~30 pts of AlpacaEval gain."*

The claim has two parts: a *mechanism* (embedding noise disrupts token-identity overfitting) and an *empirical magnitude* (≈ 30 AlpacaEval points on Alpaca-scale data). The mechanism is non-obvious — noise on embeddings is a weaker perturbation than dropout or weight decay — but the effect size at Alpaca-scale dominates both.

---

## The noise formula — one-line definition

From the raw-data notes:

> *"At each training step, for an input embedding matrix X ∈ R^{L × d}:*
> *`X_tilde = X + ε`*
> *`ε_{i,j} ~ Uniform(−1, 1) · (α / √(L · d))`"*

Formally:

```math
\tilde{X}_{i,j} = X_{i,j} + \epsilon_{i,j}, \qquad \epsilon_{i,j} \sim \mathrm{Uniform}\!\left(-\frac{\alpha}{\sqrt{L \cdot d}},\; \frac{\alpha}{\sqrt{L \cdot d}}\right)
```

where:
- `X ∈ R^{L×d}` is the post-embedding activation for a single sequence (L = sequence length, d = embedding dimension).
- `ε_{i,j}` is sampled IID per element per forward pass.
- `α` is the single tunable hyperparameter.

```python
# conceptual pseudocode, inserted after the embedding lookup
X = embed(input_ids)                         # shape (B, L, d)
if self.training:
    L, d = X.shape[-2], X.shape[-1]
    noise_scale = alpha / math.sqrt(L * d)
    eps = (2 * torch.rand_like(X) - 1) * noise_scale
    X = X + eps
# ... rest of transformer ...
```

Three choices are load-bearing and the paper justifies each:

1. **`/ √(L · d)` normalisation.** The expected L2-norm of `ε` scales as `√(L·d) · α / √(L·d) = α` independent of `L` and `d`. This makes α transferable across model sizes (d) and context lengths (L). Without the normalisation, the same α on a 7B model with d=4096 would have a very different effective magnitude than on a 70B model with d=8192.

2. **Uniform, not Gaussian.** Uniform noise has bounded support `[−α/√(Ld), α/√(Ld)]` — no outlier tails. The paper found Gaussian noise works comparably but with higher variance in outcomes; uniform is more stable.

3. **Only on input embeddings, not positional / token-type embeddings and not on hidden states.** This is discussed next.

---

## Why only on input embeddings?

From the raw-data notes:

> *"Only applied to input embeddings; positional / token-type embeddings left untouched."*

The paper's interpretation: the pathology being targeted is *token-identity overfitting* — the model memorising exactly which token occurs at which position in the Alpaca training set. Embedding-level noise perturbs the *identity* signal of each token (shifting its embedding vector by a small random direction) without disturbing positional structure.

If you added the same noise to positional embeddings, you would blur the position-of-each-token signal — which is information the model *should* use precisely. Empirically, noising positional embeddings hurts; noising input embeddings helps. The asymmetry isolates the regularisation to the dimension where overfitting lives.

**Notice:** this also explains why NEFTune does not help at the pretraining scale. Pretraining data is so large that token-identity memorisation is not the bottleneck; the gradient signal is already diverse. NEFTune's effect size is largest where the dataset is small enough that the model could plausibly memorise it, which is exactly the Alpaca regime.

---

## Hyperparameters table — reproduction of paper defaults

From the raw-data notes:

| Knob | Value |
|------|-------|
| α (LLaMA-2-7B / 13B) | 5 |
| α (OPT-6.7B) | 5 |
| Dataset size | tested 1K–50K |
| LR / optimizer | unchanged from baseline SFT |
| Other SFT hparams | unchanged |

Notice what is *not* in the table: no new warmup schedule, no noise-annealing, no learning-rate adjustment. The instruction is literally "add this one line and change nothing else." The paper's swept range for α is `{5, 10, 15}` and all three values improve over baseline; α=5 is the median and the recommended default.

---

## AlpacaEval magnitudes — Table 1 headline

From the raw-data notes:

> *"Table 1: AlpacaEval 2 win rates on LLaMA-2-7B — 29.79 → 64.69 for plain Alpaca."*

The paper's headline magnitudes across datasets (on LLaMA-2-7B):

| Dataset | Baseline AlpacaEval | With NEFTune | Δ |
|---------|--------------------:|-------------:|--:|
| Alpaca (52K) | 29.79% | 64.69% | +34.9 |
| Evol-Instruct | ~70% | ~80% | +~10 |
| ShareGPT | ~68% | ~76% | +~8 |
| OpenPlatypus | ~62% | ~70% | +~8 |

The ordering is not an accident: Alpaca is the smallest and most self-instruct-y dataset (52K examples generated by text-davinci-003), so it has the most room for token-identity memorisation. Evol-Instruct, ShareGPT, and OpenPlatypus are larger and more diverse, so the baseline is already closer to what NEFTune targets — less room to improve.

**Notice:** the ceiling effect matters. Do not expect +35 AlpacaEval points on a Tülu-3-scale dataset. NEFTune helps most where overfitting is the dominant failure.

---

## The regularisation signature — Figure 4

From the raw-data notes:

> *"Figure 4: Training loss drops slightly (regularizer), eval win rate rises sharply — classic regularization signature."*

This is a classic regulariser fingerprint: training loss moves modestly, evaluation metric moves dramatically. If NEFTune were merely reducing overfitting by decreasing effective capacity, training loss would *rise* (model fits less well) and eval would rise (generalisation improves). Instead, training loss slightly *drops* and eval jumps.

The paper's interpretation is that NEFTune pushes the model toward a *different* solution basin — not a worse fit, but a qualitatively different local minimum. The noise-perturbed gradient effectively averages over a neighbourhood of embedding configurations, yielding a model that is smoother in input-space and consequently better at generalising to novel instructions. This is the "noise-averaging" view of SGD regularisation, applied specifically to the embedding dimension.

---

## Why only AlpacaEval-style benchmarks improve

The paper is careful about *which* benchmarks move. AlpacaEval, MT-Bench, and similar LM-as-judge benchmarks evaluate the model's ability to produce *helpful, coherent, on-style* responses — they are sensitive to response quality, verbosity, and structure. NEFTune-tuned models produce noticeably longer, more structured, more "helpful-feeling" responses.

But on knowledge-based benchmarks (MMLU, HellaSwag, factual QA), NEFTune shows essentially zero delta. The noise does not add knowledge; it regularises generation style. This is consistent with the mechanism: embedding noise disrupts surface-form memorisation but does not inject new factual content. For a factual-knowledge SFT (e.g., medical QA where accuracy matters more than style), NEFTune is neutral at best.

**Notice:** this is why NEFTune is considered complementary to, not a substitute for, other SFT techniques. It improves *generation style*; packing improves *throughput*; loss masking improves *training signal quality*; they compose without interference.

---

## When NEFTune helps vs. when it doesn't

From the raw-data notes:

> *"**Helps:** small/medium instruction datasets (Alpaca, Evol-Instruct, ShareGPT); early-stopped SFT; models that otherwise overfit.*
> ***Less or no gain:** very large SFT sets (Tülu 3 full), already-RLHF'd models with tightly-controlled loss, continued pretraining."*

The empirical taxonomy:

| Regime | NEFTune effect |
|--------|---------------|
| Alpaca-scale SFT (< 100K examples) | Large, ~30 pts AlpacaEval |
| Evol-Instruct / ShareGPT (~100–500K) | Moderate, ~8–10 pts |
| Tülu-3 scale (> 1M diverse examples) | Small, < 2 pts |
| Continued pretraining | Neutral |
| Post-RLHF additional SFT | Small positive |

The ch-04 guideline ("enable NEFTune with α = 5–15 for SFT on instruction data ≤ 100K examples") targets exactly the regime where the paper's effect size is largest.

---

## Cost — why this is free

From the raw-data notes:

> *"Free: one tensor add, no extra forward/backward. Compatible with gradient checkpointing, packing, FSDP."*

The only overhead is:
1. One call to a uniform RNG of the same shape as the embedding tensor — `O(B·L·d)` work, trivial on GPU.
2. One elementwise tensor addition — also `O(B·L·d)`, fused into the embedding output.

There is no additional backward pass: `ε` is a constant with respect to `θ`, so `∂X_tilde/∂θ = ∂X/∂θ`, and the noise vanishes from the gradient computation. Memory overhead is one extra tensor of embedding shape, released after the addition.

Compatible with every other SFT optimisation in this chapter:
- **With packing:** apply per sub-sequence or per pack (the `L` in the normalisation is the sequence length of the computation, pack-length being fine).
- **With FSDP:** embeddings are sharded; noise addition happens on each shard independently after the all-gather.
- **With gradient checkpointing:** noise is re-sampled on the recompute pass (which is actually a mild additional regularisation, not a bug).

See [[excerpts/hf-alignment-handbook]] for the TRL config flag that toggles NEFTune.

---

## Connections

- Packing mechanics (the other throughput-critical SFT technique): [[excerpts/sequence-packing]]
- Loss masking (the companion loss-design choice): [[excerpts/loss-masking-prompt]]
- Reference implementation and where α is configured: [[excerpts/hf-alignment-handbook]]
- Chapter synthesis: [[ch-04]]
