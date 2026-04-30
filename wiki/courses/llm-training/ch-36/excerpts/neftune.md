---
chapter: ch-36
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/neftune.md
source_url: https://arxiv.org/abs/2310.05914
created_at: "2026-04-23"
---

# Excerpt: NEFTune — the third ablation axis of ch-36

**Source library:** `wiki/raw-data/llm-training/papers/neftune.md`
**Authors:** Neel Jain, Ping-yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Aniruddha Saha, Micah Goldblum, Jonas Geiping, Tom Goldstein
**Venue:** arXiv 2310.05914 (NeurIPS 2023 track)
**Year:** 2023

---

## Why this source anchors ch-36

NEFTune is the third axis in the 2×2×2 ablation grid (`packed × masked × NEFTune`). Unlike packing (which affects throughput) and masking (which affects correctness), NEFTune is the *optional win* — a one-line augmentation that the paper claims doubles AlpacaEval win rate on small instruction sets, and which every mature SFT recipe now ships as a toggle. Ch-36 uses it as the only "what-if" axis in the grid because:

1. Its effect is empirically large (29.79 → 64.69 on AlpacaEval 2 per the paper) *in one regime* and near-zero in another.
2. The ch-36 mix (100K full, 20K resource-constrained, with 1K-LIMA seed) straddles that regime boundary. This makes NEFTune's delta genuinely uncertain before running, which is what an ablation axis should be per [[karpathy-training-neural-net-recipe]].
3. It composes cleanly with packing and masking, so the 2×2×2 is actually orthogonal — no interaction effects the ablation can't isolate.

---

## The one-line formula and its three decisions

Source lines 30–37 give the whole paper:

> At each training step, for an input embedding matrix `X ∈ R^{L × d}`:
> `X_tilde = X + ε`
> `ε_{i,j} ~ Uniform(−1, 1) · (α / √(L · d))`

In math notation:

```math
\tilde{X}_{i,j} = X_{i,j} + \epsilon_{i,j}, \quad \epsilon_{i,j} \sim \mathcal{U}(-1, 1) \cdot \frac{\alpha}{\sqrt{L \cdot d}}
```

Three decisions are hidden in that equation and each is a potential failure mode ch-36's §6 ablation code must get right:

| Decision | Paper's choice | Why |
|----------|----------------|-----|
| Noise distribution | Uniform(−1, 1) | Bounded tail; Gaussian would occasionally emit large outliers that destabilize gradient norm |
| Scale denominator | `√(L · d)` | Keeps per-element noise energy invariant across model width and context length |
| Where to inject | Token embeddings only | Positional / type / LayerNorm params are untouched — those already have their own regularizers |

**Notice:** the denominator is `√(L · d)`, not `√d`. If you implement NEFTune and forget the `L`, noise magnitude scales with sequence length — a 4096-token packed block gets 4× the per-element perturbation of a 1024-token block, and long-context runs silently over-regularize. Ch-36's resource-constrained path uses `max_seq_length=2048`; the full path uses `2048` as well, so the `L` term is constant-ish within a run, but across ablations with different pack densities it matters.

---

## The empirical claim that makes NEFTune worth an axis

Source lines 14–15:

> Standard finetuning of LLaMA-2-7B using Alpaca achieves 29.79% on AlpacaEval, which rises to 64.69% using noisy embeddings. NEFTune also improves over strong baselines on modern instruction datasets: Evol-Instruct (~10%), ShareGPT (~8%), and OpenPlatypus (~8%).

| Dataset | Size | Baseline AlpacaEval 2 | NEFTune AlpacaEval 2 | Delta |
|---------|------|-----------------------|-----------------------|-------|
| Alpaca | 52K | 29.79 | 64.69 | +34.90 |
| Evol-Instruct | ~70K | (baseline not listed) | +~10 pts | +10 |
| ShareGPT | ~80K | (baseline not listed) | +~8 pts | +8 |
| OpenPlatypus | ~25K | (baseline not listed) | +~8 pts | +8 |

The Alpaca gain is *dramatic* (35 pts) because Alpaca responses are short and stylistically narrow — standard SFT overfits to Alpaca's surface form within an epoch. NEFTune disrupts the overfit. On ShareGPT (longer, more diverse), the gain shrinks to ~8 pts because the base model is already seeing varied response distributions. The ch-36 mix is closer to ShareGPT than Alpaca in diversity (LIMA + synthetic + No-Robots), so the prior estimate for ch-36's NEFTune delta is *~5–10 MT-Bench points*, not 30+.

---

## The regime boundary — the prediction ch-36 must make

Source lines 55–57:

> **Helps:** small/medium instruction datasets (Alpaca, Evol-Instruct, ShareGPT); early-stopped SFT; models that otherwise overfit.
> **Less or no gain:** very large SFT sets (Tülu 3 full), already-RLHF'd models with tightly-controlled loss, continued pretraining.

Ch-36's full-budget path (100K mix on Llama-3.2-3B, 1 epoch) sits roughly at the ShareGPT scale. Prior: NEFTune helps, delta in the +5 to +10 MT-Bench range.

Ch-36's resource-constrained path (20K mix on Llama-3.2-1B, 1 epoch) sits smaller — potentially in the Alpaca-like regime. Prior: NEFTune helps more, delta +8 to +15.

The [[allenai-tulu-sft-recipe]] explicitly tested NEFTune on Tülu 3's 900K+ mix and reported near-zero delta. This is the upper end of the regime map. Ch-36 sits *below* that boundary, so NEFTune should help. If the lab's ablation shows zero or negative NEFTune delta, that's a *surprise* in [[karpathy-training-neural-net-recipe]]'s sense — go back and audit α, the `L` normalization, and whether noise is being applied per-step or per-epoch.

---

## Why uniform and not Gaussian — the paper's quiet decision

Source line 44:

> Uniform (not Gaussian) per-element: keeps tail behavior bounded.

With Uniform(−1, 1), every perturbation is in `[−α/√(Ld), +α/√(Ld)]`. Maximum norm perturbation per token: `α · √(d)/√(L·d) = α/√L`. For `α=5, L=2048`, that's `≈ 0.11` per token embedding — well inside the typical embedding L2 norm of 1–2 for Llama-family models.

With Gaussian(0, σ²), the same variance-matched σ would be `σ = α/√(3·L·d)`, but maximum perturbation is unbounded — occasionally a token gets a 4-sigma hit that pushes it well outside the training data manifold. The paper's Section 3 footnote implies this was tried and found to produce occasional training spikes. Bounded tail > unbounded tail for a regularizer whose purpose is smoothing, not exploration.

Ch-36's §6 NEFTune implementation must use Uniform. A common reimplementation bug is reaching for `torch.randn_like` (Gaussian) because it's the first thing that comes to mind. The variance is similar, but the failure tail is not.

---

## Stacking with packing and masking — the orthogonality claim

Source lines 20 and 60:

> Complementary to every other SFT technique — stacks with packing, masking, RLHF.
> Free: one tensor add, no extra forward/backward. Compatible with gradient checkpointing, packing, FSDP.

The paper's empirical evidence is limited to NEFTune-on-top-of-Alpaca-SFT, so "orthogonal to packing" is more hypothesis than proof. Ch-36's 2×2×2 grid is *the test*:

```
             masked=0, masked=1
packed=0:    [N−, N−−] [N+, N−−]
packed=1:    [N−, N++] [N+, N++]
             NEFTune=0, NEFTune=1  NEFTune=0, NEFTune=1
```

If the NEFTune-on/off delta is invariant across the `packed×masked` cells (within noise), NEFTune is orthogonal as claimed. If the delta shifts — e.g., NEFTune helps more when masking is broken — that's a signal NEFTune is partially *papering over* correctness bugs, and the ablation design pays for itself.

**Notice:** this is why ch-36 insists on the full 2×2×2 on the full-budget path. The interesting answers live in the cross-terms, not the marginals. A 3-row ablation (only-packed vs only-masked vs only-NEFTune, each on/off) would miss the interaction.

---

## The "no new hyperparameters" sell, audited

Source line 22:

> No new hyperparameters except α; robust across α ∈ [5, 15].

The paper sweeps `α ∈ {5, 10, 15}` and reports near-identical gains. [[hf-alignment-handbook]] ships `α=5` as default. Ch-36 uses `α=5` without sweeping — consistent with the one-axis rule. A future lab could sweep α on the packed+masked cell; for ch-36, α is a constant.

The more interesting hyperparameter that is *not* mentioned in the paper is *whether to re-sample noise per micro-batch or per gradient step*. Per-microbatch resampling gives lower variance gradient estimates (each macro-step averages multiple noise draws) but costs nothing. Ch-36's implementation uses per-microbatch — matching HF's default — and the §6 sanity check is to verify that `model.embedding.register_forward_pre_hook(add_neftune_noise)` fires once per micro-batch and not once per macro-step.

---

## Connections

- Reference recipe that ships the toggle: [[excerpts/hf-alignment-handbook]] — `neftune_noise_alpha=5`.
- Baseline thesis on small-data SFT (regime where NEFTune wins biggest): [[excerpts/lima]].
- Label-invariant companion: [[excerpts/loss-masking-prompt]] — NEFTune cannot rescue a broken mask.
- Attention-invariant companion: [[excerpts/sequence-packing]] — NEFTune can stack on packing cleanly.
- Upper-bound empirical: [[allenai-tulu-sft-recipe]] — Tülu 3 found near-zero delta at 900K scale.
- Full-read chapter on regularization: [[ch-31]].
- Lab host: [[ch-36]] — `§6 NEFTune ablation`.
