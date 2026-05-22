<!-- chapter: ch-09
     phase: llm-ptq-2022
     title: SmoothQuant + AWQ — Activation-Aware Equivalent Transformations
     sources: [[smoothquant]], [[awq]], [[outlier-channel-splitting]], [[oscar]], [[autoawq]]
-->

# Chapter 9 — SmoothQuant + AWQ: Activation-Aware Equivalent Transformations

> **Core insight.** Activation outliers are confined to a small set of channels and those *same channels* carry small-magnitude weights — so you can introduce a per-channel diagonal `diag(s)` that shrinks the activation channel and grows the weight column, leaves the matmul output algebraically unchanged, and makes both sides simultaneously quantizable. SmoothQuant does this with a closed-form `s_j = max|X_j|^α / max|W_j|^(1−α)` for W8A8; AWQ specialises the same trick to weight-only W4A16 with a single grid-searched scalar α per layer.
>
> **Guideline.** For W8A8 deployment use SmoothQuant with `α = 0.5` (LLaMA: `0.85`) and absorb `diag(s)` into the preceding LayerNorm + next Linear; for W4A16 weight-only use AWQ with `group_size=128`, INT4 asymmetric per-group, and per-layer α grid-searched in `[0, 1]` on ~128 calibration sequences. Both are gradient-free and fuse into the runtime graph with zero overhead.

---

## Why this chapter exists

[[llm-int8]] (ch-07) isolated activation outliers into an FP16 column path. It made INT8 *inference* work but it left two large gaps: (1) the FP16 fallback complicates the kernel; (2) `<=4-bit` weight-only quantization — the regime where weight-only inference becomes memory-bandwidth-bound rather than compute-bound — was not addressed. [[gptq]] (ch-08) closed the second gap from the *weight-error* side via inverse-Hessian propagation, but it does nothing about activation outliers, so it cannot push activations below A16.

SmoothQuant (Xiao et al. 2022) and AWQ (Lin et al. 2023) attack the same problem from a third angle: equivalent transformation. The matmul `Y = X · W` is invariant under any invertible per-channel diagonal rescale `(X · diag(s)⁻¹) · (diag(s) · W)`. If activation channel `j` has a 100× outlier and weight column `j` is small, you can split the difficulty — divide the activation column by `s_j` and multiply the weight column by `s_j`. Both quantize cleanly. The scaling is folded into the preceding LayerNorm at offline time, so inference is unchanged.

This chapter covers both methods in one because they are the same equivalent-transformation principle applied at different precision points:

- **SmoothQuant** → W8A8 (activations to INT8). Closed-form scale, no search.
- **AWQ** → W4A16 (weights to INT4, activations stay FP). One α per layer, 20-point grid search.

The ancestor ideas live in [[outlier-channel-splitting]] (2019) and [[oscar]] (NeurIPS 2022, BERT-era). The production realisation is [[autoawq]] + [[marlin-kernel]], the canonical W4A16 deployment stack as of 2024.

---

## 1. The equivalent-transformation identity

For a linear layer `Y = X · W` with `X ∈ R^{T × C_in}` and `W ∈ R^{C_in × C_out}`, introduce a diagonal per-input-channel scale `s ∈ R^{C_in}`:

```
Y = X · W
  = (X · diag(s)⁻¹) · (diag(s) · W)
  = X̂ · Ŵ
```

The product is identical. What changed is the per-channel statistics:

- `max|X̂_{·, j}| = max|X_{·, j}| / s_j` — activation column j shrinks by `s_j`.
- `max|Ŵ_{j, ·}| = s_j · max|W_{j, ·}|` — weight row j grows by `s_j`.

If you pick `s_j` to be large for the activation-outlier channels (and those channels happen to have small weights, which empirically they do), you rebalance the per-tensor / per-channel max so neither side has a dominant outlier and both quantize cleanly with standard absmax INT-k.

**The fusion trick that makes this free.** In a transformer block, every Linear is preceded by a LayerNorm (or RMSNorm). The LayerNorm has a per-channel affine `γ_j, β_j`. Replace them with:

```
γ_j ← γ_j / s_j
β_j ← β_j / s_j
W_j ← s_j · W_j
```

Now `X̂ = LN(x; γ/s, β/s)` and `Ŵ = diag(s) · W`. The inference graph is unchanged; `s` exists only at calibration time.

---

## 2. SmoothQuant — closed-form W8A8

[[smoothquant]] proposes a simple closed form for `s`:

```math
s_j = \frac{\max(|X_{\cdot, j}|)^{\alpha}}{\max(|W_{j, \cdot}|)^{1-\alpha}}
```

The hyperparameter `α ∈ [0, 1]` is the **migration strength**: how much of the difficulty moves from X into W.

- `α = 0` → `s = 1/max|W|` — all difficulty stays in activations (no smoothing).
- `α = 1` → `s = max|X|` — entire activation max moved into weights.
- `α = 0.5` (default) → balanced; `max|X̂| = max|Ŵ| = (max|X| · max|W|)^{0.5}`.

After transformation the per-channel maxes of X̂ and Ŵ are equal, which is the right condition for symmetric INT8 absmax quant on both sides.

**Per-model α from the paper.** Different families have different outlier severity:

| Family | α | Notes |
|---|---|---|
| OPT, BLOOM | 0.5 | balanced default |
| LLaMA | 0.85 | stronger outliers, push more into W |
| GLM | 0.75 | |
| Falcon | 0.6 | |

**Calibration.** A single forward pass over ~512 sequences × 512 tokens (Pile / C4) collects `max|X_j|` per channel. No optimization, no gradients, no per-layer search — just one forward and a closed-form `s`.

**Final inference path.** After the smooth fusion, standard W8A8 with:
- per-token activation absmax INT8 (one scale per token row)
- per-channel weight absmax INT8 (one scale per output channel)
- symmetric, dispatched to INT8 tensor cores

**Results.** SmoothQuant is the first method to make W8A8 viable at 175B+ scale. On OPT-175B: ≤0.1 PPL loss vs FP16 with 1.51× speedup. Single-node serving of OPT-175B and BLOOM-176B becomes possible because INT8 halves the activation memory in addition to halving weights.

> **Pitfall.** SmoothQuant assumes the LayerNorm-then-Linear pattern. Where the activation is post-GeLU/SiLU (no LN directly before), the `diag(s)⁻¹` cannot fuse — you need an extra scale-multiply at runtime. Plan around the architecture, not against it.

---

## 3. AWQ — activation-aware weight-only W4A16

[[awq]] takes the equivalent-transformation idea and specialises it to the weight-only regime. The key empirical observation, from AWQ Figure 2:

**Only ~1% of weight channels are "salient" — those that multiply with the largest-magnitude activation channels.** If you protect those 1% via per-channel scaling, INT4 weight-only quantization is nearly lossless. The other 99% can be quantized aggressively with group-wise RTN.

The transformation is the same as SmoothQuant, but `(diag(s) · X)` is *not* quantized (we're in weight-only A16). Only `W · diag(s)⁻¹` is quantized to INT4: dividing the salient weight columns by `s_j > 1` reduces their dynamic range so RTN preserves them.

### Per-channel scale from activation magnitude

For input channel j, AWQ parameterises:

```math
s_j = \big(\text{mean}(|X_{\cdot, j}|)\big)^{\alpha}, \qquad \alpha \in [0, 1]
```

Note this differs from SmoothQuant: AWQ uses activation **mean** (not max), and there is no weight term — only a single scalar `α` per *layer* to grid-search.

### The grid search (no backprop)

For each layer, sweep α on 20 points in `[0, 1]`:

```
for α in linspace(0, 1, 20):
    s = mean(|X_j|) ** α                          # [C_in]
    W_scaled = W * s                              # equivalent xform
    W_q = quantize_group(W_scaled, G=128, bits=4) # RTN
    loss = ||W · X − dequant(W_q) · (X / s)||²
α* = argmin loss
```

One forward pass per α; no gradients. Total cost: 20 × (one layer forward) per layer. Whole model in minutes on a single A100.

The bowl shape of this curve is concave (single minimum), so the 20-point grid is reliable.

### Salient channels — the empirical justification

AWQ's Figure 2 shows two ablations:

- Keep top-1% of channels by **activation magnitude** in FP16: recovers near-FP16 PPL.
- Keep a random 1% of channels in FP16: substantial PPL loss.

This isolates activation magnitude (not weight magnitude, not random importance) as the right saliency signal. AWQ's scaling protects exactly these channels by giving them larger dynamic range in the INT4 grid.

### Why AWQ beats GPTQ on out-of-distribution data

GPTQ minimises `||W X − Ŵ X||²` against a calibration X — the rounding is **overfit to that calibration's covariance**. AWQ only uses calibration to estimate `mean(|X_j|)` and pick one scalar α per layer. The per-channel scale is data-cheap and shifts gracefully across domains.

Empirical evidence ([[awq]] Table 8): on instruction-tuned LLMs and vision-language models (LLaVA), GPTQ regresses 1–3% on tasks unseen in its calibration set; AWQ holds steady. This is the practical reason AWQ became the W4A16 default for general-purpose deployment.

> **Pitfall.** AWQ's `mean(|X_j|)` is computed from a small calibration corpus. If the deployment distribution differs catastrophically (e.g. you calibrate on English and deploy on code), recompute α — the bowl shape moves.

---

## 4. Pre-LLM ancestors: OCS and Outlier Suppression

The equivalent-transformation idea is older than LLMs.

**[[outlier-channel-splitting]] (OCS, Zhao et al. 2019)** addresses the same outlier-channel problem in CNNs by *duplicating* outlier input channels and *halving* the weights:

```
y = ... + W_{·,c} · x_c + ...
  = ... + (W_{·,c}/2) · x_c + (W_{·,c}/2) · x_c + ...
```

Exact in FP, but the per-column max is now half. This is a *discrete* (split or don't) version of what SmoothQuant generalises to a *continuous* per-channel scale. The conceptual line is: OCS (discrete) → SmoothQuant (continuous diag scale) → OmniQuant (learned diag) → AffineQuant (learned full affine) → FlatQuant (learned Kronecker affine).

**[[oscar]] (Outlier Suppression, Wei et al. NeurIPS 2022)** identifies LayerNorm's per-channel `γ` as the *structural source* of activation outliers in BERT and proposes Gamma Migration: fold `γ` into the next Linear (`W ← W · diag(γ)`, `γ ← 1`). This eliminates the LN-injected per-channel dilation. SmoothQuant generalises Gamma Migration: instead of migrating exactly `γ`, migrate an arbitrary learned per-channel scale `s` chosen to minimise the joint X+W quant difficulty.

The lesson: outlier handling has a long lineage of "move the difficulty around without changing the function." SmoothQuant and AWQ are the LLM-era successors.

---

## 5. Comparison: W8A8 (SmoothQuant) vs W4A16 (AWQ)

| Aspect | SmoothQuant (W8A8) | AWQ (W4A16) |
|---|---|---|
| Target | activation + weight INT8 | weight INT4, activation FP16 |
| Scale form | `s_j = max|X|^α / max|W|^{1−α}` | `s_j = mean(|X|)^α` |
| α | closed-form (per-family default) | grid-searched per layer |
| Calibration | 512 × 512 tokens, one pass | 128 × 512 tokens, 20 passes |
| Group size | per-channel (whole row) | 128 |
| Bits/weight | 8 + ~0 | 4 + 16/128 ≈ 4.125 |
| Inference kernel | INT8 GEMM (e.g. CUTLASS) | INT4 dequant + FP16 GEMM (Marlin/AWQ) |
| Memory savings | 2× weights, 2× activations | 4× weights, 1× activations |
| Speedup vs FP16 (175B) | ~1.5× | ~3× (decode, batch=1) |
| Where it wins | high-throughput serving | memory-bandwidth-bound decode |

**Choose SmoothQuant** when activations are the memory bottleneck (long-context prefill, large-batch serving). **Choose AWQ** when weight memory + decode bandwidth dominate (single-batch chatbot, edge deployment).

These are also composable: SmoothQuant for the activation side + GPTQ or AWQ for the weight side gives W4A8, the recipe behind QServe and Atom (see [[ch-14]]).

---

## 6. The production stack: AutoAWQ

[[autoawq]] is the de-facto open implementation. Three knobs matter:

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
quant_config = {
    "w_bit": 4,
    "q_group_size": 128,
    "version": "GEMM",         # GEMM=prefill, GEMV=batch=1, Marlin=Ampere+Hopper
    "zero_point": True,        # asymmetric INT4
}
model.quantize(tokenizer, quant_config=quant_config, n_samples=128)
model.save_quantized("llama-3-8b-awq")
```

**Kernel choice (the `version` arg).**
- `GEMM` — fused dequant + INT4 GEMM via `mma.sync` on A100/H100; best for prefill / batch>1.
- `GEMV` — dequant into shared memory + FP16 vector-matrix; best for batch=1 decode.
- `Marlin` — newer backend, same kernel family as GPTQ Marlin; faster prefill on Ampere/Hopper.

**The absorption trick** (matches §1): AutoAWQ folds the per-channel scale `s` into the preceding LayerNorm's affine (`γ_new = γ / s`), so the runtime graph has zero extra ops — the scale exists only in the quantized checkpoint.

---

## 7. Perplexity numbers — the calibration

From the AWQ paper Tables 4–6 and AutoAWQ's reference benchmarks (LLaMA-2-7B, WikiText-2 PPL, lower is better):

| Method | Bits | LLaMA-2-7B | LLaMA-2-13B | LLaMA-2-70B |
|---|---|---|---|---|
| FP16 | 16 | 5.47 | 4.88 | 3.32 |
| GPTQ (g=128) | 4 | 5.69 | 4.98 | 3.42 |
| AWQ (g=128) | 4 | **5.60** | 4.97 | **3.41** |
| RTN | 4 | 6.66 | 5.51 | 3.67 |

SmoothQuant W8A8 (from [[smoothquant]] Table 4, OPT-175B / WikiText-2):

| Method | OPT-175B PPL | Δ vs FP16 |
|---|---|---|
| FP16 baseline | 8.34 | — |
| LLM.int8() | 8.40 | +0.06 |
| ZeroQuant | 8.61 | +0.27 |
| SmoothQuant (α=0.5) | **8.35** | +0.01 |

The numbers to internalise: AWQ + GPTQ both sit within ~0.1–0.2 PPL of FP16 at W4 (when used correctly with `group_size=128`). SmoothQuant brings W8A8 within ~0.01 PPL on OPT-175B — effectively lossless.

---

## Connections and what's next

- **[[smoothquant]]** — full extract; closed-form W8A8 transformation.
- **[[awq]]** — full extract; activation-aware W4A16 grid search.
- **[[outlier-channel-splitting]]** — pre-LLM ancestor; discrete duplication instead of continuous scaling.
- **[[oscar]]** — Gamma Migration in BERT; structural predecessor of SmoothQuant.
- **[[autoawq]]** — production implementation; GEMM/GEMV/Marlin kernel selection.
- **[[ch-10]] / [[omniquant]]** — replaces the closed-form / grid-search with a learnable per-channel scale + shift trained via block-wise MSE.
- **[[ch-14]] / [[quarot]]** — generalises diag-scaling to full orthogonal rotation; eliminates outliers in a rotated basis rather than rebalancing them.
- **[[ch-19]] / [[marlin-kernel]]** — the W4A16 GEMM that turns AWQ checkpoints into 3× FP16 throughput.

## Further reading

- AWQ won MLSys 2024 Best Paper Award. Read the paper end-to-end; the salient-channel discovery (§3.1) is the kernel of the activation-aware era.
- SmoothQuant's Figure 2 (per-channel activation magnitude heatmap before/after smoothing) is the single most useful visualisation in this entire chapter sequence.
