---
chapter: ch-21
course: model-quantization
phase: read
excerpt_of: "Design notes for the three ablation options in the ch-21 lab"
created_at: "2026-05-21"
---

# Excerpt: Required-Ablation Design Notes

**Sources:** [[raw-data/gptq]], [[raw-data/awq]], [[raw-data/qlora]]

---

## Why the lab requires one ablation

A four-method head-to-head Pareto tells you *which* method to pick. It does not tell you *why*. The required ablation forces you to interact with one method's internal knob — making the difference between "I ran GPTQ" and "I understand what GPTQ's group_size does to it."

Pick one. Don't try all three — you'll dilute the depth that makes the ablation educational.

---

## Option A — GPTQ `group_size`

### What `group_size` does

GPTQ stores INT4 weights with one `(scale, zero_point)` per group of consecutive input dims, per output row. `group_size=128` means 8 groups per 1024-dim row; `group_size=-1` means one scale for the whole row (per-channel).

Smaller groups → tighter local scale → less quantization error per weight, *but* more scale overhead per stored weight:
- `g=128`: 4 + 16/128 ≈ 4.125 effective bits/weight
- `g=64`: 4 + 16/64 ≈ 4.25 effective bits/weight
- `g=32`: 4 + 16/32 ≈ 4.5 effective bits/weight
- `g=256`: 4 + 16/256 ≈ 4.0625 effective bits/weight

### The sweep

Run `quantize/gptq.py` with `group_size ∈ {32, 64, 128, 256}`. Eval PPL + MMLU on each.

Expected shape from [[gptq]] and the [[survey-low-bit-llm-2024]] cross-references:

| group_size | Llama-3-8B PPL gap vs FP16 | MMLU gap | Effective bits |
|-----------|-----------------------------|----------|----------------|
| 256 | ~0.20 | ~0.5 pp | 4.06 |
| 128 | ~0.15 | ~0.3 pp | 4.13 |
| 64 | ~0.13 | ~0.2 pp | 4.25 |
| 32 | ~0.12 | ~0.1 pp | 4.50 |

### The interpretation

The marginal gain from `g=128 → g=64` is ~0.02 PPL. The marginal storage cost is ~0.12 bits/weight (3% more storage). In production, this is almost never worth it — `g=128` is the Pareto sweet spot, which is why every paper uses it.

The marginal gain from `g=256 → g=128` is ~0.05 PPL. The marginal storage cost is ~0.07 bits/weight. This *is* usually worth it; this is why `g=128` is the floor, not `g=256` or `g=-1`.

The lesson the ablation teaches: production defaults exist because the Pareto knee is sharp and well-located. The ablation either confirms the knee for your model (good — you understand the design space) or moves the knee (interesting — investigate why your model differs).

---

## Option B — AWQ `α` grid extension

### What `α` does

AWQ chooses a per-input-channel scale `s_c = (mean|X_c|)^α` and applies the equivalent transformation `Y = (W · diag(s)⁻¹) · (diag(s) · X)`. The weights `W · diag(s)⁻¹` are quantized to INT4; the activation factor `diag(s) · X` is folded back into the preceding LayerNorm.

`α` controls how aggressively the salient (large-activation) channels are protected:
- `α = 0`: `s_c = 1` for all c — no scaling, equivalent to plain RTN.
- `α = 1`: `s_c = mean|X_c|` — full activation-magnitude protection.
- `α ∈ (0, 1)`: smooth interpolation.

AWQ default: grid search 20 points in `[0, 1]`, pick the `α` that minimises layer output MSE.

### The sweep

Force `α ∈ {0.3, 0.5, 0.7}` (override the grid search). Eval PPL + MMLU on each.

Expected shape:

| α | PPL gap | Layer-output MSE | Notes |
|---|---------|------------------|-------|
| 0.0 | ~0.40 | high | RTN baseline; reference |
| 0.3 | ~0.15 | medium | scaling helping but undershoot |
| 0.5 | ~0.10 | min (usually) | sweet spot — bowl minimum |
| 0.7 | ~0.13 | rising | overshoot; bulk channels start to suffer |
| 1.0 | ~0.20 | high | full activation scale; bulk crushed |

### The interpretation

The PPL-vs-α curve is a *concave bowl* with a minimum around 0.5. Plotting it (Figure 5 of the [[awq]] paper) is the most direct way to understand why AWQ's grid search is robust: the function being optimised is smooth and concave, so a coarse grid is enough.

The lesson the ablation teaches: AWQ's gradient-free grid search is not a hack — it's the right tool for the shape of the optimization landscape. Compare this to OmniQuant ([[omniquant]]) which learns the same scale by gradient descent and pays for it in calibration time, fragility to OOD data, and tighter coupling to calibration corpus.

---

## Option C — QLoRA LoRA rank `r`

### What `r` does

QLoRA freezes the NF4-quantized base and trains a LoRA `ΔW = α/r · B A` on top, with `A ∈ R^{r × d_in}` and `B ∈ R^{d_out × r}` in BF16. `r` controls the rank of the adapter — the dimensionality of the subspace in which fine-tuning can move the weights.

`r` trade-offs:
- Smaller `r`: fewer trainable params, faster fine-tune, more memory headroom, but less expressive adapter.
- Larger `r`: more params (linear in `r`), slower fine-tune, more memory, more expressive adapter.

QLoRA's default is `r=64` on all linear layers.

### The sweep

Run QLoRA fine-tune (1 epoch on 5K Alpaca samples) with `r ∈ {4, 8, 16, 32, 64}`. Eval PPL + MMLU + IFEval on each. Report adapter parameter count and wall-clock.

Expected shape:

| r | Trainable params (Llama-3-8B) | Wall-clock | Wikitext PPL | MMLU | IFEval-strict |
|---|------------------------------|------------|---------------|------|---------------|
| 4  | ~17 M (0.2% of base) | 25 min | 6.9 | 65 | 35 |
| 8  | ~34 M (0.4%) | 27 min | 6.8 | 65.5 | 38 |
| 16 | ~68 M (0.8%) | 30 min | 6.75 | 66 | 42 |
| 32 | ~135 M (1.6%) | 35 min | 6.7 | 66.5 | 44 |
| 64 | ~270 M (3.3%) | 45 min | 6.7 | 66.5 | 44 |

(Numbers illustrative; your run will produce its own.)

### The interpretation

PPL improves through `r=16` or `32` and plateaus — Wikitext loss is a smooth task that doesn't need many degrees of freedom.

IFEval and other instruction-following metrics often keep improving through `r=64` because they reward more *behavior changes* than the LM loss tracks.

The lesson the ablation teaches: QLoRA's `r=64` default is conservative — it's the rank where instruction-following gains saturate, not where PPL saturates. If you only care about academic PPL, `r=16` is the Pareto sweet spot. If you care about chat behavior, stick with `r=64`.

A bigger lesson: the right rank depends on the *task*. This is why QLoRA papers report multiple ranks; the "best" rank is workload-specific.

---

## What makes a good ablation memo entry

Bad: "I ran group_size = {32, 64, 128, 256} and got the expected curve."

Good: "I ran group_size = {32, 64, 128, 256}. The PPL Pareto knee is between 128 and 64, marginal gain 0.02 PPL for 0.12 bits/weight overhead. But on IFEval-strict, group_size=128 lost 4 points where group_size=64 lost only 1.5 points — the finer grouping mattered specifically for format-token paths. Recommendation: use group_size=64 for chat models even though the Wikitext Pareto says 128."

The good entry surfaces a *specific* phenomenon and reframes the default recommendation in light of it. That's the gap between "ran the lab" and "learned from the lab."

---

## Connections

- [[ch-21]] §required-ablation — chapter section.
- [[gptq]] / [[awq]] / [[qlora]] — papers behind the three options.
- [[autogptq]] / [[autoawq]] / [[bitsandbytes-nf4]] — implementations being tuned.
- [[ch-20]] §1 — PPL-vs-task-eval asymmetries the good ablation entry exploits.
