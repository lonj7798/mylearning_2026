---
chapter: ch-06
course: model-quantization
phase: read
excerpt_of: "ZeroQ: A Novel Zero Shot Quantization Framework (Cai et al. 2020)"
source_url: https://arxiv.org/abs/2001.00281
arxiv: 2001.00281
created_at: "2026-05-21"
---

# Excerpt: ZeroQ — data-free PTQ via BN-distilled synthetic data

**Authors:** Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami, Michael W. Mahoney, Kurt Keutzer
**Year:** 2020 (CVPR)
**Raw-data source:** [[raw-data/classics/zeroq]]

---

## The BN-distillation loss (the load-bearing formula)

For each conv layer with BatchNorm, optimise a synthetic input batch `x` to match the FP model's BN running statistics:

```math
L_{\text{dist}}(x) \;=\; \sum_\ell \Bigl(\, \bigl\| \mu_\ell(x) - \mu_\ell^{\text{BN}} \bigr\|^2_F
\;+\; \bigl\| \sigma^2_\ell(x) - \sigma^{2,\text{BN}}_\ell \bigr\|^2_F \,\Bigr)
```

`μ_ℓ(x), σ²_ℓ(x)` are the per-channel mean / variance computed on the forward pass of `x` through the FP model up to layer `ℓ`. `μ_ℓ^{BN}, σ²_ℓ^{BN}` are the running statistics recorded during training (free, no extra storage).

---

## The synthesis procedure

```python
def zeroq_synthesize(fp_model, batch_size=32, shape=(3,224,224), steps=500):
    x = torch.randn(batch_size, *shape, requires_grad=True)
    opt = Adam([x], lr=0.01)
    for _ in range(steps):
        run_forward_capturing_per_layer_stats(fp_model, x)
        loss = sum(
            (mu(x_ℓ) - mu_bn_ℓ).pow(2).sum() +
            (var(x_ℓ) - var_bn_ℓ).pow(2).sum()
            for ℓ in fp_model.bn_layers
        )
        loss.backward()
        opt.step()
    return x.detach()
```

No model weights are updated. The optimization shape-matches per-channel statistics — sufficient for activation-scale calibration because PTQ only cares about the *marginal* per-channel distribution, not realistic semantics.

The output images look like noise, not ImageNet, but they suffice for PTQ.

---

## PTQ step (after synthesis)

```text
1. Per-channel weight scale: S_w_c = max|W_c| / Q_max     # purely weight-driven; no data
2. Per-tensor activation scale: 99.9% percentile of |activation| on synthetic batch
3. (Optional) HAWQ mixed-precision bit allocation using synthetic data for Hessian-vector products
```

Total wall clock: ~30 seconds on V100 (synthesis ~10s + PTQ + HAWQ).

---

## Empirical effect (ResNet-50 ImageNet)

| Setting | Top-1 | Notes |
|---|---|---|
| FP32 | 77.7 | baseline |
| **ZeroQ 8-bit (no data)** | **77.4 (Δ = −0.3)** | calibrated on synth |
| ZeroQ 6-bit mixed (no data, + HAWQ) | 77.2 (Δ = −0.5) | matches PACT-6 *with* data |
| Uniform 6-bit PTQ (no data) | < 50 (collapsed) | scales overfit to random batch |

Within 0.2% of QAT methods that *do* have real ImageNet data. The "data-free" claim is real.

---

## Why this matters historically

ZeroQ proved that **calibration distributions, not calibration semantics, drive PTQ accuracy**. You don't need real images; you need a batch whose per-layer marginals match the FP model's training distribution. BN running stats provide that anchor for free.

This finding is conceptually deep: it implies PTQ is essentially solving a *moment-matching* problem — and any source of correct per-channel moments (synthetic data, teacher-generated text, real but cross-domain data) should work equally well.

---

## Why it fails on transformers

Transformers have **no BatchNorm** — only LayerNorm with no running statistics to anchor on. ZeroQ's distillation loss has nothing to optimise against.

The data-free idea survives in **[[llm-qat]]** (Liu et al. 2023): teacher-generated text replaces BN distillation as the calibration source. The recipe:

```text
1. Use the FP LLM to sample N sequences of length L (free, no extra data).
2. Calibrate the quantized student against those sequences.
3. (Optional) distill student outputs to match teacher logits.
```

Same data-free philosophy, different distributional anchor. Practitioners often skip even this in 2026 — modern LLM PTQ pipelines just calibrate on a generic corpus like C4 or WikiText, which is essentially free.

---

## What survives, what dies

| Idea | Survives? |
|---|---|
| Moment-matching as PTQ calibration | ✓ (LLM-QAT, BN-folding tricks) |
| BN-stat distillation specifically | dies (no BN in transformers) |
| HAWQ combination for mixed precision | mostly dies (uniform W4 + group-size wins) |
| "Calibration data isn't strictly required" | ✓ (mostly true for LLMs too) |

---

## Common pitfalls

- **Synthesised images look bad → assume the method failed.** Wrong: visual quality is uncorrelated with calibration quality. Check the per-layer activation histogram alignment, not the pixels.
- **Using BN-stats from a different model.** Statistics must come from the *exact* FP checkpoint you're quantizing; even small training differences shift per-layer moments.
- **Replacing 32 with 1 sample.** Larger synth batch gives more reliable per-channel mean estimates; below 8 samples, the calibration is too noisy.

---

## Connections

- [[excerpts/integer-only-inference]] — companion: the GEMM half of integer-only inference.
- [[ch-06]] — parent synthesis; ZeroQ is the no-data corner of the chapter.
- [[ch-04]] — HAWQ + AdaRound + BRECQ are the methods ZeroQ plugs into.
- [[llm-qat]] — LLM-era heir: teacher-generated text replaces BN stats.
