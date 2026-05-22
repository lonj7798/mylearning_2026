<!-- scope: ZeroQ — data-free PTQ via BN-statistics-distilled synthetic data
     deps: data-free-quantization, quantization-mapping
     see-also: hawq, brecq, llm-qat
-->

# ZeroQ: A Novel Zero Shot Quantization Framework
- **Core Insight:** Calibration data isn't strictly required — synthetic images can be generated to match the per-layer batch-norm running mean/variance, producing a "distilled" dataset that calibrates a PTQ scheme nearly as well as real data; the BN stats themselves encode enough distributional structure.
- **Guideline:** For each conv layer with BN, optimise a synthetic input batch `x` (initialised as Gaussian noise) to minimise `Σ_ℓ ‖μ_ℓ(x) − μ_ℓ^{BN}‖² + ‖σ_ℓ²(x) − σ_ℓ²^{BN}‖²`; use the resulting 32–64 synthetic images as the PTQ calibration set; combine with Hessian-aware mixed precision (HAWQ) for the bit allocation.
- **Authors:** Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami, Michael W. Mahoney, Kurt Keutzer
- **Year:** 2020 (CVPR)
- **URL:** https://arxiv.org/abs/2001.00281
- **Relevant topics:** data-free PTQ, BN-stat distillation, synthetic calibration, mixed-precision

## Abstract
ZeroQ tackles the deployment scenario where calibration data is unavailable (privacy, gating, cross-organisation handoff). Rather than rely on real images, it synthesises a tiny calibration set by gradient-optimising random inputs to match the per-layer batch-norm running statistics of the FP model. These "distilled" images suffice for calibrating per-channel weight scales and per-tensor activation scales. ZeroQ is then combined with HAWQ-style Hessian-aware mixed precision: more bits to layers with high trace(H_ℓ), fewer to flat-loss layers. On ImageNet ResNet-50 the system delivers 6-bit mixed-precision quantization within 0.5% of FP, using zero real images.

## Key Contributions
- BN-statistics-matching synthetic data generation — a closed pipeline requiring only the FP model.
- Integrates with HAWQ for mixed-precision bit allocation.
- 30-second total calibration time on a single GPU (synthesis + PTQ).
- Demonstrates near-FP accuracy at 6/8-bit mixed precision data-free on ResNet, MobileNet, InceptionV3.
- Predecessor to LLM-QAT's data-free fine-tuning via teacher self-generation.

## Key Figures/Tables to Study
- **Figure 3** — synthesised images vs real ImageNet; visualises that mode-matching is enough.
- **Table 2** — ResNet-50 mixed precision 6-bit: ZeroQ data-free vs PACT with real data.

## Technical Details

### BN distillation loss (the load-bearing formula)
Let μ_ℓ^{BN}, σ_ℓ^{2,BN} be the BN running stats of layer ℓ in the FP model.
For a synthetic batch x ∈ ℝ^{B×C×H×W}:
`L_dist(x) = Σ_ℓ ‖μ_ℓ(x) − μ_ℓ^{BN}‖_F² + ‖σ_ℓ²(x) − σ_ℓ²^{BN}‖_F²`
where μ_ℓ(x), σ_ℓ²(x) are the per-channel mean / variance computed on the forward pass of x through the FP model up to layer ℓ.

### Synthesis procedure
1. Initialise x ~ N(0, 1), shape (32, 3, 224, 224).
2. Adam optimise x for ~500 steps to minimise L_dist(x).
3. Output x is the synthetic calibration set.

No model weights are updated. The optimization shape-matches per-channel statistics — sufficient for activation-scale calibration because PTQ cares about the per-channel marginal distribution, not realistic semantics.

### PTQ step (after synthesis)
- Per-channel weight scale: S_w_c = max|W_c|/Q_max (purely weight-driven; doesn't need data).
- Per-tensor activation scale: percentile (99.9%) of activation magnitudes on synthetic batch.

### HAWQ mixed-precision allocation
For each layer ℓ, compute top eigenvalue λ_ℓ of the layer-output Hessian H_ℓ via Hutchinson trace estimator (a few backward passes on synthetic data). Sort layers by `λ_ℓ · ‖ΔW_ℓ(b)‖²` and assign bit-widths to satisfy a model-size budget.

### Compute cost
- Synthesis: 500 Adam steps × 32 images ≈ 10 seconds on V100.
- PTQ scale + HAWQ allocation: ~20 seconds.
- Total: ~30 seconds end-to-end.

### Failure mode
Networks without BN (transformers, LayerNorm-only architectures) cannot use ZeroQ directly — the running-stat anchor is absent. Workarounds in the LLM era use teacher-generated text (see [[llm-qat]]) as the analogous "distilled" calibration.

## Connections
- [[data-free-quantization]] — alternative data-free strategy (equalize + bias-correct instead of synthesise).
- [[hawq]] — direct co-publication on mixed-precision bit allocation; ZeroQ feeds HAWQ.
- [[quantization-mapping]] — sits inside the PTQ branch.
- [[brecq]] — block-wise PTQ that can consume ZeroQ-generated calibration data.
- [[llm-qat]] — LLM-era heir: replaces BN distillation with teacher-generated text for the same data-free goal.
