<!-- scope: Gholami 2021 — canonical survey of quantization methods for efficient neural network inference
     deps:
     see-also: [[quantization-mapping]], [[lsq]], [[hawq]], [[adaround]]
-->

# A Survey of Quantization Methods for Efficient Neural Network Inference (Gholami et al. 2021)
- **Core Insight:** Pre-LLM neural-network quantization is organized along three orthogonal axes — uniform vs non-uniform code, symmetric vs asymmetric mapping, and PTQ vs QAT calibration — and within each cell the dominant tradeoff is between calibration-data requirement and recoverable accuracy.
- **Guideline:** Use this survey as the canonical taxonomy reference when planning a quantization pipeline; the decision tree (per-tensor vs per-channel, symmetric vs asymmetric, PTQ vs QAT, uniform vs non-uniform) maps cleanly onto modern LLM-era methods.
- **Authors:** Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer (UC Berkeley)
- **Year:** 2021 (published in book "Low-Power Computer Vision," 2022)
- **URL:** https://arxiv.org/abs/2103.13630
- **Relevant topics:** survey, quantization taxonomy, PTQ, QAT, mixed-precision, calibration

## Abstract
Gholami et al. survey the pre-2021 deep-learning quantization literature, structuring the field along three axes: (1) uniform vs non-uniform quantization grids, (2) symmetric vs asymmetric mapping, (3) post-training quantization (PTQ) vs quantization-aware training (QAT). The survey covers integer-only inference (Jacob 2018), per-channel scaling, dynamic vs static activation quantization, straight-through estimators, learned step size (LSQ), HAWQ mixed-precision, AdaRound, and clip-aware methods. It standardizes the vocabulary that the entire LLM-quantization literature inherited — every modern paper from LLM.int8() onward uses Gholami's taxonomy. The survey predates the LLM era, so it does not cover outlier handling, rotation, KV-cache quant, or 1-bit LLMs, but is the canonical foundation reference.

## Key Contributions
- Unified taxonomy: uniform/non-uniform × symmetric/asymmetric × PTQ/QAT.
- Decision-tree for choosing a quantization method based on data availability and target hardware.
- Comprehensive comparison of integer-only, mixed-precision, and binary methods on CV benchmarks.
- Establishes vocabulary (calibration, activation range, dynamic vs static quant, fake-quant).
- Single-document reference for the pre-LLM lineage that every modern LLM-quant paper builds on.

## Key Figures/Tables to Study
- **Figure 4 (Quantization taxonomy)**: the canonical decision tree; reused in nearly every subsequent quantization talk.
- **Table 1 (PTQ vs QAT comparison)**: when to use each, calibration-set size requirements, expected accuracy gap.
- **Table 4 (Non-uniform quantization)**: log / power-of-2 / k-means codes compared on ImageNet ResNets.

## Technical Details

### Three orthogonal axes

**1. Uniform vs non-uniform code**
- **Uniform**: reconstruction values evenly spaced (INT8, INT4); cheap hardware (just shift + add).
- **Non-uniform**: log-spaced, k-means, or learned codebook; better quality for log/Gaussian distributions; needs LUT or shift-and-add tricks.

**2. Symmetric vs asymmetric mapping**
- **Symmetric**: `x_q = round(x/s)`; force zero-point z = 0. Cheaper matmul (no cross-terms).
- **Asymmetric**: `x_q = round(x/s) + z`; uses full integer range for one-sided distributions (ReLU, softmax).

**3. PTQ vs QAT**
- **PTQ (post-training)**: calibrate on small unlabeled set; one-shot quantize a trained FP model. Cheap; ~1–2% accuracy drop typical.
- **QAT (quantization-aware training)**: insert fake-quant ops during fine-tuning; learn around quantization. Expensive; ~0.1–0.5% drop typical.

### Per-tensor vs per-channel vs per-group granularity
- Per-tensor: one (s, z) per tensor. Fastest, lowest quality.
- Per-channel (per-output-channel): one (s, z) per weight row. Standard for CNN/transformer weights.
- Per-group (sub-channel): finer; standard for sub-INT8 (group-128 INT4).

### Static vs dynamic activation quantization
- **Static**: pre-compute activation ranges on calibration set; fixed at inference. Fast.
- **Dynamic**: compute range online per inference. Slower; needed for sequence-length-varying activations (transformers, RNNs).

### Mixed-precision quantization (HAWQ)
Different layers / different operations get different bit-widths. The Hessian eigenvalue of the loss w.r.t. each weight group gives a sensitivity score; high-sensitivity layers get more bits. Cited as foundation for [[hawq]] and subsequent sensitivity-aware methods including [[squeezellm]].

### Quantization-aware training framework
Insert "fake-quantize" nodes:
```
forward:  x_fake = dequant(quant(x))    (quantize → dequantize round-trip)
backward: ∂L/∂x = ∂L/∂x_fake             (STE; identity-pass)
```
Train with this in the graph; deploy by removing the dequant step.

### Calibration objectives
- **MSE minimization**: `s* = argmin_s Σ (x − dequant(quant(x; s)))²`
- **Cosine similarity**: minimize 1 − cos(W, W_q)
- **KL divergence**: between FP and quantized output distributions (TensorRT default)
- **Percentile clipping**: clip range to a percentile (e.g. 99.9%) before quantizing
- **Hessian-weighted**: weight per-element MSE by the loss Hessian diagonal (foundation of OBS, GPTQ)

### Methods covered (selected)
- **Integer-only inference** (Jacob 2018): full INT8 inference graph; no FP ops at runtime.
- **PACT** (Choi 2018): learned activation clip threshold.
- **LSQ** (Esser 2020): learn the quantizer step size itself.
- **AdaRound** (Nagel 2020): per-weight learned rounding direction; closed-form via Hessian.
- **HAWQ** (Dong 2019): Hessian-aware mixed-precision bit allocation.
- **DoReFa** (Zhou 2016): arbitrary-bit weights, activations, gradients.
- **BNN / XNOR-Net** (Courbariaux / Rastegari 2016): binary weights via STE.

### Pre-LLM blind spots
The survey predates:
- Outlier-channel handling at LLM scale (LLM.int8(), SmoothQuant).
- Equivalent-transformation methods (AWQ, OmniQuant).
- Rotation-based quant (QuaRot, SpinQuant).
- KV-cache quantization (KIVI, KVQuant).
- Sub-2-bit LLM quant (QuIP, AQLM).
- Native low-precision LLM training (FP8-LM, BitNet, DeepSeek V3 FP8).

But every one of those methods cites Gholami 2021 for its taxonomy section.

### When to use this survey
- Onboarding to quantization: read this *before* any modern LLM-quant paper.
- Vocabulary disambiguation: "per-tensor" vs "per-channel" vs "per-group" definitions are canonical here.
- Reviewer reference: cite for any PTQ/QAT classification.

## Connections
- [[quantization-mapping]] — Krishnamoorthi 2018 whitepaper covering the same taxonomy from Google's perspective.
- [[lsq]] / [[adaround]] / [[hawq]] — specific methods covered in detail.
- [[straight-through-estimator]] — gradient device the survey covers.
- [[survey-llm-quantization-2024]] — modern LLM-only successor.
- [[integer-only-inference]] — Jacob 2018; the foundational pre-LLM PTQ paper.
- [[llm-int8]] — first major method outside the survey's scope; outlier discovery starts the LLM era.
