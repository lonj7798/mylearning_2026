<!-- scope: Abstract Float / Asymmetric Float 4-bit variants — successors and tunings of NF4
     deps: [[nf4]], [[lloyd-max-quantizer]]
     see-also: [[int4]], [[hqq]], [[fp4-e2m1]]
-->

# AF4 — Abstract / Asymmetric Float 4-bit Variants
- **Core Insight:** NF4 fixes a symmetric, Gaussian-optimal 4-bit codebook, but real LLM weight blocks are mildly asymmetric (skewness ≠ 0) and have heavier tails than N(0,1); AF4 variants either (a) drop the symmetry constraint and re-tune codebook per layer, or (b) learn the codebook directly from the empirical weight distribution to recover the residual 0.1–0.3 PPL gap NF4 leaves.
- **Guideline:** Use AF4 / learned-codebook 4-bit when squeezing the last fraction of a PPL out of W4 quantization matters and you can afford the per-layer calibration cost; otherwise stay with NF4 (cheaper, near-equivalent quality).
- **Authors:** Various — initially explored by Dettmers and collaborators as a NF4 follow-up; refined in the SqueezeLLM (Kim 2023) and HQQ (Badri 2024) lineages as "non-uniform LUT" or "learned codebook" methods
- **Year:** 2023–2024
- **URL:** https://arxiv.org/abs/2306.03078 (SpQR; discusses non-uniform code design); https://arxiv.org/abs/2306.07629 (SqueezeLLM, non-uniform LUT); https://mobiusml.github.io/hqq_blog/ (HQQ)
- **Relevant topics:** non-uniform 4-bit, learned codebook, asymmetric quantile, sensitivity-weighted LUT

## Abstract
AF4 ("Asymmetric Float 4") and related "Abstract Float" variants are non-uniform 4-bit codes that generalize NF4 by removing one or more of NF4's simplifying assumptions: (a) the codebook need not be symmetric around zero (asymmetric variant), (b) the codebook need not match a fixed Gaussian prior (learned variant), (c) the codebook can be weighted by per-weight Fisher / Hessian sensitivity (SqueezeLLM-style sensitivity-aware). Each refinement closes a fraction of the remaining gap between NF4 and the layer-wise Lloyd-Max optimum, at the cost of per-layer (or per-block) calibration storage. None has become a single canonical "AF4" specification; the term covers a family of refinements.

## Key Contributions
- Generalizes NF4 to asymmetric / data-fit codebooks.
- Recovers residual quality not captured by NF4's symmetric Gaussian assumption.
- Bridges NF4 (data-free) and SqueezeLLM (data-driven non-uniform).
- Demonstrates that 4-bit quality is bounded by the *codebook choice*, not the bit-width — once you exceed Lloyd-Max for the actual distribution you're at the rate-distortion floor.

## Key Figures/Tables to Study
- **Asymmetric vs symmetric 4-bit codebook PPL** on Llama-7B/13B — asymmetric wins by ~0.05–0.15 PPL when weight blocks have skew > 0.2.
- **Sensitivity-weighted codebook** (SqueezeLLM Fig. 3): k-means on weights *weighted* by squared Fisher information per element → ~0.2 PPL gain over plain k-means.

## Technical Details

### Generic non-uniform 4-bit code
A 4-bit non-uniform code is a 16-entry LUT C[0..15]:
```
encode(w):   q = argmin_k |C[k] − w/s|
decode(q):   w_recon = s · C[q]
```
where s is a per-block / per-channel scale and C is the chosen codebook.

### AF4 (asymmetric float) variant
Drop the constraint C[k] = −C[15−k]. Codebook is fit by:
```
C* = argmin_C Σ_i (w_i − s · NN_C(w_i / s))²
```
solved via Lloyd-Max iteration on the empirical weight distribution (k-means with K = 16 in 1-D). Per-block (or per-channel) codebook stored — 16 × 8-bit = 128 bits overhead, amortized over the block.

### Sensitivity-weighted variant (SqueezeLLM-style)
Weight the k-means objective by per-element Fisher information F_i = (∂L/∂w_i)² estimated from a small calibration set:
```
C* = argmin_C Σ_i F_i · (w_i − s · NN_C(w_i / s))²
```
This makes the codebook spend more resolution on *high-sensitivity* weights — empirically the most important refinement beyond NF4.

### Storage cost (with per-channel codebook)
```
4 (weight) + 16 × 16 / channel_size            (FP16 codebook per channel)
= 4 + 256 / 4096  (for Llama-7B intermediate dim)
≈ 4.06 effective bits/weight
```
Cheaper than NF4 with double quant (4.127 bits/weight), but each kernel pass needs the per-channel LUT in shared memory.

### When AF4 beats NF4
- Layers with significant weight-distribution skew: early embedding rows, output-projection biases-baked-in, certain attention layers with heavy-tailed outputs.
- Tested fine-tuned (LoRA-merged) checkpoints where the base distribution has shifted away from pretraining-Gaussian.
- Sub-3-bit regimes (extended AF3 / learned 3-bit codes) where the symmetric Gaussian assumption is too coarse.

### When NF4 is sufficient
- Mid-training pretrained weights of Llama-class models: NF4 captures > 99% of the quality.
- Inference where per-layer codebook storage is unacceptable (e.g. mobile, very tight memory).
- Plug-and-play deployment (no calibration data needed).

### Comparison table
| Variant | Codebook source | Per-tensor data? | Typical PPL gap vs FP16 |
|---------|----------------|------------------|--------------------------|
| INT4 (RTN) | uniform | no | 0.5–1.5 |
| NF4 | fixed Gaussian quantile | no | 0.3–0.7 |
| AF4 (asymmetric) | Lloyd-Max per-block | optional | 0.2–0.5 |
| SqueezeLLM LUT | Fisher-weighted k-means | yes | 0.15–0.3 |
| GPTQ INT4 | uniform + Hessian update | yes | 0.1–0.3 |
| AWQ INT4 | uniform + activation scale | yes | 0.1–0.3 |

### Lloyd-Max optimum as floor
At 4 bits, the Lloyd-Max-optimal layer-specific code achieves the Gish-Pierce floor of D ≈ ||p||_{1/3}³ / (12 · 2^8). For Llama-7B intermediate weights this corresponds to ~0.1 PPL gap from FP16 — the absolute lower bound any scalar 4-bit code can achieve. AF4 + sensitivity weighting brings you within ~0.1 PPL of this; the remaining gap requires vector quantization ([[aqlm]]) or rotation ([[quarot]]).

### Practical guidance
- Default: **NF4** (no data, no per-layer storage, near-best quality).
- Quality-critical: **GPTQ** or **AWQ** with INT4 + group-128 (data-driven, no LUT storage).
- Sub-2-bit research: move to **AQLM** / **QuIP#** (vector codes), not further LUT refinement.

## Connections
- [[nf4]] — the symmetric Gaussian-prior baseline AF4 refines.
- [[lloyd-max-quantizer]] — AF4's per-block codebook is a 16-level Lloyd-Max code.
- [[squeezellm]] — sensitivity-weighted LUT; canonical AF4 instantiation.
- [[hqq]] — half-quadratic data-free non-uniform code.
- [[int4]] — uniform alternative.
- [[information-theoretic-bounds]] — Gish-Pierce p^{1/3} density is the asymptotic target.
- [[aqlm]] — vector-quantization alternative that beats AF4 at sub-2-bit.
