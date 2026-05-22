<!-- scope: QuaRot — random Hadamard rotations remove outliers from LLM hidden states, enabling W4A4KV4 PTQ
     deps: [[quip]], [[gptq]]
     see-also: [[spinquant]], [[duquant]], [[flatquant]], [[quip-sharp]]
-->

# QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- **Core Insight:** Inserting computationally-invariant orthogonal (Hadamard) rotations into the residual stream and into attention/FFN projections destroys the large activation outliers that block 4-bit quantization, without changing model outputs — making true end-to-end W4A4KV4 inference possible.
- **Guideline:** When pursuing W4A4 PTQ, fuse a random Hadamard rotation into each linear's weights (Q·W with Q orthogonal) and apply the inverse rotation online to activations; pair with GPTQ for weights and round-to-nearest dynamic quant for activations.
- **Authors:** Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2404.00456
- **Relevant topics:** rotation-based PTQ, outlier elimination, W4A4 quantization, KV-cache quantization, computational invariance

## Abstract
QuaRot rotates LLMs in a way that removes outliers from the hidden state without changing the output, making quantization easier. The scheme quantizes all weights, activations, and KV cache to 4 bits via computational invariance: rotations are folded into adjacent weight matrices (offline) or applied via fast Hadamard transforms (online). On LLaMa2-70B the W4A4KV4 model loses at most 0.47 WikiText-2 perplexity and retains 99% of zero-shot performance, with no channels kept at higher precision.

## Key Contributions
- Identifies that residual-stream outliers are the fundamental blocker for W4A4 and shows they can be rotated away via orthogonal Q.
- Defines a "computationally invariant" rotation insertion pattern: each rotation that enters a residual branch is undone before it leaves, so logits are unchanged in FP.
- Folds rotations into weights offline where possible (Q·W·Qᵀ pattern across consecutive linears); applies online Hadamard transforms only at the FFN-down and value/output paths.
- Achieves first lossless W4A4KV4 PTQ for LLaMa-2 70B, plus calibration-free W6/W8 results.

## Key Figures/Tables to Study
- **Figure 1:** Activation magnitude before vs after Hadamard rotation — the dramatic flattening of the outlier spike.
- **Figure 2 / Section 3:** The rotation-insertion diagram for a transformer block (R1 residual, R2 V-cache, R3 FFN-down, R4 K-cache).
- **Table 2:** W4A4KV4 PPL on LLaMa2-7B/13B/70B — 5.6/5.0/3.8 vs FP16 5.5/4.9/3.3.

## Technical Details

### Rotation insertion (the invariance trick)
For a residual block `y = x + f(x)` and any orthogonal Q (QᵀQ = I), pre-rotate the input and post-rotate inside f:
`y = Qx + Q · f(Qᵀ · Qx) = Q(x + f(x))`
The residual stream is rotated by Q everywhere; classifier head absorbs Qᵀ at the end. Logits unchanged in full precision.

### Where Q lives in each weight (folded offline)
- `W_q, W_k, W_v ← W_{q,k,v} · Qᵀ` (input is Qx)
- `W_o ← Q · W_o` (output re-enters residual)
- `W_{up}, W_{gate} ← W_{up,gate} · Qᵀ`
- `W_{down} ← Q · W_{down}`
Embedding `E ← E · Qᵀ`; LM head `W_lm ← W_lm · Q`. All FP outputs identical to the un-rotated model.

### Hadamard choice
- **R1 (residual rotation):** a single randomized Hadamard `Q = H_d · D` (D = ±1 diagonal). H_d eliminates per-token outliers because Hx spreads any single large coordinate over all d coordinates with magnitude 1/√d.
- **R2 (V → O path):** Hadamard applied online after V, fused into W_o.
- **R3 (FFN-down):** online Hadamard between SwiGLU and W_down to kill the gated activation spikes.
- **R4 (K-cache):** Hadamard applied after RoPE to keys, fused with subsequent attention dot-product (since Hadamards commute with the dot product up to scale).

### Quantization recipe on top of rotation
- Weights: GPTQ at 4-bit, group size 128 (or per-channel).
- Activations: dynamic per-token round-to-nearest at 4-bit (symmetric).
- KV cache: per-head per-token at 4-bit.

### Why it works (the flattening claim)
For x with one outlier of magnitude M, `(H_d x)_i ≈ M/√d` for all i. The post-rotation max is reduced by ~√d (≈ 90× for d=8192), pulling the quantization range back to the dense bulk and dropping the round-to-nearest error to near-uniform-noise levels.

## Connections
- Direct ancestor: [[quip]] introduced incoherence processing via random rotation; QuaRot productionizes it end-to-end with KV-cache and activation paths.
- Successor with learned rotations: [[spinquant]].
- Successor adding permutation: [[duquant]].
- Successor with affine (non-orthogonal) flattening: [[flatquant]].
- Lattice-based weight codebook successor: [[quip-sharp]].
- Weight-side calibration paired with QuaRot: [[gptq]].
- Activation-flattening alternative (no rotation): [[smoothquant]], [[awq]].
