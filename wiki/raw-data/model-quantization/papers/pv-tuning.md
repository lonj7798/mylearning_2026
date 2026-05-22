<!-- scope: PV-Tuning — fine-tunes discrete codebook indices of compressed (AQLM/QuIP#) LLMs without straight-through estimator
     deps: [[aqlm]], [[quip-sharp]]
     see-also: [[straight-through-estimator]], [[vptq]]
-->

# PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression
- **Core Insight:** Fine-tuning extremely-compressed (1–2 bit) LLMs via the straight-through estimator (STE) is provably suboptimal because the discrete codebook indices have a structured solution geometry that STE ignores — direct combinatorial updates of the indices converge faster and to better minima.
- **Guideline:** When fine-tuning an AQLM or QuIP# quantized model, use PV-Tuning: alternate between (P) continuous codebook (centroid) updates by SGD and (V) discrete index updates by local search; replaces ad-hoc STE fine-tuning recipes.
- **Authors:** Vladimir Malinovskii, Denis Mazur, Ivan Ilin, Denis Kuznedelev, Konstantin Burlachenko, Kai Yi, Dan Alistarh, Peter Richtárik
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2405.14852
- **Relevant topics:** quantization-aware fine-tuning, discrete optimization, vector quantization fine-tuning, sub-2-bit LLM

## Abstract
Existing methods like QuIP# and AQLM employ straight-through-estimator (STE) fine-tuning after quantization, but STE assumes continuous gradients flow through the discrete quantizer — a biased and often unstable approximation in the extreme low-bit regime. PV-Tuning is a representation-agnostic framework that decouples *parameter* updates (continuous codebooks) from *value* updates (discrete indices), generalizing and improving over STE-based fine-tuning. Provides convergence guarantees in restricted settings. Achieves the first Pareto-optimal Llama-2 quantization at 2 bits/param; beats prior STE-based fine-tuning on Llama and Mistral.

## Key Contributions
- Frames quantized-model fine-tuning as a *mixed* continuous-discrete optimization: codebooks live in ℝ^{K×d}, indices live in {0, ..., K−1}^{N}.
- Replaces STE with an alternating PV scheme: (P) gradient step on codebooks holding indices fixed; (V) local discrete search on indices holding codebooks fixed.
- Provides theoretical convergence rate in the convex / strongly-convex regime.
- Demonstrates that PV beats STE fine-tuning on AQLM-2bit and QuIP#-2bit LLaMA-2 by margins worth ~0.3 PPL.

## Key Figures/Tables to Study
- **Figure 1:** STE vs PV training loss curves on Llama-2-7B — STE plateaus, PV continues descending.
- **Figure 2:** PV alternating-step diagram.
- **Table 2:** Llama-2 7B/13B/70B 2-bit PPL — AQLM+PV vs AQLM+STE vs QuIP#+PV.

## Technical Details

### Setup
Compressed weights:
`ŵ_g = Σ_m C_m[i_m^{(g)}]` (AQLM additive form) or `ŵ_g = LatticeDecode(i^{(g)})` (QuIP# E₈ form).
Variables: codebooks {C_m} (continuous, in ℝ^{K×d}) and indices {i_m^{(g)}} (discrete, in [0..K)).

Objective (calibration cross-entropy):
`L(C, i) = E_{x,y∼D_cal}[−log p_θ(y|x; C, i)]`

### PV alternation
1. **P-step (parameter):** freeze indices i, do SGD on codebooks C — fully differentiable.
2. **V-step (value):** freeze codebooks C, for each group g find better indices via local search:
   - `i_m^{(g)} ← argmin_{j ∈ N(i_m^{(g)})} L_g(C, i with i_m^{(g)} ← j)`
   - N is the k-nearest neighbour set in codebook space; typically k=8 to keep V-step cost manageable.
3. Alternate until convergence.

### Why this beats STE
STE backprops through the discrete quantizer with `dq/dx ≈ 1`, which is a biased gradient (the true derivative is 0 a.e. and δ on the bin boundaries). In the extreme low-bit regime the bin boundaries dominate and STE's bias drives optimization off-direction. PV's V-step makes the discrete jumps explicitly, with the *true* loss difference as the criterion — no gradient estimation needed for the discrete part.

### Convergence guarantee (informal)
Under convexity of L w.r.t. C and finite-radius local-search neighbourhood, the joint alternating scheme converges to a critical point in O(1/√T) — STE has no such guarantee.

### Cost
P-step: standard quantized-forward / dequantize-backward, same cost as STE fine-tuning.
V-step: per-group loss evaluation across k=8 candidate codes — adds ~30% wall-clock vs STE. Total: a few hundred steps on a small calibration set (~hundreds of MB).

## Connections
- Targets compressed format produced by: [[aqlm]], [[quip-sharp]], [[vptq]].
- The STE baseline it replaces: [[straight-through-estimator]].
- Related: quantization-aware training literature — [[lsq]], [[llm-qat]].
- Concurrent fine-tuning for compressed LLMs: [[bitdistiller]], [[efficientqat]].
