<!-- scope: AQLM — additive (multi-codebook) quantization brings classical product-quantization to 2-bit LLMs
     deps: [[product-quantization]], [[gptq]]
     see-also: [[pv-tuning]], [[vptq]], [[gptvq]], [[quip-sharp]]
-->

# Extreme Compression of Large Language Models via Additive Quantization (AQLM)
- **Core Insight:** At 2 bits and below, scalar quantization is provably suboptimal (Gish-Pierce bound is loose) — vector / additive quantization with multiple small learned codebooks reaches the Pareto frontier by exploiting *joint* structure across groups of weights.
- **Guideline:** For ≤3-bit weight-only LLM compression, use AQLM: encode each group of d weights as a sum of M codewords drawn from M trained codebooks (≈ 2^B entries each); learn codebooks and indices jointly per transformer block.
- **Authors:** Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh
- **Year:** 2024 (ICML 2024)
- **URL:** https://arxiv.org/abs/2401.06118
- **Relevant topics:** additive quantization, multi-codebook, sub-2-bit, vector quantization, joint block optimization

## Abstract
AQLM revisits extreme LLM compression (≤3 bits per parameter) through the lens of classical Multi-Codebook Quantization. Each weight group is approximated as a *sum* of codewords drawn from M trained codebooks, generalizing product quantization. Two innovations: (1) input-adaptive learning of the additive codebooks via a GPTQ-style Hessian update, and (2) joint optimization of codebook parameters across an entire transformer block. AQLM is the first scheme Pareto-optimal in accuracy-vs-size below 3 bits per parameter, with optimized GPU/CPU kernels matching or beating FP16 speed.

## Key Contributions
- Adapts Additive Quantization (AQ, from ANN search literature) for LLM weights.
- Defines the AQLM quantization rule: a group of weights w ∈ ℝ^d is approximated as Σ_{m=1..M} C_m[i_m], where each codebook C_m has 2^B entries.
- Hessian-aware codebook + index learning (GPTQ heritage) compensates for input distribution.
- Block-level joint calibration that updates codebooks together across all linears in a transformer block, exploiting cross-layer dependencies.
- Reaches Pareto-optimal accuracy at 2 bits/param on LLaMA-2 70B.

## Key Figures/Tables to Study
- **Figure 1:** Pareto curve (PPL vs bits/param) — AQLM dominates QuIP#, GPTQ-2bit, OmniQuant in the sub-3-bit regime.
- **Figure 2:** Additive-quantization decoding schematic — M sequential lookups + sum.
- **Table 2:** LLaMA-2 7B/13B/70B at 2-bit and 2.5-bit — AQLM vs QuIP# vs SqueezeLLM.

## Technical Details

### The additive quantization rule
Split each linear's weights into groups of d (typically d = 8 or 16). Each group `w ∈ ℝ^d` is approximated by
`ŵ = Σ_{m=1}^{M} C_m[i_m]`
where
- `C_m ∈ ℝ^{2^B × d}` is the m-th codebook (each row is one length-d codeword),
- `i_m ∈ {0, ..., 2^B − 1}` is the m-th selected index,
- M codebooks, M indices per group.

Total bits per group = M·B (indices) + amortised codebook bits.
Bits per weight = (M·B) / d + ε (codebook overhead).
Example: M=2, B=8, d=8 → 16 bits per 8 weights = 2.0 bits/weight; codebook 2·256·8·FP16 = 8 KB per linear, amortised over millions of weights → ε ≈ 0.05.

### Why additive beats product quantization (PQ)
- PQ partitions the d-dim group into M disjoint sub-vectors and quantizes each independently.
- AQ allows codewords from each codebook to span the *full* d dimensions, with the sum recovering the group. Far more expressive at the same bit budget — captures cross-coordinate structure that PQ rules out by construction.

### Encoding (the hard combinatorial problem)
Picking the best (i_1, ..., i_M) given codebooks is a discrete-OPT problem. AQLM uses beam search (beam size ~ 1) plus iterative residual encoding:
1. `r ← w`; for m = 1..M: `i_m ← argmin_j ||r − C_m[j]||²; r ← r − C_m[i_m]`.
2. Refine by local search (swap one i_m at a time).

### Hessian-aware codebook update (GPTQ heritage)
Codebooks are not learned by raw weight MSE; they are learned by *output* reconstruction MSE, which is a weighted-MSE over weights with weight `diag(X^T X) / n` (the per-column input variance — same as GPTQ's diagonal Hessian). For full-matrix dependency they apply the GPTQ sequential column update.

### Block-level joint optimization
After per-linear AQLM, freeze indices and fine-tune all codebooks in a transformer block jointly to minimize block-output MSE on a small calibration set. Recovers ~0.3 PPL.

### Kernel
GPU decode = M cache-resident table lookups + horizontal sum, then a standard FP16 matmul against the dequantized W. AQLM ships fused CUDA kernels reaching ≥ FP16 throughput on A100/H100 at 2 bits.

## Connections
- Classical ancestor: [[product-quantization]] (Jégou 2011); generalised here from disjoint sub-vectors to overlapping codeword sums.
- Hessian-aware lineage: [[gptq]], [[obc]], [[obs-obd]].
- Concurrent / sibling vector PTQ: [[vptq]], [[gptvq]], [[quip-sharp]].
- Fine-tuning of the resulting discrete codes: [[pv-tuning]] (the natural follow-up).
- 1-bit alternative line: [[bitnet-b158]], [[onebit]].
