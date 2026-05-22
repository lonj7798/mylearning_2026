<!-- scope: QuIP# — Hadamard incoherence + E8 lattice codebooks + fine-tuning for sub-4-bit LLM PTQ
     deps: [[quip]]
     see-also: [[aqlm]], [[quarot]], [[spinquant]]
-->

# QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
- **Core Insight:** Once you've made the weight matrix incoherent via random Hadamard rotation, the optimal codebook is no longer scalar — it should be a *lattice* matched to the resulting near-Gaussian distribution, specifically the E₈ lattice which is optimal for 8-dim sphere packing.
- **Guideline:** For 2-bit weight-only LLM PTQ, use QuIP# = randomized Hadamard (incoherence) + E₈ lattice codebook (vector quant per 8-dim block) + light fine-tuning on calibration data; achieves near-FP16 accuracy at 2 bits on LLaMA-2-70B.
- **Authors:** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa
- **Year:** 2024 (ICML 2024)
- **URL:** https://arxiv.org/abs/2402.04396
- **Relevant topics:** lattice quantization, Hadamard incoherence, E8 lattice, sub-4-bit PTQ, fine-tuning

## Abstract
QuIP# extends QuIP with three innovations: (1) randomized Hadamard transforms providing strong, fast incoherence; (2) vector quantization with E₈ lattice codebooks, optimal for the spherical Gaussian distribution that incoherent weights approximate; (3) post-quantization fine-tuning recovering further accuracy. Achieves SOTA results in ≤4-bit extreme compression regimes.

## Key Contributions
- Replaces QuIP's expensive Kronecker random rotation with a fast randomized Hadamard transform, enabling per-row online application during inference.
- Uses the E₈ lattice (optimal 8-D sphere packing, gap-density 2^4) as the codebook — every block of 8 weights snaps to the nearest E₈ lattice point.
- Adds calibration-data fine-tuning of (a) the rotation seeds and (b) the residual after lattice quantization.
- Releases CUDA kernels for fused Hadamard + lattice decode + FP16 GEMM.

## Key Figures/Tables to Study
- **Figure 1:** Pareto curve — QuIP# vs QuIP, AQLM, GPTQ at 2/3/4 bits.
- **Figure 3:** E₈ lattice illustration — densest sphere packing in 8 dimensions.
- **Table 2:** LLaMA-2 70B 2-bit PPL — QuIP# vs AQLM, QuIP, SqueezeLLM.

## Technical Details

### Randomized Hadamard incoherence
Pre-multiply each weight column by a fast Hadamard transform `H_d · D` with D a random ±1 diagonal — same construction as [[quarot]]. After rotation, the distribution of weight entries is approximately N(0, σ²I) (incoherent), which is exactly the prior for which lattice quant is optimal.

### E₈ lattice codebook
- E₈ = the unique even unimodular lattice in 8 dimensions; achieves the densest packing of unit balls in ℝ⁸ (kissing number 240).
- A weight block of 8 entries is rounded to the nearest E₈ lattice point.
- For 2-bit/weight target: subset of 2^16 lattice points kept (16 bits per 8 weights = 2 bits/weight). The chosen subset is the inner shell of E₈ — also called the "D₈" sublattice subset — giving near-Gaussian-optimal scalar resolution per coordinate.
- Decoding is a single table lookup per 8-dim block.

### Fine-tuning
After lattice quantization, fine-tune (1) the Hadamard rotation seeds (small parameter count), (2) per-block scaling factors, and (3) optionally a low-rank residual. ≤200 SGD steps on a small calibration set; recovers up to 0.5 PPL at 2-bit.

### Inference cost
Hadamard transform: O(d log d) per token, fused into a single CUDA kernel with the lattice decode and GEMM. Throughput on H100 at 2-bit weight ≈ FP16 throughput at full memory; the bottleneck is HBM read of weights, which is now 8× smaller.

### Why E₈ specifically
The E₈ lattice has the highest packing density in dimensions ≤ 24 (proved by Viazovska 2017, Fields Medal). For an isotropic Gaussian source, lattice quantization with E₈ achieves rate-distortion within ~0.1 bit of the Shannon lower bound — better than scalar (round-to-nearest), product quantization (PQ), or random vector codes at the same bit budget.

## Connections
- Direct predecessor: [[quip]] — original incoherence-processing approach.
- Rotation siblings sharing the Hadamard insight: [[quarot]], [[spinquant]].
- Additive / vector quant siblings: [[aqlm]], [[vptq]], [[gptvq]].
- Lattice-theory ancestor: classical lattice quantization (Conway-Sloane); rate-distortion bound [[rate-distortion-theory]].
- Fine-tuning of compressed codes: [[pv-tuning]].
