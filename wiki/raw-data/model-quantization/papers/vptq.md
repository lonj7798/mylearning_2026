<!-- scope: VPTQ — vector post-training quantization with channel-independent second-order optimization
     deps: [[product-quantization]], [[gptq]]
     see-also: [[aqlm]], [[gptvq]], [[quip-sharp]]
-->

# VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models
- **Core Insight:** Vector quantization on LLM weights can be made Hessian-aware just like scalar GPTQ — for each output channel, treat the per-channel weight vector as a sequence of d-dim groups and apply a GPTQ-style sequential Cholesky update where each "step" picks an entire codeword (not a scalar) from the codebook.
- **Guideline:** For 2-bit weight-only LLM PTQ with vector codebook, use VPTQ: channel-independent second-order optimization + decomposed codebook initialization + residual / outlier sub-codebooks for the top ~1% of groups.
- **Authors:** Yifei Liu, Jicheng Wen, Yang Wang, Shengyu Ye, Li Lyna Zhang, Ting Cao, Cheng Li, Mao Yang (Microsoft)
- **Year:** 2024 (EMNLP 2024)
- **URL:** https://arxiv.org/abs/2409.17066
- **Relevant topics:** vector PTQ, second-order Hessian, channel-independent quantization, residual codebook

## Abstract
VPTQ formulates LLM vector quantization as a second-order optimization problem and derives a Channel-Independent Second-Order Optimization (CISO) algorithm for granular per-channel codebook learning. It introduces (1) a codebook initialization scheme from a decomposed optimization sub-problem and (2) extensions for residual and outlier vector codebooks. Achieves 0.01–0.34 PPL improvement on LLaMA-2 and 11–22% QA accuracy improvement on LLaMA-3 at 2-bit vs prior VQ methods, while running in 10–19% of the calibration time.

## Key Contributions
- Defines the per-channel VQ objective as `min_{C, {i_g}} Σ_g (ŵ_g − w_g)^T H (ŵ_g − w_g)` with H the Hessian of the layer's output reconstruction.
- Channel-Independent Second-Order Optimization (CISO): decouples the channel-axis allowing parallel solves, dramatically cutting calibration time.
- Codebook initialization via decomposition: solves a smaller weighted-k-means sub-problem to seed the codebook before joint refinement.
- Residual + outlier sub-codebooks: ~1% of groups with high reconstruction error get a second-stage residual codeword from a smaller codebook.

## Key Figures/Tables to Study
- **Figure 2:** CISO algorithm pseudocode — the per-channel Hessian update loop.
- **Table 1:** LLaMA-2/3, Mistral at 2-bit — VPTQ vs AQLM vs QuIP#.
- **Figure 5:** Throughput/calibration time vs PPL — VPTQ Pareto curve.

## Technical Details

### Vector quantization rule
For each linear, each output-channel weight vector w ∈ ℝ^{C_in} is split into G groups of d entries; each group is replaced by a codeword from a learned codebook of size K = 2^B:
`ŵ_g = C[i_g]`, i_g ∈ {0, ..., K−1}, C ∈ ℝ^{K×d}.
Bits per weight = B/d + amortised codebook bits.

### Channel-Independent Second-Order Optimization (CISO)
For each output channel independently:
1. Build per-channel diagonal Hessian `H = diag(X^T X)` (input covariance restricted to that channel's column block).
2. Initialize indices i_g by nearest-codeword in the H-weighted norm.
3. Sequentially update groups: for g = 1..G,
   `i_g ← argmin_k (C[k] − w_g)^T H_g (C[k] − w_g)` and propagate the Cholesky residual to remaining groups (GPTQ-style).
4. After all indices fixed, refit codebook C via weighted k-means on (w_g, H_g).
5. Iterate 3–4 a few times.

The "channel-independent" part is key: each output channel has its own H_g and can be solved in parallel across channels — massively cuts calibration time vs joint multi-channel solvers.

### Codebook initialization (decomposed sub-problem)
Solve a smaller weighted PCA on the full weight matrix to obtain initial K codewords; then run weighted k-means restricted to a subsample for refinement. Avoids the bad local minima that random k-means initialization falls into in the d-dim continuous space.

### Residual + outlier extensions
Pick the top 1% of groups by reconstruction error; assign each a secondary code from a smaller residual codebook C_res ∈ ℝ^{K_res × d}. Effective bit cost increases by `(K_res bits) × 1%` ≈ +0.05 bits/weight, but reconstruction error on outlier groups drops by ~5×.

### Bits/weight examples
- d = 8, B = 12, K = 4096 → 1.5 bits/weight.
- d = 8, B = 16, K = 65536 → 2.0 bits/weight.
- d = 16, B = 12 → 0.75 bits/weight (super-low regime).

## Connections
- Hessian / second-order ancestry: [[gptq]], [[obc]], [[obs-obd]].
- VQ ancestors: [[product-quantization]], [[vector-quantization]].
- Concurrent / sibling VQ-LLM: [[aqlm]] (additive form, multiple codebooks), [[gptvq]] (related Hessian-update VQ).
- Lattice-based VQ alternative: [[quip-sharp]].
- Discrete-index fine-tuning: [[pv-tuning]].
