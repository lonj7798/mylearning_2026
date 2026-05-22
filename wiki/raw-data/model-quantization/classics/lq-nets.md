<!-- scope: LQ-Nets — learnable basis vectors for non-uniform quantizers
     deps: straight-through-estimator, dorefa-net
     see-also: lsq, pact, squeezellm
-->

# LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks
- **Core Insight:** Instead of forcing weights/activations onto a uniform grid, learn a small set of K basis vectors so the quantized value is `q = Σ b_k · v_k`, b_k ∈ {−1,+1}; the basis is updated jointly with weights so the codebook matches the distribution.
- **Guideline:** Use LQ-Nets when activation distributions are heavy-tailed (post-ReLU exponential or LayerNorm output); k-bit budget gives 2^k codebook entries built as sums of K basis vectors; quantize via closed-form least-squares assignment given the basis.
- **Authors:** Dongqing Zhang, Jiaolong Yang, Dongqiangzi Ye, Gang Hua
- **Year:** 2018
- **URL:** https://arxiv.org/abs/1807.10029
- **Relevant topics:** non-uniform quantization, learnable codebook, basis vectors, inner-product-friendly

## Abstract
Most QAT works (BNN, DoReFa, PACT) assume uniform quantization levels. LQ-Nets relaxes this: each quantizer maintains K real-valued basis vectors v₁,…,v_K, and quantized values are formed as signed sums q = Σ b_k v_k with b_k ∈ {−1,+1}. The basis vectors are learned to minimise quantization error on a running set of pre-quantization weights/activations, while the binary code b is chosen by least-squares assignment. Because the code is still binary, the inner product W·A reduces to K² XNOR-popcount kernels with K small scalars — keeping the speed advantage of binary networks but with much richer codebooks.

## Key Contributions
- Learnable non-uniform quantizer parameterised by K basis vectors (vs fixed uniform grid).
- Closed-form least-squares assignment of binary codes given basis.
- Compatible with binary-style inner-product kernels (K² XNOR-popcount instead of 1 fp matmul).
- ImageNet ResNet-18: 2-bit W/2-bit A reaches 64.9% top-1 vs DoReFa's 62.6%.

## Key Figures/Tables to Study
- **Figure 2** — codebook visualisation: learned grid vs uniform grid for ResNet activations.
- **Table 2** — head-to-head vs DoReFa-Net and PACT across bit budgets.

## Technical Details

### Quantizer parameterisation
K-bit budget → K basis vectors v ∈ ℝ^K (one per output channel for weights):
`Q(x) = vᵀ b,  b ∈ {−1,+1}^K`
Codebook has 2^K levels (not 2^K uniform spaced).

### Code assignment (forward)
Given x and current basis v, choose b to minimise (x − vᵀb)²:
`b* = arg min_{b ∈ {−1,+1}^K} (x − vᵀb)²`
solved by enumeration when K small (K=2 → 4 codes, K=3 → 8 codes); closed-form via dynamic programming for larger K.

### Basis update (per minibatch)
With code matrix B ∈ {−1,+1}^{K×N} from N samples and target vector x ∈ ℝ^N:
`v ← (B Bᵀ)⁻¹ B x`
A small per-batch least-squares solve; updated after every gradient step.

### Inner product structure
For weight basis v_w and activation basis v_a (lengths K_w, K_a), with codes b_w, b_a:
`W·A = Σ_{i,j} v_w[i] v_a[j] · ⟨b_w[i], b_a[j]⟩`
The inner products ⟨b_w[i], b_a[j]⟩ are binary — K_w·K_a XNOR-popcount kernels.

### STE for codes
Backward through arg-min uses STE: `∂Q/∂x ≈ 1` inside the quantizer's clipping range.

## Connections
- [[straight-through-estimator]] — gradient mechanism through the basis quantizer.
- [[dorefa-net]] — uniform-quantizer baseline LQ-Nets beats.
- [[pact]] — combines well: PACT clip + LQ codebook.
- [[squeezellm]] — modern LLM-era reincarnation of non-uniform codebooks via sensitivity-weighted k-means.
- [[product-quantization]] — analogous decomposition into smaller-dim codebooks.
