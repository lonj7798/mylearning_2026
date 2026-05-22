<!-- scope: Binary Neural Networks — first end-to-end 1-bit weight + activation training
     deps: straight-through-estimator
     see-also: xnor-net, dorefa-net, bitnet
-->

# Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or −1
- **Core Insight:** Networks can be trained from scratch with both weights and activations constrained to {-1, +1} if the forward sign() uses real-valued shadow weights and the backward pass uses an STE through a tanh-shaped surrogate.
- **Guideline:** Keep a real-valued latent weight buffer for the optimizer; binarize via sign() on the forward pass; clip the latent weights to [−1, 1] after each update; use the `1[|x|≤1]` (clipped STE) gradient for activations.
- **Authors:** Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio
- **Year:** 2016
- **URL:** https://arxiv.org/abs/1602.02830
- **Relevant topics:** 1-bit weights, 1-bit activations, QAT, XNOR-popcount, BitNet lineage

## Abstract
The paper introduces Binarized Neural Networks (BNNs), where weights and activations are constrained to two values (+1, −1) during inference and forward propagation. Real-valued gradients accumulate into latent weights between updates. On MNIST, CIFAR-10 and SVHN, BNNs reach near-FP32 accuracy. The binary multiply collapses to XNOR + popcount, giving a 32× memory reduction and large speedups on custom hardware. BNNs are the proof-of-concept that established the entire low-bit training literature.

## Key Contributions
- End-to-end 1-bit training pipeline: forward sign(), backward STE.
- Demonstrates that shadow real-valued weights are necessary for SGD to find good binary minima.
- Replaces fp matmul with XNOR + popcount for binary-binary inner products.
- Empirically: MNIST 0.96% error, CIFAR-10 10.15% error with 1-bit weights/activations.
- Shows batchnorm shift-based approximations make binary inference fully integer.

## Key Figures/Tables to Study
- **Algorithm 1** — full training loop with shadow weights, sign() forward, STE backward, weight clipping.
- **Table 1** — accuracy vs FP baseline on MNIST/CIFAR/SVHN; demonstrates feasibility.
- **Figure of XNOR-popcount** — shows binary inner product = `2·popcount(XNOR(a,b)) − N`.

## Technical Details

### Weight binarization (deterministic)
`W_b = sign(W_r) ∈ {−1, +1}`
Latent real weight W_r is updated by SGD; W_b is what enters the matmul.
After each step: `W_r ← clip(W_r − η·g, −1, 1)`.

### Activation binarization
`a_b = sign(a_r)` with STE backward:
`∂L/∂a_r = ∂L/∂a_b · 1[|a_r| ≤ 1]` (clipped STE — gradient passes inside [−1,1], dies outside).

### Binary inner product
For x_b, w_b ∈ {−1,+1}^n:
`xᵀw = N − 2·popcount(x_b XOR w_b)`
implementable on a CPU with one xnor + one popcnt instruction per 64-bit word.

### Stochastic variant
`P(W_b = +1) = σ(W_r) · 2 − 1`, sampled per step; lower variance but harder to implement.

### Batch norm trick
After binarization, BN constants α, β fold into a single shift + sign threshold τ:
`a_b = sign(a − τ)` with τ = β·σ/γ − μ — pure integer at inference.

## Connections
- [[straight-through-estimator]] — the gradient backbone BNN depends on.
- [[xnor-net]] — adds scale factor α for both weights and activations.
- [[dorefa-net]] — extends to arbitrary k-bit weights / activations / gradients.
- [[bitnet]] — modern LLM-era reincarnation of binary weights.
- [[bitnet-b158]] — ternary extension {-1, 0, +1}.
