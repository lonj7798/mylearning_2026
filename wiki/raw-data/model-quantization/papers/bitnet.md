<!-- scope: BitNet — 1-bit transformer trained from scratch; BitLinear layer with sign(W) + INT8 activations
     deps: [[straight-through-estimator]], [[bnn]], [[xnor-net]]
     see-also: [[bitnet-b158]], [[bitnet-a48]], [[onebit]], [[era-of-1bit-llms]]
-->

# BitNet: Scaling 1-bit Transformers for Large Language Models
- **Core Insight:** A transformer can be trained from scratch with 1-bit (sign-quantized) weights and INT8 activations — the same scaling-law slope as FP16 — by replacing every `nn.Linear` with a *BitLinear* layer that holds latent FP weights, binarises them via `sign(W − mean(W))` in the forward pass, uses a straight-through estimator on the backward, and adds a SubLN before the binary matmul to stabilise the now-discretised activations.
- **Guideline:** When pretraining a 1-bit transformer, swap every Linear for BitLinear (sign-quant W + INT8 absmax X + pre-matmul LayerNorm), use β1=0.9 β2=0.95 AdamW, lr 5× a comparable FP16 model's lr, and the same training data — the loss curve will track FP16's slope but with a constant offset that shrinks with scale.
- **Authors:** Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, Furu Wei
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2310.11453
- **Relevant topics:** 1-bit LLM pretraining, BitLinear, sign-quantization, INT8 activations, straight-through estimator, SubLN

## Abstract
BitNet is the first scalable 1-bit transformer architecture trained from scratch. The architectural change is to replace every linear projection (qkv, output, FFN-up, FFN-down) with a **BitLinear** layer: weights are stored in FP for the optimizer but binarised to ±1 via the sign function (after centering by mean) in the forward pass; activations are quantized per-token to INT8 absmax. A SubLN (LayerNorm before the binary matmul) absorbs the distribution shift from the binary multiplication. Backward uses STE through both quantizers. Trained from scratch up to 13B parameters on standard pretraining data, BitNet exhibits a scaling law parallel to FP16 (same slope, modest offset), and on inference enables an integer-only kernel with ~10× energy savings vs FP16.

## Key Contributions
- **BitLinear layer** — the drop-in 1-bit linear with explicit (binarise weights, absmax-INT8 activations, SubLN, STE) machinery.
- First scaling-law study of 1-bit LLM pretraining; demonstrates the slope tracks FP16.
- Provides the architectural template for the entire BitNet line (b1.58, a4.8).
- Establishes inference energy and memory savings (10× and 8× respectively vs FP16 on custom ASIC; ~4× and ~16× achievable on existing INT8 hardware).

## Key Figures/Tables to Study
- **Figure 1:** BitLinear data flow — latent FP weight → sign-quant → matmul with INT8 X → output rescale.
- **Figure 4 (scaling law):** BitNet vs FP16 LLM perplexity vs compute — parallel curves.
- **Table 3:** zero-shot accuracy at 1.3B / 3B / 13B vs FP16 — BitNet trails by a fixed gap that shrinks with scale.

## Technical Details

### Weight binarisation
For each BitLinear with latent weight `W ∈ R^{d_out × d_in}`:
```
α = (1 / (d_out · d_in)) · Σ_{i,j} W_{i,j}     # scalar mean
W̃ = Sign(W − α)                                  # ±1 binarisation
β = (1 / (d_out · d_in)) · ||W − α||_1            # absmean scale (re-magnitude)
```
The forward uses `β · W̃` (signs scaled back up). The mean-centering removes any DC component that would otherwise be wasted by the sign function.

### Activation quantization
Per-token absmax INT8:
```
γ = max|x| (per token)
x̃ = round(clip(x · 127/γ, −128, 127))            # INT8
```
Asymmetric variant subtracts a per-token min and uses unsigned uint8 — used for post-ReLU/post-GELU activations.

### SubLN (the stability trick)
Insert a LayerNorm immediately before the binary matmul:
```
y = β · ( SubLN(x̃) · W̃^⊤ ) · γ / 127
```
SubLN normalises the activations to roughly unit variance before they encounter the {±1} weight, keeping the dot-product magnitudes well-controlled across depth.

### Backward (STE)
Gradients of `Sign(·)` and the INT8 round are passed through as identity:
```
∂L/∂W = ∂L/∂W̃  (through Sign-STE)
∂L/∂x = ∂L/∂x̃  (through round-STE)
```
Optimizer updates the *latent* FP weight W; the binarisation is recomputed each forward.

### Architecture and training
- Same backbone as FP16 baseline (GPT-style decoder, RoPE, SwiGLU).
- Lr ~5× higher than FP16 (Sign(·) makes the loss surface "flatter" — bigger steps help).
- Adam β1=0.9, β2=0.95.
- All standard MLM/CLM tricks unchanged.

### Inference
At inference the FP weight is permanently replaced by its binarised `W̃` plus the per-tensor scale β. The matmul becomes:
```
y = β · (W̃ ⊙ x_int8)  =  β · popcount/XNOR-style accumulate
```
On INT8 tensor cores: still a regular INT8 GEMM with W in {−128, 127} → effectively 1-bit with constant scale. On dedicated 1-bit ASIC: XNOR + popcount as in BNN/XNOR-Net.

### Hyperparameters
| Knob | Value |
|------|-------|
| Weight bits | 1 (sign) + per-tensor β |
| Activation bits | 8 (per-token absmax) |
| Optimizer | AdamW, β1=0.9, β2=0.95 |
| LR | 5× FP16 baseline |
| Schedule | linear warmup + cosine decay |
| Stability layer | SubLN before each BitLinear |
| Backward | STE through Sign and round |

## Connections
- Pre-LLM ancestors: [[bnn]] (binary nets), [[xnor-net]], [[dorefa-net]].
- STE foundation: [[straight-through-estimator]].
- Ternary successor (the big one): [[bitnet-b158]].
- Activation-precision extension: [[bitnet-a48]].
- Survey-style follow-up: [[era-of-1bit-llms]].
- Alternative 1-bit weight via SVID: [[onebit]].
- Format reference: [[bitnet-w158]] (ternary).
- Lab summary: [[microsoft-bitnet]].
