---
chapter: ch-16
course: model-quantization
phase: read
excerpt_of: "BitNet: Scaling 1-bit Transformers for Large Language Models"
source_url: https://arxiv.org/abs/2310.11453
created_at: "2026-05-21"
---

# Excerpt: BitNet — the BitLinear layer for 1-bit transformer pretraining

**Authors:** Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, Furu Wei (Microsoft)
**Year:** 2023
**URL:** https://arxiv.org/abs/2310.11453
**Raw-data source:** [[raw-data/bitnet]]

---

## The thesis

A transformer can be trained from scratch with **1-bit (sign-quantized) weights and INT8 activations**, with the same scaling-law slope as FP16. The architectural change: replace every `nn.Linear` with a **BitLinear** layer that holds latent FP weights, binarises them via `sign(W − mean(W))` in the forward, uses an STE on the backward, and adds a SubLN before the binary matmul.

This is the first scalable 1-bit transformer; it establishes the recipe that BitNet b1.58 (ternary, ch-16 main) and a4.8 (4-bit acts) extend.

---

## The BitLinear layer

For a linear `y = W x` with `W ∈ ℝ^{d_out × d_in}`, BitLinear computes:

```python
class BitLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # latent FP weight — never discarded
        self.W = nn.Parameter(torch.randn(d_out, d_in) / d_in ** 0.5)
        self.subln = RMSNorm(d_in)

    def weight_quant(self, W):
        # 1) mean-centered binary {-1, +1}
        alpha = W.mean()
        W_centered = W - alpha
        sign_W = torch.sign(W_centered)
        # 2) absmean scale (re-magnitudes the signs)
        beta = W_centered.abs().mean()
        W_q = sign_W * beta
        # STE: forward returns W_q, backward gets ∂L/∂W (identity)
        return W + (W_q - W).detach()

    def act_quant(self, x):
        # per-token absmax INT8
        gamma = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_q = (x * 127 / gamma).round().clamp(-128, 127) * (gamma / 127)
        return x + (x_q - x).detach()

    def forward(self, x):
        x = self.subln(x)             # stability layer
        x_q = self.act_quant(x)       # INT8 activation
        W_q = self.weight_quant(self.W)  # ±β weight
        return F.linear(x_q, W_q)
```

---

## Weight binarisation — the math

```math
\begin{aligned}
\alpha &= \frac{1}{d_{\text{out}} \cdot d_{\text{in}}} \sum_{i,j} W_{i,j}      \quad \text{(scalar mean)} \\
\tilde{W} &= \mathrm{Sign}(W - \alpha)                                          \quad \text{(\pm 1 binarisation)} \\
\beta &= \frac{1}{d_{\text{out}} \cdot d_{\text{in}}} \|W - \alpha\|_1            \quad \text{(absmean scale)}
\end{aligned}
```

Forward uses `β · W̃`. Mean-centering removes any DC component that would otherwise be wasted by the sign function; the absmean β re-magnitudes the signs so the spectral norm of the layer stays comparable to the latent W.

---

## Activation quantization — per-token absmax INT8

```math
\gamma = \max_i |x_i| \quad \text{(per token)}
\quad\quad
\tilde{x} = \mathrm{round}\left(\mathrm{clip}\left(x \cdot \frac{127}{\gamma},\, -128,\, 127\right)\right)
```

Symmetric INT8. Asymmetric variant subtracts a per-token min and uses unsigned uint8 for post-ReLU / post-GELU activations.

---

## SubLN — the stability trick

Insert a LayerNorm (RMSNorm) **immediately before** the binary matmul:

```math
y = \beta \cdot (\mathrm{SubLN}(\tilde{x}) \cdot \tilde{W}^\top) \cdot \frac{\gamma}{127}
```

SubLN normalises activations to roughly unit variance before they encounter the {±1} weight, keeping dot-product magnitudes well-controlled across depth. Without SubLN, BitNet training is unstable past ~1B parameters — gradients explode after a few hundred steps.

---

## Backward — straight-through estimator

```math
\frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial \tilde{W}}
\quad\text{(through Sign-STE)}
\qquad
\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial \tilde{x}}
\quad\text{(through round-STE)}
```

The optimizer updates the latent FP weight W; the binarisation is recomputed each forward. The latent FP weight is kept in BF16/FP32 and **never discarded** — quantization happens only on the forward path.

This is the key difference between BitNet and PTQ: the training signal sees the quantization, so the latent weights drift to a configuration that's quantization-robust.

---

## Inference

At inference the FP weight is permanently replaced by its binarised `W̃` plus the per-tensor scale β. The matmul becomes:

```
y = β · (W̃ ⊙ x_int8) = β · popcount/XNOR-style accumulate
```

On INT8 tensor cores: regular INT8 GEMM with W in {−127, 127} → effectively 1-bit with constant scale. On dedicated 1-bit ASIC: XNOR + popcount as in [[bnn]] / [[xnor-net]].

---

## Architecture and training recipe

- Same backbone as FP16 baseline (Llama-style decoder, RoPE, SwiGLU).
- Lr ~**5× higher** than FP16 baseline (Sign(·) flattens the loss surface; bigger steps help).
- AdamW β₁=0.9, β₂=0.95.
- Linear warmup + cosine decay.
- All standard MLM/CLM tricks unchanged.
- Same data, same tokenizer as the FP16 scaling-law comparison.

---

## Scaling law (the headline)

BitNet vs FP16 LLM perplexity vs compute — **parallel curves**. Same slope, modest offset that shrinks with scale. The plot is BitNet's Figure 4.

| Params | FP16 ppl | BitNet ppl | Δ |
|--------|----------|------------|---|
| 700M | baseline | +1 to +2 | small gap |
| 1.3B | baseline | +0.5 to +1.5 | shrinking |
| 3B | baseline | +0.5 | shrunk |
| 13B (extrapolated in paper) | baseline | ~0 | parity |

BitNet b1.58 (ternary) closes the gap completely at 3B; pure binary BitNet trails by a small constant.

---

## Inference energy / memory savings

- Memory: **8× reduction** vs FP16 weight (1 bit vs 16 bits).
- Energy: **~10× reduction** vs FP16 on custom ASIC (no multiplier); ~4× on existing INT8 hardware.

---

## Hyperparameters

| Knob | Value |
|------|-------|
| Weight bits | 1 (sign) + per-tensor β |
| Activation bits | 8 (per-token absmax) |
| Optimizer | AdamW, β₁=0.9, β₂=0.95 |
| LR | 5× FP16 baseline |
| Schedule | linear warmup + cosine decay |
| Stability layer | SubLN before each BitLinear |
| Backward | STE through Sign and round |

---

## Pitfalls

- **SubLN must sit immediately before the binary matmul** — misplaced SubLN destabilises training.
- **The latent FP weight is permanent.** Cannot be discarded mid-training. After training, the ternary forward output is the final artefact.
- **`weight_quant` and `act_quant` both use the STE trick** `x + (x_q - x).detach()`. Easy to mistype; if `detach()` is missing, gradient flows through the quantizer and breaks.
- **Lr scaling matters.** Use 5× FP16 baseline LR; smaller LRs leave BitNet unable to find the binary-friendly weight configuration.
- **AdamW β₂ = 0.95** (not 0.999) — the standard LLM default; with binary weights the gradient distribution is non-stationary and 0.999 is too smooth.

---

## Connections

- [[excerpts/bitnet-b158]] — direct successor with ternary weights; matches FP16 at 3B.
- [[ch-04]] — STE foundation; [[bnn]] / [[xnor-net]] / [[dorefa-net]] as pre-LLM ancestors.
- [[ch-14]] — [[pv-tuning]] is the principled discrete-optimization replacement for STE on codebook methods (not used here since BitNet's discrete set is just {-1, +1}).
- [[bitnet-models]] — production releases that build on this recipe.
