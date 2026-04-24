<!-- scope: RMSNorm vs LayerNorm comparison, parent: [[ch-09]] -->

# RMSNorm vs LayerNorm: The Essential Comparison

Why does dropping half of LayerNorm work? This excerpt pins down exactly what RMSNorm removes, what it keeps, why the removed pieces are dispensable, and where the computational savings come from.

---

## The Two Components of LayerNorm

LayerNorm ([[layer-norm|paper]]) performs two distinct operations on an input $x \in \mathbb{R}^d$:

1. **Mean-centering:** subtract $\mu = \frac{1}{d}\sum x_i$ to produce a zero-mean vector. This gives **re-centering invariance**: $f(x + c\mathbf{1}) = f(x)$.

2. **Variance normalization:** divide by $\sigma = \sqrt{\frac{1}{d}\sum(x_i - \mu)^2}$ to produce a unit-variance vector. This gives **re-scaling invariance**: $f(\alpha x) = f(x) \cdot \text{sign}(\alpha)$.

LayerNorm then applies learnable scale $\gamma$ and shift $\beta$: $\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$.

RMSNorm ([[rmsnorm|paper]]) keeps only variance normalization — reframed as RMS normalization — and drops mean-centering, the bias $\beta$, and the mean reduction entirely:

$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2}$$

---

## Why Dropping Mean-Centering Works

The central argument from Zhang & Sennrich (2019):

**Re-scaling invariance is sufficient for training stability.** The dominant failure mode in deep networks is activation *scale* explosion — magnitudes growing exponentially with depth. Normalization by RMS controls scale. Mean drift (a constant offset across all hidden dimensions) is not a significant source of instability in practice: Transformer activations don't develop large DC offsets during training.

A second subtlety: mean-centering projects out one dimension of information (the mean direction $\mathbf{1}/\sqrt{d}$). In a $d$-dimensional hidden space, losing one direction is negligible ($d$ is typically 4096+), but in principle the network cannot represent features that require a nonzero mean across the hidden dimension after LayerNorm. RMSNorm avoids this projection, preserving the full $d$-dimensional space.

**Implicit learning rate adaptation** comes from re-scaling invariance alone. When a weight matrix $W$ is scaled by $\alpha$, the gradient $\nabla_W \mathcal{L}$ is scaled by $1/\alpha$, producing an effective learning rate of $\eta / \|W\|$. This self-regulating property prevents any single layer from dominating gradient updates. RMSNorm inherits this entirely.

---

## Computational Savings: Where They Come From

Each normalization call on a $(B, T, d)$ tensor:

| Operation | LayerNorm | RMSNorm |
|-----------|-----------|---------|
| Reductions over $d$ | 2 (mean + variance) | 1 (sum of squares) |
| Elementwise ops | subtract mean, divide, scale, shift | divide, scale |
| Learnable params | $2d$ ($\gamma$, $\beta$) | $d$ ($\gamma$) |
| GPU sync barriers in fused kernel | 2 | 1 |

Each reduction across the hidden dimension $d$ requires a warp/block-level synchronization — a pipeline stall on the GPU. Eliminating one reduction per norm call, across 160 norm operations per forward pass in a 80-layer LLM (2 per block: before attention, before FFN), compounds significantly.

Zhang & Sennrich measured **7-64% wall-clock speedup** depending on model size:

- Small hidden dim ($d = 256$): ~7% (reduction cost is small fraction of total)
- Large hidden dim ($d = 1024$+): 30-64% (reduction dominates kernel time)
- Modern LLMs ($d = 4096$-$16384$): substantial, as the single eliminated reduction+subtraction would otherwise be the bottleneck in a fused kernel

### Fused Kernel Comparison

```python
# LayerNorm: 2 reductions, 4 elementwise ops
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)           # reduction 1
    x_centered = x - mean                          # elementwise
    var = (x_centered ** 2).mean(dim=-1, keepdim=True)  # reduction 2
    return gamma * x_centered / (var + eps).sqrt() + beta

# RMSNorm: 1 reduction, 2 elementwise ops
def rms_norm(x, gamma, eps=1e-5):
    rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt()  # reduction 1
    return gamma * x / (rms + eps)
```

---

## Empirical Evidence

Zhang & Sennrich tested across machine translation, language modeling, and multiple architectures. **No statistically significant quality difference** between LayerNorm and RMSNorm on any benchmark.

This held at scale too. Every major LLM since LLaMA (2023) uses RMSNorm:

| Model | Norm | Year |
|-------|------|------|
| LLaMA 1/2/3 | RMSNorm | 2023-24 |
| Mistral / Mixtral | RMSNorm | 2023-24 |
| PaLM / Gemini | RMSNorm | 2022-24 |
| GPT-NeoX | RMSNorm | 2022 |
| OLMo 2 | RMSNorm | 2025 |
| Gemma 3 | RMSNorm | 2025 |

No published ablation at any scale has shown a quality advantage for LayerNorm over RMSNorm.

---

## Partial RMSNorm (pRMSNorm)

Zhang & Sennrich also proposed estimating RMS from only $p\%$ of hidden dimensions:

$$\text{RMS}_p(x) = \sqrt{\frac{1}{|S|}\sum_{i \in S} x_i^2}$$

At $p = 25\%$, the reduction cost drops 4x with negligible quality impact. This variant hasn't been adopted in practice: the full reduction is already fast with modern warp-level primitives, and the marginal speedup doesn't justify the sampling variance.

---

## Summary

RMSNorm drops mean-centering and the bias term from LayerNorm. It keeps re-scaling invariance — the only property that matters for training stability. The result is simpler, faster (one fewer GPU sync barrier per call), and empirically equivalent. It is the universal default for modern LLM architectures.

---

**References:** [[rmsnorm|Zhang & Sennrich (2019)]], [[layer-norm|Ba, Kiros & Hinton (2016)]]
