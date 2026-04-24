---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "Ba, Kiros, Hinton — LayerNorm (2016); Zhang & Sennrich — RMSNorm (2019); Xiong et al. — On Layer Normalization in the Transformer Architecture (2020)"
source_url: https://arxiv.org/abs/1607.06450
created_at: "2026-04-23"
---

# Excerpt: Normalization — why reductions stay in fp32

**Papers:**
- *Layer Normalization* — Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton. 2016. arXiv: [1607.06450](https://arxiv.org/abs/1607.06450)
- *Root Mean Square Layer Normalization* — Biao Zhang, Rico Sennrich. NeurIPS 2019. arXiv: [1910.07467](https://arxiv.org/abs/1910.07467)
- *On Layer Normalization in the Transformer Architecture* — Ruibin Xiong et al. ICML 2020. arXiv: [2002.04745](https://arxiv.org/abs/2002.04745)
- (Contrast) *Batch Normalization* — Ioffe & Szegedy. ICML 2015. arXiv: [1502.03167](https://arxiv.org/abs/1502.03167)

This excerpt is the **numerical-precision** view of normalization: why the `mean(x²)` reduction must happen in fp32 even under bf16 compute, why pre-norm survives depth, and what the norm layer's `eps` is actually protecting against.

---

## 1. The operations (LayerNorm → RMSNorm)

### LayerNorm (Ba, Kiros, Hinton 2016, §3)

For a feature vector `x ∈ ℝ^d` (one token):

```math
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
y_i = \gamma_i \hat{x}_i + \beta_i
```

Two reductions (`mean`, `var`), one element-wise subtract, one element-wise divide, and a learnable affine `(γ, β)`. The paper's Figure 1 shows that unlike BatchNorm, LayerNorm is independent of batch composition — essential for variable-length sequences and RL rollouts.

### RMSNorm (Zhang & Sennrich 2019, §3)

> "We hypothesize that the re-centering invariance in LayerNorm is dispensable."

The simplification:

```math
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
y_i = \gamma_i \cdot \frac{x_i}{\text{RMS}(x)}
```

**Savings (Table 1 of the paper):** 1 fewer reduction (no `mean`), 1 fewer subtract, 1 fewer parameter set (no `β`). Reported end-to-end speedups of **7–64%** across six tasks, with no quality regression. For a modern LLM the norm op is called twice per block — in a 32-block model, that's 64 norm evaluations per forward. RMSNorm is the default in T5 and every Llama-family model.

---

## 2. Why the reduction must be fp32 under bf16 compute

For a typical LLM, `d = 4096` (or 8192 for large models). The reduction:

```math
\sum_{i=1}^{d} x_i^2
```

adds 4096 squared activations into a running scalar. Precision analysis:

| Quantity | Magnitude (post-residual, trained LLM) | Squared |
|---|---|---|
| `x_i` | `~1` to `~10` | `1` to `100` |
| Cumulative sum at `i = 4096` | `~d × E[x²] ≈ 4000` | — |
| Expected `mean(x²)` | `~1` | — |

bf16 has 7 mantissa bits ⇒ relative precision ≈ `2^-7 ≈ 0.78%`. When the running sum reaches `~1000` and you add a new increment of `~1`, the increment is `0.1%` of the running sum — near the representable precision floor. Over 4096 additions the bias compounds.

**Concrete failure mode.** The paper's own accuracy numbers assume fp32 reduction. A bf16-reduction implementation of RMSNorm biases the computed `RMS(x)` low by several percent, which scales the normalized output high, which feeds the residual stream with slightly-too-large magnitudes. Over 24 pre-norm layers this compounds — the final layer sees activation magnitudes that diverge from the design spec, and the LM head's logits get oversharp or dull. The symptom is **~0.5 PPL worse than expected** with no crash, no NaN, no spike.

**The universal pattern** (reproduced from the [[ch-02]] read):

```python
class RMSNorm(nn.Module):
    def forward(self, x):                          # x: bf16 (or fp8)
        in_dtype = x.dtype
        x_fp32 = x.float()                         # promote for reduction
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(in_dtype) * self.weight
```

The `x.float()` cast is non-negotiable. PyTorch, JAX, and Transformer Engine all promote inside `nn.LayerNorm` and `RMSNorm` by default — **do not override it**. Every framework's "use fp32 reductions" flag inside the norm is defaulted to `True` for exactly this reason.

---

## 3. `eps` — what it actually protects against

```math
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
```

The `eps` exists to prevent division by zero when `σ² = 0` — i.e. a constant feature vector. In practice for a trained LLM this never happens mid-training. But at initialization, a layer can produce near-constant activations, and without `eps` the first backward pass is NaN.

**Values seen in the wild:**

- LayerNorm default: `eps = 1e-5` (Llama)
- T5 RMSNorm: `eps = 1e-6`
- BERT / GPT-2: `eps = 1e-12` (too small for fp16, but these papers were fp32)

**Precision trap.** Under fp16, `eps = 1e-12` underflows (fp16 subnormal floor is ~`5.96e-8`). The `√(σ² + eps)` becomes `√(σ²)` = 0 when `σ = 0`. Bump to `1e-5` or (preferred) do the reduction and `eps` add in fp32 regardless of compute dtype.

Under bf16, `1e-12` rounds to the nearest representable bf16 value (bf16 subnormal floor is fp32-class: `~1.18e-38`). It doesn't underflow, but 7 mantissa bits mean `σ² + 1e-12` is indistinguishable from `σ²` unless `σ²` is similarly small — i.e. `eps` is effectively a no-op at the scales this trained LLM operates at. That's fine. In practice `eps = 1e-5` is the universal conservative default.

---

## 4. Pre-norm vs post-norm: the depth-stability story

From Xiong et al. 2020, the two placements:

```python
# Post-norm (Vaswani 2017 original)
x = LN(x + Sublayer(x))

# Pre-norm (GPT-2+, every modern LLM)
x = x + Sublayer(LN(x))
```

Figures 2–3 of Xiong 2020 plot the gradient norm at initialization against depth. Post-norm's `‖∇‖` grows exponentially with depth; pre-norm's stays `O(1)`. The derivation (§3 of the paper) is Jacobian-chain algebra:

```math
\text{Post-norm:} \quad \|\nabla_{\ell-1}\| \le \|\nabla_\ell\| \cdot (1 + \|J_{\text{sub}}\|) \cdot \|J_{\text{LN}}\|
\text{Pre-norm:} \quad \|\nabla_{\ell-1}\| \le \|\nabla_\ell\| \cdot (1 + \|J_{\text{sub}}\| \cdot \|J_{\text{LN}}\|)
```

The `+ 1` in pre-norm is the identity residual — gradients always have a depth-independent path back through the residual stream. In post-norm, every layer's gradient goes through `LN` (whose Jacobian's operator norm can exceed 1), so the product diverges.

**Precision consequence.** Post-norm transformers of depth ≥ 24 require warmup-LR schedules tuned to 3-significant-figure precision or training diverges around step 1000. Pre-norm transformers tolerate imprecise LR schedules — and tolerate bf16-vs-fp32 reduction errors in the norm, because the residual path provides gradient bypass. This is one quiet reason bf16 training "just works" in 2025 for pre-norm architectures.

---

## 5. Modern variants (2024–2025 frontier)

### QK-Norm

```math
Q' = \text{LayerNorm}(Q)
K' = \text{LayerNorm}(K)
\text{Attn}(Q, K, V) = \text{softmax}(Q' K'^{\top} / \sqrt{d_k}) V
```

Used in ViT-22B, OLMo-2, Qwen-2.5. Purpose: the softmax logits `Q K^T / √d` can explode in long-sequence training when Q or K acquires outlier rows. Re-normalizing each row to unit magnitude bounds the logits to `‖Q'‖ · ‖K'‖ / √d = 1 / √d` (they're unit norm).

**Precision angle.** QK-Norm *replaces* one source of instability (logit explosion) with another (two more fp32 reductions per attention head per layer). The reductions are cheap (`d_head ≈ 64–128`) and mandatory in fp32. OLMo-2's public training logs show QK-Norm eliminating a specific class of mid-training loss spikes that remained under standard pre-norm.

### Reordered-Norm (OLMo-2)

Place the second norm *after* the residual addition:

```python
# OLMo-2 reordered-norm
x = x + Sublayer(LN(x))       # normal pre-norm attention
x = LN(x + MLP(LN(x)))        # post-residual norm for MLP output
```

Eliminates a separate class of loss spikes. The reasoning: the pre-norm residual stream can grow in magnitude with depth; applying a norm after the residual add bounds the magnitude before it feeds the next block. The extra norm is another fp32 reduction — routine.

### Sandwich-Norm, DeepNorm

Niche. Sandwich-Norm adds a norm both before and after each sub-layer; DeepNorm combines init scaling with post-norm for 1000-layer stability. Neither is used in 2025 frontier LLMs.

---

## 6. The LN `γ` / `β` parameters and weight decay

From the Ba et al. 2016 formulation, LayerNorm has **two** learnable parameter groups per layer: `γ` (scale) and `β` (shift). RMSNorm has only `γ` (no `β`). Standard practice:

- `γ` initialized to `1.0`.
- `β` initialized to `0.0` (LayerNorm only).
- Both **excluded from weight decay** (see [[excerpts/adam]] §6). Decaying `γ` toward 0 squashes the layer's output; decaying `β` toward 0 is harmless but unnecessary.

**Precision requirement.** `γ` is multiplied back onto the normalized `x̂` in *compute dtype* (bf16). This is fine because `γ ≈ 1` and the output magnitude is controlled by the normalization, not by `γ`.

---

## 7. Why BatchNorm is wrong for LLMs

BatchNorm normalizes along the batch dimension:

```math
\mu_c = \frac{1}{B \cdot T} \sum_{b, t} x_{b, t, c}
\sigma_c^2 = \frac{1}{B \cdot T} \sum_{b, t} (x_{b, t, c} - \mu_c)^2
```

which couples examples in a batch. For autoregressive LMs with variable-length sequences and causal masking, this coupling breaks in two ways:

1. **Inference mismatch.** BN uses running statistics at inference; LN uses per-example statistics. LMs generate one token at a time at inference — running stats would have to be updated sequentially, which is expensive and introduces train/inference skew.
2. **RL / rollout incompatibility.** PPO / GRPO reuse rollout batches with different effective batch compositions. BN's running stats would drift with every rollout phase.

LayerNorm / RMSNorm sidestep both by being **per-example** statistics. This is why LLM training in 2025 uses no BN anywhere — not in embeddings, not in attention, not in MLPs.

---

## 8. The "never override fp32 reductions" rule, restated

From the read's §3:

> "The `mean(x²)` over 4096-dim vectors accumulates errors in bf16 that bias normalisation. Frameworks default to fp32 reductions here; never override it."

The flag to watch:

- PyTorch: `nn.LayerNorm` / `nn.functional.layer_norm` — internal fp32 reduction enabled by default.
- Transformer Engine: `te.LayerNorm` / `te.RMSNorm` — `sm_margin` and `zero_centered_gamma` flags do not disable fp32 reduction.
- JAX / Flax: `nn.LayerNorm` has a `dtype` parameter; set to `jnp.float32` or leave default.
- DeepSpeed: MoE norms need the same fp32 rule; `DeepSpeedTransformerConfig` defaults to fp32 norms.

Overriding any of these to bf16 is the single most common "silent quality regression" bug in a mixed-precision codebase.

---

## Connections

- [[ch-02]] — §3 "which ops must live in fp32" is anchored by this excerpt's §2.
- [[excerpts/mixed-precision]] — §3 "stability tricks (universal)" lists norm-reduction-in-fp32 as universal.
- [[excerpts/adam]] — LN `γ` is in the no-decay group; optimizer state for norm params is still fp32.
- [[excerpts/gradient-clipping]] — pre-norm relaxes the need for aggressive clipping; post-norm is unworkable without it.
- [[excerpts/deepseek-v3]] — 671B MoE uses RMSNorm + pre-norm; norm reductions stay in fp32 even under fp8 matmul.
