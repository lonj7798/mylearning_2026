# Chapter 8: Feed-Forward Networks, Activations, and Width

<!-- scope: FFN sublayer as key-value memory, activation functions (ReLU/GELU/SwiGLU), GLU variants, width-depth tradeoffs, hidden dimension ratios
     deps: [[ch-03]]
     see-also: [[ch-14]], [[ch-09]]
-->

## Overview

The feed-forward network (FFN) is the quiet workhorse of the Transformer — roughly two-thirds of total model parameters, yet a fraction of the research attention that self-attention receives. Each layer applies a two-layer MLP independently to every token position: project up, activate, project down. This chapter explains what that network actually does, why it may function as a key-value memory storing factual knowledge, and how the activation function inside it evolved from ReLU to the gated SwiGLU variant now used by virtually every frontier model.

The shift to gated activations is not cosmetic. Shazeer (2020) ([[glu-variants|paper]]) showed GLU variants consistently outperform standard activations, and the consequences ripple through the whole design: a third weight matrix, a hidden dimension that shrinks from $4d$ to $\frac{8}{3}d$, and a multiplicative gating branch. We also address the width-vs-depth question — given a fixed parameter budget, should you widen the FFN (more memory per layer) or deepen the network (more composition)? Real configs from Llama 3 ([[llama-3|report]]), Mistral ([[mistral-7b|report]]), and gpt-oss show practitioners experimenting with both ends.

---

## 1. The Position-Wise FFN: Original Design

The original Transformer (Vaswani et al. 2017, [[attention-is-all-you-need|paper]]) defines the FFN sublayer as a two-layer MLP applied independently at each token position:

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

where $W_1 \in \mathbb{R}^{d_{ff} \times d_{model}}$, $W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$, and $\sigma$ is an activation function (ReLU in the original paper). The original config used $d_{model} = 512$ and $d_{ff} = 2048$ — a 4:1 ratio.

**Parameter count for the standard FFN:** Two matrices gives $2 \times d_{model} \times d_{ff}$ parameters. With the 4x ratio: $2 \times d \times 4d = 8d^2$ parameters per FFN sublayer.

For comparison, the multi-head self-attention sublayer uses $4d^2$ parameters (four projection matrices $W^Q, W^K, W^V, W^O$, each $d \times d$). So the FFN has **twice the parameters** of self-attention in each layer — it is the majority of the model.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0; font-family:monospace;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Standard FFN Computation (per token position)</div>
<div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; justify-content:center;">
<div style="background:#16213e; padding:12px 16px; border-radius:8px; text-align:center; min-width:80px;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">x</div>
<div style="color:#888; font-size:10px;">d_model</div>
</div>
<div style="color:#666; font-size:18px;">→</div>
<div style="background:#0f3460; padding:12px 16px; border-radius:8px; text-align:center; min-width:100px;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">W₁x + b₁</div>
<div style="color:#888; font-size:10px;">project up → d_ff</div>
</div>
<div style="color:#666; font-size:18px;">→</div>
<div style="background:#1a1a2e; padding:12px 16px; border-radius:8px; border:2px solid #e94560; text-align:center; min-width:80px;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">σ( · )</div>
<div style="color:#888; font-size:10px;">activation</div>
</div>
<div style="color:#666; font-size:18px;">→</div>
<div style="background:#0f3460; padding:12px 16px; border-radius:8px; text-align:center; min-width:100px;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">W₂ · h</div>
<div style="color:#888; font-size:10px;">project down → d_model</div>
</div>
<div style="color:#666; font-size:18px;">→</div>
<div style="background:#16213e; padding:12px 16px; border-radius:8px; text-align:center; min-width:80px;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">out</div>
<div style="color:#888; font-size:10px;">d_model</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:14px; text-align:center; font-family:sans-serif;">
Up-projection to d_ff = 4 × d_model, activation, down-projection back to d_model.<br/>
Two matrices: W₁ ∈ ℝ^(d_ff × d_model), W₂ ∈ ℝ^(d_model × d_ff). Total: 8d² params.
</div>
</div>

**Why position-wise?** The FFN processes each token independently — there is no cross-position communication in this sublayer. All cross-position interaction happens in the attention sublayer. This division of labor (attention = routing information between positions; FFN = processing information at each position) is a design choice, not a necessity, and it has implications for the "FFN as memory" hypothesis discussed next.

---

## 2. The FFN-as-Memory Hypothesis

Geva et al. (2021) proposed a provocative interpretation: the FFN sublayer functions as a **key-value memory**, where the rows of $W_1$ are keys and the columns of $W_2$ are values. When an input $x$ is projected through $W_1$, the resulting activations determine *which* rows activate (matching keys). The corresponding columns of $W_2$ then contribute to the output (retrieved values). Under this framing:

- $W_1 x$ computes a similarity score between the input and each of the $d_{ff}$ "key" patterns
- $\sigma(\cdot)$ thresholds these similarities (ReLU zeros out non-matching keys)
- $W_2$ maps the activated pattern back to the residual stream, effectively "writing" the recalled information

This interpretation explains several empirical observations:

[Deep Dive: The FFN-as-Memory Hypothesis](excerpts/ffn-as-memory.md) — evidence for and against, from Geva et al.'s key-value memory interpretation through ROME editing to the superposition complication.

**Factual knowledge localization.** Meng et al. (2022) demonstrated that specific factual associations (e.g., "The Eiffel Tower is in [Paris]") can be traced to individual FFN layers and even specific neurons. Their ROME (Rank-One Model Editing) technique edits knowledge by modifying a single rank-one update to a specific $W_2$ matrix — and the edit sticks, changing the model's factual output without retraining.

**Sparse activation patterns.** In practice, ReLU-based FFNs activate only a small fraction of neurons for any given input (often <10%). This sparsity is consistent with the memory interpretation: most "keys" don't match, so most "values" aren't retrieved. It also motivates Mixture-of-Experts architectures ([[ch-14]]), which make this sparsity structural rather than emergent.

**Why this matters for architecture research:** If FFNs are memories, then $d_{ff}$ controls memory capacity — the number of key-value pairs the model can store per layer. This reframes the width question: wider FFNs don't just add "compute," they add *storage slots* for factual knowledge. The 4x ratio is not an arbitrary magic number; it's a bet about how much memory each layer needs relative to its representation dimension.

---

## 3. Activation Functions: ReLU to GELU to SwiGLU

The activation function inside the FFN has evolved through three major generations:

### ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

The original Transformer used ReLU. Its key property is **hard sparsity**: exactly zero for all negative inputs. This aligns well with the memory interpretation (most keys "off"), but has two well-known issues:

- **Dead neurons:** Once a neuron's pre-activation is consistently negative, it never activates and receives zero gradient. It's permanently dead.
- **Non-smooth gradient:** The gradient is discontinuous at $x = 0$, which can create optimization difficulties.

### GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)$$

where $\Phi$ is the standard Gaussian CDF. GELU was adopted by GPT-2 and BERT. It provides **soft gating**: instead of a hard cutoff at zero, inputs are smoothly weighted by their probability of being positive under a Gaussian assumption. Small negative inputs get small but nonzero activation, avoiding the dead neuron problem. The smooth gradient everywhere aids optimization.

### Swish / SiLU (Sigmoid Linear Unit)

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

where $\sigma$ is the sigmoid function and $\beta$ is typically 1. Closely related to GELU; discovered via neural architecture search (Ramachandran et al. 2017). Swish is the activation used *inside* the SwiGLU gate.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Activation Function Comparison</div>
<div style="display:flex; gap:16px; flex-wrap:wrap; justify-content:center;">
<div style="background:#16213e; padding:16px; border-radius:10px; min-width:180px; flex:1;">
<div style="color:#e94560; font-weight:bold; font-size:14px; margin-bottom:8px;">ReLU</div>
<div style="color:#aaa; font-size:12px; font-family:monospace;">max(0, x)</div>
<div style="color:#888; font-size:11px; margin-top:8px;">
<span style="color:#4ecca3;">+</span> Hard sparsity (memory-like)<br/>
<span style="color:#4ecca3;">+</span> Trivially fast<br/>
<span style="color:#e94560;">-</span> Dead neurons<br/>
<span style="color:#e94560;">-</span> Discontinuous gradient
</div>
<div style="color:#555; font-size:10px; margin-top:8px;">Used by: Original Transformer</div>
</div>
<div style="background:#16213e; padding:16px; border-radius:10px; min-width:180px; flex:1;">
<div style="color:#e94560; font-weight:bold; font-size:14px; margin-bottom:8px;">GELU</div>
<div style="color:#aaa; font-size:12px; font-family:monospace;">x · Φ(x)</div>
<div style="color:#888; font-size:11px; margin-top:8px;">
<span style="color:#4ecca3;">+</span> Smooth everywhere<br/>
<span style="color:#4ecca3;">+</span> No dead neurons<br/>
<span style="color:#e94560;">-</span> Less sparse<br/>
<span style="color:#e94560;">-</span> Slightly slower than ReLU
</div>
<div style="color:#555; font-size:10px; margin-top:8px;">Used by: GPT-2, BERT, GPT-3</div>
</div>
<div style="background:#16213e; padding:16px; border-radius:10px; min-width:180px; flex:1; border:2px solid #e94560;">
<div style="color:#e94560; font-weight:bold; font-size:14px; margin-bottom:8px;">SwiGLU ★</div>
<div style="color:#aaa; font-size:12px; font-family:monospace;">(Swish(W₁x) ⊙ Vx)W₂</div>
<div style="color:#888; font-size:11px; margin-top:8px;">
<span style="color:#4ecca3;">+</span> Best quality (empirical)<br/>
<span style="color:#4ecca3;">+</span> Learnable gating<br/>
<span style="color:#e94560;">-</span> 3 matrices (not 2)<br/>
<span style="color:#e94560;">-</span> 50% more matmuls per FFN
</div>
<div style="color:#555; font-size:10px; margin-top:8px;">Used by: Llama 3, Mistral, PaLM, Gemma</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:14px; text-align:center; font-family:sans-serif;">
The field has converged on SwiGLU. Raschka's 2025 comparison ([[raschka-llm-architecture-comparison|blog]]) confirms every major model uses it.
</div>
</div>

[Interactive: Activation Function Comparison](figures/activation-comparison.html) — plot ReLU, GELU, Swish, and their gated variants with adjustable parameters; toggle between activation, derivative, and gated output views.

Both GELU and Swish have a **non-monotonic region** near zero — small negative inputs get slightly negative outputs before being suppressed, allowing the network to propagate weak signals rather than killing them. Whether this property explains the empirical gains is debatable, but the gains are consistent enough that the field has moved on.

---

## 4. GLU Variants: The Gating Revolution

### The Core Idea

The Gated Linear Unit (Dauphin et al. 2017) introduced a multiplicative gating mechanism into the FFN:

$$\text{GLU}(x) = (W_1 x) \odot \sigma(V x)$$

where $\odot$ is element-wise multiplication. Instead of one linear projection followed by an activation, you compute **two parallel projections** and multiply them element-wise. One branch (the "gate") passes through a sigmoid to produce values in $[0, 1]$; the other branch (the "value") provides the content. The gate controls how much of the value passes through.

### Shazeer's GLU Variants (2020)

Shazeer's ([[glu-variants|paper]]) key contribution was systematically replacing the sigmoid in GLU with other activation functions and testing them all on Transformer language modeling:

| Variant | Gate Function | Formula |
|---------|--------------|---------|
| **GLU** | $\sigma(Vx)$ | $(W_1 x) \odot \sigma(Vx)$ |
| **ReGLU** | $\text{ReLU}(Vx)$ | $(W_1 x) \odot \text{ReLU}(Vx)$ |
| **GEGLU** | $\text{GELU}(Vx)$ | $(W_1 x) \odot \text{GELU}(Vx)$ |
| **SwiGLU** | $\text{Swish}(Vx)$ | $(W_1 x) \odot \text{Swish}(Vx)$ |

The full gated FFN then becomes:

$$\text{FFN}_{\text{SwiGLU}}(x) = \left(\text{Swish}(W_1 x) \odot V x\right) W_2$$

Note: modern implementations typically put the activation on the $W_1$ branch and leave $V$ as the linear (ungated) branch, but the roles can be swapped since multiplication is commutative.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0; font-family:monospace;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Gated FFN (SwiGLU) — Three-Matrix Architecture</div>
<div style="display:flex; flex-direction:column; gap:16px; align-items:center;">

<div style="display:flex; align-items:center; gap:12px;">
<div style="background:#16213e; padding:12px 16px; border-radius:8px; text-align:center; width:80px;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">x</div>
<div style="color:#888; font-size:10px;">d_model</div>
</div>
<div style="color:#666; font-size:14px; width:20px; text-align:center;">→</div>
<div style="display:flex; flex-direction:column; gap:8px;">
<div style="display:flex; align-items:center; gap:8px;">
<div style="background:#0f3460; padding:8px 14px; border-radius:8px; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:12px;">W₁x</div>
<div style="color:#888; font-size:9px;">gate branch</div>
</div>
<div style="color:#666; font-size:14px;">→</div>
<div style="background:#1a1a2e; padding:8px 14px; border-radius:8px; border:2px solid #e94560; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:12px;">Swish(·)</div>
</div>
</div>
<div style="display:flex; align-items:center; gap:8px;">
<div style="background:#0f3460; padding:8px 14px; border-radius:8px; text-align:center;">
<div style="color:#4ecca3; font-weight:bold; font-size:12px;">Vx</div>
<div style="color:#888; font-size:9px;">value branch</div>
</div>
<div style="color:#666; font-size:14px;">→</div>
<div style="background:#16213e; padding:8px 14px; border-radius:8px; text-align:center;">
<div style="color:#888; font-size:12px;">(linear)</div>
</div>
</div>
</div>
<div style="color:#666; font-size:14px; width:20px; text-align:center;">→</div>
<div style="background:#1a1a2e; padding:12px 16px; border-radius:8px; border:2px dashed #e94560; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">⊙</div>
<div style="color:#888; font-size:9px;">element-wise</div>
</div>
<div style="color:#666; font-size:14px; width:20px; text-align:center;">→</div>
<div style="background:#0f3460; padding:12px 16px; border-radius:8px; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">W₂ · h</div>
<div style="color:#888; font-size:10px;">project down</div>
</div>
<div style="color:#666; font-size:14px; width:20px; text-align:center;">→</div>
<div style="background:#16213e; padding:12px 16px; border-radius:8px; text-align:center; width:80px;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">out</div>
<div style="color:#888; font-size:10px;">d_model</div>
</div>
</div>

</div>
<div style="color:#888; font-size:11px; margin-top:14px; text-align:center; font-family:sans-serif;">
Three matrices: W₁ (gate), V (value), W₂ (down-projection). The gate branch controls information flow from the value branch.
</div>
</div>

### Why Gating Works

Shazeer offers no definitive theoretical explanation — the improvements are empirically observed. Three plausible accounts:

1. **Learned feature selection.** The gate learns *which* features matter; the value branch computes *what* to output — more expressive than a fixed activation threshold.
2. **Richer gradient paths.** The element-wise product creates two gradient paths per neuron, providing more diverse optimization signal.
3. **Multiplicative interactions.** The $\odot$ operation creates second-order interactions between two different linear projections — strictly more expressive than a nonlinearity applied to a single projection.

The key empirical finding: **the gating mechanism matters more than the specific activation function.** ReGLU, GEGLU, and SwiGLU all outperformed their non-gated counterparts. SwiGLU and GEGLU were close, with SwiGLU slightly preferred.

---

## 5. The 8/3 Ratio: Matching Parameter Budgets

[Deep Dive: SwiGLU Parameter Cost Derivation](excerpts/swiglu-parameter-cost.md) — full derivation of the iso-parameter ratio, FLOP analysis, and why real models exceed 8/3.

[Interactive: FFN Hidden Dimension Ratio Explorer](figures/ffn-dimension-explorer.html) — adjust d_model, ratio, and activation type to see parameter counts, FLOP breakdowns, and comparisons with real model configurations.

The gated FFN introduces a third weight matrix ($V$), which increases parameters. To compare fairly, you must adjust $d_{ff}$ to match the parameter budget of the standard FFN.

**Standard FFN (2 matrices):**
$$\text{Params} = 2 \times d_{model} \times d_{ff} = 2 \times d \times 4d = 8d^2$$

**Gated FFN (3 matrices):**
$$\text{Params} = 3 \times d_{model} \times d_{ff}' = 3 \times d \times d_{ff}'$$

Setting these equal: $3d \cdot d_{ff}' = 8d^2$, which gives:

$$d_{ff}' = \frac{8}{3}d_{model} \approx 2.667 \times d_{model}$$

In practice, this is rounded to a convenient multiple (often a multiple of 128 or 256 for hardware efficiency). Let's verify with real models:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">FFN Dimension Ratios in Production Models</div>
<table style="width:100%; border-collapse:collapse; font-size:13px; color:#e0e0e0;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px 12px; color:#e94560;">Model</th>
<th style="text-align:right; padding:8px 12px; color:#e94560;">d_model</th>
<th style="text-align:right; padding:8px 12px; color:#e94560;">d_ff</th>
<th style="text-align:right; padding:8px 12px; color:#e94560;">Ratio</th>
<th style="text-align:left; padding:8px 12px; color:#e94560;">Activation</th>
<th style="text-align:left; padding:8px 12px; color:#e94560;">Note</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px;">Transformer (orig.)</td>
<td style="padding:8px 12px; text-align:right;">512</td>
<td style="padding:8px 12px; text-align:right;">2,048</td>
<td style="padding:8px 12px; text-align:right; color:#4ecca3;">4.00x</td>
<td style="padding:8px 12px;">ReLU</td>
<td style="padding:8px 12px; color:#888;">2 matrices</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px;">Llama 3 8B ([[llama-3|report]])</td>
<td style="padding:8px 12px; text-align:right;">4,096</td>
<td style="padding:8px 12px; text-align:right;">14,336</td>
<td style="padding:8px 12px; text-align:right; color:#4ecca3;">3.50x</td>
<td style="padding:8px 12px;">SwiGLU</td>
<td style="padding:8px 12px; color:#888;">3 matrices, ~8/3 rounded up</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px;">Llama 3 70B</td>
<td style="padding:8px 12px; text-align:right;">8,192</td>
<td style="padding:8px 12px; text-align:right;">28,672</td>
<td style="padding:8px 12px; text-align:right; color:#4ecca3;">3.50x</td>
<td style="padding:8px 12px;">SwiGLU</td>
<td style="padding:8px 12px; color:#888;">3 matrices</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px;">Llama 3 405B</td>
<td style="padding:8px 12px; text-align:right;">16,384</td>
<td style="padding:8px 12px; text-align:right;">53,248</td>
<td style="padding:8px 12px; text-align:right; color:#4ecca3;">3.25x</td>
<td style="padding:8px 12px;">SwiGLU</td>
<td style="padding:8px 12px; color:#888;">slightly under 8/3 budget</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px;">Mistral 7B ([[mistral-7b|report]])</td>
<td style="padding:8px 12px; text-align:right;">4,096</td>
<td style="padding:8px 12px; text-align:right;">14,336</td>
<td style="padding:8px 12px; text-align:right; color:#4ecca3;">3.50x</td>
<td style="padding:8px 12px;">SwiGLU</td>
<td style="padding:8px 12px; color:#888;">identical to Llama 3 8B</td>
</tr>
<tr>
<td style="padding:8px 12px;">DeepSeek-V3 ([[deepseek-v3|report]])</td>
<td style="padding:8px 12px; text-align:right;">7,168</td>
<td style="padding:8px 12px; text-align:right;">—</td>
<td style="padding:8px 12px; text-align:right; color:#e94560;">MoE</td>
<td style="padding:8px 12px;">SwiGLU</td>
<td style="padding:8px 12px; color:#888;">256 routed experts + 1 shared</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px; text-align:center; font-family:sans-serif;">
The theoretical 8/3 ≈ 2.667x ratio gets rounded up to 3.25x–3.50x for hardware alignment. The "extra" params over budget are a deliberate choice to give the FFN more capacity.
</div>
</div>

Real models don't use exactly $\frac{8}{3}$. Llama 3 and Mistral both use 3.5x, giving $3 \times d \times 3.5d = 10.5d^2$ FFN parameters — 30% **more** than the standard $8d^2$. Labs have decided the extra parameter spend is worth the quality gains. The $\frac{8}{3}$ ratio is the *iso-parameter* floor; practice goes higher.

---

## 6. Width vs. Depth Tradeoffs

[Deep Dive: Width vs Depth Experimental Evidence](excerpts/width-vs-depth.md) — collected evidence across scales, from the depth floor for reasoning through Llama 3's 126-layer design to MoE's decoupling of the tradeoff.

Given a fixed parameter budget, should you make each FFN wider (larger $d_{ff}$) or add more Transformer layers?

**Wider = more memory capacity per layer.** Under the FFN-as-memory hypothesis, each additional unit in $d_{ff}$ adds another key-value pair. A wider FFN can store more facts and patterns per layer. This benefits tasks that require recall of factual knowledge.

**Deeper = more compositional steps.** Each additional layer adds another round of (attention + FFN), allowing the model to compose more complex functions of the input. Reasoning, multi-step inference, and syntactic processing benefit from depth.

### Evidence from Real Architectures

**gpt-oss (2025)** made a notable bet on width over depth: its 20B model uses embedding dimension 2880 with only 24 layers, versus Qwen3 3B which uses embedding dimension 2048 with 48 layers. The gpt-oss designers explicitly chose "width over depth" — a wider model with fewer layers of composition.

**Llama 3** scales both dimensions but leans toward depth at the largest scale: the 405B model has 126 layers with $d_{model}$ = 16,384. The FFN dimension (53,248) maintains a 3.25x ratio, so relative width actually *decreases* at larger scale.

**DeepSeek-V3** ([[deepseek-v3|report]]) sidesteps the tradeoff entirely via Mixture-of-Experts: 256 experts per layer means the *effective* FFN width is enormous (256 expert FFNs available), but each token only activates 8 of them. This gives massive memory capacity without the compute cost of a fully-wide FFN. The FFN-as-memory interpretation maps directly onto MoE: more experts = more key-value memory slots, but routing selects which slots to query.

### The Empirical Picture

1. **Very shallow models fail at reasoning** — fewer than ~12 layers and multi-hop reasoning degrades, regardless of width.
2. **Very narrow models fail at recall** — if $d_{ff}$ is too small for the knowledge required, factual accuracy drops even with many layers.
3. **At the frontier, depth has higher marginal returns.** Llama 3's scaling experiments suggested adding layers was more compute-efficient than widening — hence the 126-layer 405B design.
4. **MoE decouples the tradeoff** by making width conditional: memory capacity of extreme width, compute cost of moderate width ([[ch-14]]).

---

## 7. FFN in the MoE Context

MoE replaces the single dense FFN with a collection of expert FFNs and a router. DeepSeek-V3 ([[deepseek-v3|report]]) illustrates the state of the art: 256 routed SwiGLU experts per layer, 8 active per token, plus 1 always-on shared expert. Total parameters: 671B; active per token: 37B.

This is the FFN-as-memory hypothesis made structural: different experts store different knowledge, the router selects which "memory bank" to query, and the emergent sparsity of ReLU FFNs becomes an architectural guarantee. You get the capacity of a $256 \times d_{expert}$-wide FFN at the compute cost of $8 \times d_{expert}$. Full treatment in [[ch-14]].

---

## Core Insights from the Literature

### Insight 1: The gating mechanism matters more than the specific activation function
**Paper:** Shazeer, "GLU Variants Improve Transformer" (2020) ([[glu-variants|paper]])

All GLU variants (ReGLU, GEGLU, SwiGLU) outperformed their non-gated counterparts on language modeling perplexity. The gap between gated and non-gated was larger than the gap between different gating activations. SwiGLU and GEGLU performed similarly, with SwiGLU marginally preferred. This suggests the multiplicative interaction created by gating — not the shape of the activation curve — is the primary source of improvement. **Guideline:** When designing FFN sublayers, the decision to gate or not gate matters more than which activation function to use inside the gate.

### Insight 2: FFN width is memory capacity, not just compute
**Papers:** Geva et al. "Transformer Feed-Forward Layers Are Key-Value Memories" (2021); Meng et al. "Locating and Editing Factual Associations in GPT" (ROME, 2022)

Individual rows of $W_1$ correspond to interpretable "key" patterns, and specific factual knowledge can be localized to individual FFN layers and edited via rank-one updates to $W_2$. The practical implication: when a model fails on factual recall, the bottleneck may be FFN width (not enough storage), not depth (not enough composition). **Guideline:** If your model underperforms on knowledge-intensive tasks, consider increasing $d_{ff}$ before adding layers. If it underperforms on reasoning, consider the reverse.

### Insight 3: The 8/3 ratio is a floor, not a ceiling
**Paper:** Llama 3 Technical Report (Meta, 2024) ([[llama-3|report]]); Mistral 7B (Mistral AI, 2023) ([[mistral-7b|report]])

Both Llama 3 and Mistral use a 3.5x ratio — about 30% more FFN parameters than the iso-parameter $\frac{8}{3}$ baseline would suggest. This means practitioners have decided the quality-parameter tradeoff favors allocating *extra* budget to the FFN beyond what simple parameter matching requires. The FFN's two-thirds share of total parameters has, if anything, grown with SwiGLU adoption. **Guideline:** Use $\frac{8}{3}d$ as the starting point for SwiGLU FFN width, but don't hesitate to go wider if parameter budget allows. Round to a multiple of 128 or 256 for hardware efficiency.

### Insight 4: MoE makes the width-depth tradeoff obsolete at scale
**Paper:** DeepSeek-V3 Technical Report (2024) ([[deepseek-v3|report]])

By replacing the dense FFN with 256 conditionally-activated expert FFNs, DeepSeek-V3 achieves the memory capacity of an enormously wide FFN while activating only 5.5% of total parameters per token. The sparse activation pattern that was *emergent* in ReLU FFNs becomes *structural* in MoE. This is the logical endpoint of the FFN-as-memory hypothesis: if the FFN is a memory and most keys don't match, don't compute them. **Guideline:** At scales beyond ~100B parameters, MoE is the practical way to increase FFN capacity without proportional compute cost. Dense models pay for width they mostly don't use.

---

## Key Takeaways

1. **The FFN consumes ~2/3 of model parameters** and functions as a position-wise key-value memory. Width ($d_{ff}$) directly controls how much factual knowledge each layer can store.

2. **The standard FFN formula $W_2 \cdot \sigma(W_1 x)$ evolved into the gated variant $(Swish(W_1 x) \odot Vx) W_2$**, adding a third matrix and a multiplicative gating branch.

3. **SwiGLU has won.** Every major frontier model (Llama 3, Mistral, PaLM, Gemma, DeepSeek-V3) uses it. The gating mechanism — not the specific activation — drives the improvement.

4. **The $\frac{8}{3}$ ratio is the iso-parameter baseline** for gated FFNs (3 matrices at $\frac{8}{3}d$ = same total params as 2 matrices at $4d$). In practice, models use 3.25x-3.50x, deliberately spending extra parameters on FFN capacity.

5. **Width stores knowledge; depth composes it.** When a model fails at factual recall, the FFN may be too narrow. When it fails at reasoning, it may lack depth. MoE decouples this tradeoff.

6. **Sparse FFN activation is the bridge to MoE.** ReLU FFNs naturally activate <10% of neurons. MoE makes this sparsity structural by routing tokens to a subset of expert FFNs, achieving massive effective width at modest compute cost ([[ch-14]]).

7. **Modern biases are eliminated.** Post-LLaMA convention drops all bias terms ($b_1, b_2$) from the FFN, relying on RMSNorm and the gating mechanism for expressiveness. If you see biases in an FFN, you're looking at a pre-2023 architecture.

---

## References

- [[attention-is-all-you-need|Vaswani et al. "Attention Is All You Need" (2017) (paper)]] — original FFN design, $d_{ff} = 4d_{model}$
- [[glu-variants|Shazeer, "GLU Variants Improve Transformer" (2020) (paper)]] — systematic comparison of gated activations
- Dauphin et al. "Language Modeling with Gated Convolutional Networks" (2017) — original GLU
- Geva et al. "Transformer Feed-Forward Layers Are Key-Value Memories" (2021)
- Meng et al. "Locating and Editing Factual Associations in GPT" (ROME, 2022)
- Ramachandran et al. "Searching for Activation Functions" (2017) — discovered Swish
- Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016)
- [[llama-3|Meta AI, "The Llama 3 Herd of Models" (2024) (report)]] — SwiGLU at scale, FFN ratios
- [[mistral-7b|Mistral AI, "Mistral 7B" (2023) (report)]] — SwiGLU + GQA at 7B scale
- [[deepseek-v3|DeepSeek AI, "DeepSeek-V3 Technical Report" (2024) (report)]] — MoE FFN with 256 experts
- [[raschka-llm-architecture-comparison|Sebastian Raschka, "The Big LLM Architecture Comparison" (2025) (blog)]] — convergence on SwiGLU
