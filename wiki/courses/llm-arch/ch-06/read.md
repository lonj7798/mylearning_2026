# Chapter 6: Positional Encoding

<!-- scope: why transformers need position information, evolution from sinusoidal to RoPE/ALiBi/iRoPE
     deps: [[ch-03]]
     see-also: [[ch-07]], [[ch-16]]
-->

## Overview

Self-attention is a set operation. Given a bag of token embeddings, the attention mechanism computes pairwise dot products and weighted sums — and the result is identical regardless of the order the tokens arrived in. Formally, attention is **permutation-equivariant**: permuting the input permutes the output in the same way, but the actual values are unchanged. Without positional encoding, "The dog bit the man" and "The man bit the dog" produce identical representations for every shared token.

This chapter traces the full arc of how the field solved this problem: from the original sinusoidal encodings in [[ch-03]] ([[attention-is-all-you-need|Attention Is All You Need (paper)]]), through learned embeddings (GPT-2/3), to the rotary formulation (RoPE, [[rope|paper]]) that dominates modern LLMs, the zero-parameter alternative (ALiBi, [[alibi|paper]]), and the cutting-edge interleaved approach (iRoPE) that enabled Llama 4's ([[llama-4|report]]) 10-million-token context window. The mathematical depth increases as we go — RoPE's derivation from first principles in complex space is the centerpiece.

Position encoding is not a minor implementation detail. It is the primary mechanism that determines whether a model can generalize to sequence lengths beyond training, how it represents locality versus long-range dependencies, and ultimately how much context it can effectively use. Every major context-length breakthrough (4K to 32K to 128K to 10M tokens) was driven by a positional encoding innovation.

---

## 1. Why Position Information Is Needed

Self-attention computes, for each token $i$:

$$\text{Attn}(Q, K, V)_i = \sum_j \text{softmax}\left(\frac{q_i^T k_j}{\sqrt{d_k}}\right) v_j$$

The dot product $q_i^T k_j$ depends only on the **content** of tokens $i$ and $j$, not their positions. If you swap the positions of two tokens in the input, the attention weights and outputs permute accordingly but the values themselves don't change. This is the "Dog Problem" from the HF blog ([[hf-positional-encoding-design|blog]]): in "The dog chased another dog," both instances of "dog" produce identical query, key, and value vectors — the model literally cannot tell them apart.

Compare this to RNNs, which process tokens sequentially and encode position implicitly through the hidden state's trajectory. Convolutions encode position through their receptive field structure. Attention must be **explicitly told** where each token sits.

The design space for "how to tell it" splits along two axes:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Design Axes for Positional Encoding</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px 12px;"></th>
<th style="text-align:center; padding:8px 12px; color:#e94560;">Absolute</th>
<th style="text-align:center; padding:8px 12px; color:#e94560;">Relative</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px; font-weight:bold; color:#e94560;">Additive</td>
<td style="padding:8px 12px; text-align:center;">Sinusoidal (2017)<br/>Learned (GPT-2, 2019)</td>
<td style="padding:8px 12px; text-align:center;">T5 RPE (2020)<br/>ALiBi (2022)</td>
</tr>
<tr>
<td style="padding:8px 12px; font-weight:bold; color:#e94560;">Multiplicative</td>
<td style="padding:8px 12px; text-align:center;">—</td>
<td style="padding:8px 12px; text-align:center;">RoPE (2021)<br/>iRoPE (2025)</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">Additive = added to embeddings or attention logits. Multiplicative = applied as rotation to Q/K vectors.</div>
</div>

The history of positional encoding is a march from the upper-left cell to the lower-right: from absolute-additive to relative-multiplicative. Understanding **why** requires understanding what goes wrong with each predecessor.

---

## 2. Sinusoidal Positional Encoding (2017)

The original Transformer added a fixed, deterministic positional signal to token embeddings before they entered the attention layers:

$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad \text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Each dimension pair $(2i, 2i+1)$ oscillates at a different frequency $\omega_i = 1/10000^{2i/d}$. Low-indexed dimensions cycle rapidly (capturing fine-grained local position); high-indexed dimensions cycle slowly (capturing global position). The geometric progression of wavelengths from $2\pi$ to $\sim 2\pi \times 10000$ gives approximately 63,000 unique position encodings.

### The Evolutionary Path: Why Sinusoidal?

The HF blog walks through the design evolution that makes the choice feel inevitable:

**Attempt 1 — Integer encoding:** Add the position index (0, 1, 2, ...) directly. Fails immediately: position magnitudes swamp embedding values, and the encoding is sequence-length-dependent (position 512 means different things in a 1024-token vs 2048-token sequence).

**Attempt 2 — Binary encoding:** Represent position in binary, spread across dimensions. The LSB flips every token, MSB flips every $2^{n-1}$ tokens. This gives unique, length-independent codes — but the discrete jumps are hostile to gradient-based optimization.

**Attempt 3 — Sinusoidal:** Replace the binary step functions with smooth sinusoids at the same frequencies. Each dimension pair encodes a "bit" as a continuous rotation instead of a discrete flip. Satisfies uniqueness, length-independence, and smoothness simultaneously.

### The Hidden Rotation Matrices

Here is the critical observation that took four years to fully exploit. Given sinusoidal encoding at position $p$, can we find a matrix $M_k$ that shifts it to position $p + k$?

$$M_k \begin{bmatrix} \sin(\omega_i \cdot p) \\ \cos(\omega_i \cdot p) \end{bmatrix} = \begin{bmatrix} \sin(\omega_i \cdot (p+k)) \\ \cos(\omega_i \cdot (p+k)) \end{bmatrix}$$

Applying the angle addition formulas and matching coefficients:

$$M_k = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix}$$

This is a 2D rotation matrix. The relative position offset $k$ is encoded as a rotation. **The rotation matrices were present in the sinusoidal formulation since 2017** — but Vaswani et al. used them additively (added to embeddings), not multiplicatively (applied to Q/K). The conceptual leap to multiplicative application took until 2021.

### Practical Limits of Sinusoidal Encoding

Despite theoretical extrapolation properties ($M_k$ is defined for any $k$), sinusoidal encodings have a fundamental problem: they are **added** to the token embedding before projection into Q, K, V:

$$\text{input} = \text{token\_embedding} + \text{PE}$$

This creates **semantic pollution** — positional information contaminates the semantic signal. The model must learn to disentangle position from meaning throughout every layer. Additionally, the vector norm changes unpredictably when two unrelated vectors are summed, making the signal-to-noise ratio position-dependent.

---

## 3. Learned Positional Embeddings (GPT-2/3)

GPT-2 and GPT-3 replaced fixed sinusoidal functions with a learned embedding table $W_p \in \mathbb{R}^{L \times d}$, where $L$ is the maximum sequence length:

$$\text{input}_t = \text{token\_embedding}_t + W_p[t]$$

The position embedding for each position $t$ is a learned $d$-dimensional vector, trained end-to-end with the rest of the model. This is identical in mechanism to a token embedding lookup — positions are just "special tokens."

### Why GPT-2/3 Chose Learned Embeddings

**Flexibility:** The model can learn arbitrary position-dependent patterns rather than being constrained to sinusoidal structure. In practice, learned embeddings capture subtle position-dependent biases in the training data (e.g., the first token of a document tends to behave differently from the 500th).

**Simplicity:** One embedding lookup, same as token embeddings. No special implementation needed.

### Why Learned Embeddings Were Abandoned

**Hard length ceiling:** A model trained with $L = 1024$ has exactly 1024 position embeddings. Position 1025 simply does not exist. There is no principled way to extrapolate — you can interpolate (average nearby embeddings) or extrapolate (continue the pattern), but both are heuristics that degrade rapidly.

**No structural prior:** Learned embeddings treat each position as independent. The model must learn from scratch that position 100 is "near" position 101 and "far" from position 900. Sinusoidal and rotary encodings encode this structure for free.

**Parameter cost:** For GPT-3 with $L = 2048$ and $d = 12288$, the position embedding table is $\sim 25M$ parameters — small relative to 175B total, but still a waste of capacity on something that can be computed analytically.

The field moved away from learned embeddings once context-length extension became a priority. You cannot extend what doesn't generalize.

---

## 4. Rotary Position Embedding (RoPE) — The Deep Dive

RoPE is the positional encoding used by LLaMA 1/2/3, Mistral, Qwen, GPT-NeoX, PaLM, and essentially every modern open-weight LLM. Understanding it thoroughly is non-negotiable for architecture research.

### 4.1 The Core Idea

Instead of adding positional information to the input embedding (polluting semantics), RoPE applies a **position-dependent rotation** to the query and key vectors **after** projection, right before the dot product. Position information enters exactly where it matters — in the $QK^T$ attention computation — and nowhere else.

### 4.2 First-Principles Derivation ([[eleutherai-rope|EleutherAI blog]])

We want a function $f(\mathbf{x}, m)$ that encodes position $m$ into vector $\mathbf{x}$ such that the inner product between two encoded vectors depends only on their relative position:

$$\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)$$

**Step 1: Work in complex space.** Pair consecutive dimensions: $(x_1, x_2) \to x_1 + ix_2$. Now we work in $\mathbb{C}^{d/2}$.

**Step 2: Polar decomposition.** Write $f(\mathbf{q}, m) = R_f(\mathbf{q}, m) \cdot e^{i\Theta_f(\mathbf{q}, m)}$ where $R_f$ is the magnitude and $\Theta_f$ is the phase.

**Step 3: Exploit the inner product constraint.** The inner product of two complex numbers gives:
- Magnitude: $R_f(\mathbf{q}, m) \cdot R_f(\mathbf{k}, n) = R_g(\mathbf{q}, \mathbf{k}, m - n)$
- Phase: $\Theta_f(\mathbf{q}, m) - \Theta_f(\mathbf{k}, n) = \Theta_g(\mathbf{q}, \mathbf{k}, m - n)$

**Step 4: Magnitude is position-independent.** Setting $m = n$ with $f(\mathbf{x}, 0) = \mathbf{x}$ shows $R_f(\mathbf{x}, m) = |\mathbf{x}|$ for all $m$. Position encoding cannot change the vector's magnitude — only its angle.

**Step 5: Phase decomposes additively.** The phase constraint requires $\Theta_f(\mathbf{q}, m) - \Theta_f(\mathbf{k}, n)$ to depend only on $m - n$. The unique solution is $\Theta_f(\mathbf{x}, m) = \Theta(\mathbf{x}) + m\theta$ — a content-dependent base angle plus a position-proportional rotation.

**Final formula:**

$$f(\mathbf{q}, m) = \sum_{j=1}^{d/2} q_j \cdot e^{i(m\theta_j)} \cdot \mathbf{e}_j$$

where $\theta_j = 10000^{-2j/d}$ (the same frequency schedule as sinusoidal encoding).

This is not a design choice — it is the **unique solution** to the constraint "inner product depends only on relative position." The rotation matrices are forced by the math.

### 4.3 Matrix Form

In real-valued coordinates, RoPE applies a block-diagonal rotation matrix to each query/key vector:

$$R_{\Theta, m} = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & & & \\ \sin m\theta_1 & \cos m\theta_1 & & & \\ & & \cos m\theta_2 & -\sin m\theta_2 & \\ & & \sin m\theta_2 & \cos m\theta_2 & \\ & & & & \ddots \end{pmatrix}$$

Each $2 \times 2$ block independently rotates a pair of dimensions by angle $m\theta_j$. The block-diagonal structure means dimensions are coupled in pairs but independent across pairs.

The critical property:

$$(\mathbf{R}_{\Theta,m} \mathbf{q})^T (\mathbf{R}_{\Theta,n} \mathbf{k}) = \mathbf{q}^T \mathbf{R}_{\Theta,m}^T \mathbf{R}_{\Theta,n} \mathbf{k} = \mathbf{q}^T \mathbf{R}_{\Theta,n-m} \mathbf{k}$$

Because $R^T(m) R(n) = R(n - m)$ for rotation matrices. The dot product depends only on the relative position $(n - m)$.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">RoPE Rotation Geometry (Single Dimension Pair)</div>
<div style="display:flex; justify-content:center; gap:60px; flex-wrap:wrap; align-items:flex-start;">
<div style="text-align:center;">
<div style="width:180px; height:180px; border:2px solid #444; border-radius:50%; position:relative; margin:0 auto;">
<div style="position:absolute; top:50%; left:50%; width:1px; height:1px;">
<div style="position:absolute; width:70px; height:2px; background:linear-gradient(90deg, #e94560, #e94560); transform-origin:left center; transform:rotate(-20deg); top:-1px; left:0;"></div>
<div style="position:absolute; width:70px; height:2px; background:linear-gradient(90deg, #4ecdc4, #4ecdc4); transform-origin:left center; transform:rotate(-50deg); top:-1px; left:0;"></div>
<div style="position:absolute; width:70px; height:2px; background:linear-gradient(90deg, #f7dc6f, #f7dc6f); transform-origin:left center; transform:rotate(-80deg); top:-1px; left:0;"></div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:8px;">
<span style="color:#e94560;">&#9632;</span> q at pos m &nbsp;
<span style="color:#4ecdc4;">&#9632;</span> q at pos m+1 &nbsp;
<span style="color:#f7dc6f;">&#9632;</span> q at pos m+2
</div>
<div style="color:#666; font-size:11px; margin-top:4px;">Each position rotates by angle theta_j</div>
</div>
<div style="text-align:center; max-width:280px;">
<div style="color:#e0e0e0; font-size:12px; text-align:left; line-height:1.8;">
<div><span style="color:#e94560; font-weight:bold;">Key insight:</span> Rotation preserves magnitude</div>
<div>||R(m) q|| = ||q|| for all m</div>
<div style="margin-top:8px;"><span style="color:#4ecdc4; font-weight:bold;">Relative encoding:</span></div>
<div>dot(R(m)q, R(n)k) = dot(q, R(n-m)k)</div>
<div style="margin-top:8px;"><span style="color:#f7dc6f; font-weight:bold;">Natural decay:</span></div>
<div>Large |m-n| = large angle = smaller cosine = lower attention weight</div>
</div>
</div>
</div>
</div>

### 4.4 Efficient Implementation

The matrix form looks expensive, but the block-diagonal structure admits an element-wise trick:

```python
def apply_rope(q, k, position_ids):
    d = q.shape[-1]
    inv_freq = 1.0 / (10000 ** (2.0 * torch.arange(0, d, 2) / d))
    t = position_ids.unsqueeze(-1) * inv_freq  # [seq_len, d/2]
    cos = t.cos()  # [seq_len, d/2]
    sin = t.sin()  # [seq_len, d/2]
    
    # Equivalent to block-diagonal rotation matrix multiplication
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)
```

**Computational cost:** With Torchscript fusion, RoPE adds 1-3% overhead to the full transformer forward pass. Essentially free.

### 4.5 Why RoPE Works Better Than Sinusoidal

Two differences explain the performance gap, despite sharing the same frequency schedule:

1. **Multiplicative vs additive application.** Sinusoidal adds $\text{PE}$ to the embedding, polluting the semantic signal. RoPE rotates Q and K, preserving their norms ($\|R\mathbf{q}\| = \|\mathbf{q}\|$). The semantic magnitude is untouched; only the angular relationship changes.

2. **Position in the right place.** Sinusoidal encoding is applied at the input, before Q/K/V projection. The position signal must survive multiple linear transformations. RoPE is applied directly to Q and K, right before the dot product — position information enters exactly where it's used.

### 4.6 The Extrapolation Problem

Despite the elegant math, RoPE has a practical limitation: models trained at context length $L$ degrade at lengths $L' \gg L$. The issue is not the rotation formulation itself but the **frequency distribution.** High-frequency RoPE dimensions (small $\theta_j$) experience position values outside their training distribution when $m > L$. The model has never seen these rotation angles and cannot reliably use them.

This motivated a family of context-extension methods — most notably YaRN (Section 6).

---

## 5. ALiBi: Attention with Linear Biases

ALiBi takes a radically different approach: **remove positional embeddings entirely** and inject position information as a bias on the attention logits.

### 5.1 Mechanism

$$\text{attention}(q_i, k_j) = \frac{q_i^T k_j}{\sqrt{d_k}} - m_h \cdot |i - j|$$

where $m_h$ is a head-specific slope. No positional embedding is added to the input at any point. Token embeddings are purely semantic. Position enters only as a linear penalty on distance in the attention score.

### 5.2 Head-Specific Slopes

Each attention head $h$ (of $H$ total) gets a different slope from a geometric sequence:

$$m_h = 2^{-8h/H}, \quad h = 1, \ldots, H$$

For 8 heads: $m = \frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}, \frac{1}{64}, \frac{1}{128}, \frac{1}{256}$

Steeper slopes (larger $m_h$) create heads that focus almost exclusively on nearby tokens. Shallower slopes create heads with broader, more global attention. This automatically produces a **multi-scale attention pattern** — some heads are local, some are global — without any learned parameters.

### 5.3 Why ALiBi Extrapolates

The bias $-m_h \cdot |i - j|$ is defined for any distance $|i - j|$, including distances never seen during training. There are no learned embeddings that become undefined at novel positions. The linear penalty simply continues — a position 10,000 tokens away gets a proportionally larger penalty. This enabled training at length 1024 and testing at length 2048 with no quality loss (11% faster training, 11% less memory than sinusoidal).

### 5.4 RoPE vs ALiBi: The Tradeoff

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">RoPE vs ALiBi Comparison</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px 12px;">Dimension</th>
<th style="text-align:left; padding:8px 12px;">RoPE</th>
<th style="text-align:left; padding:8px 12px;">ALiBi</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px; font-weight:bold;">Where position enters</td>
<td style="padding:8px 12px;">Rotation of Q, K vectors</td>
<td style="padding:8px 12px;">Bias on attention logits</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px; font-weight:bold;">Type</td>
<td style="padding:8px 12px;">Relative (multiplicative)</td>
<td style="padding:8px 12px;">Relative (additive bias)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px; font-weight:bold;">Learned parameters</td>
<td style="padding:8px 12px;">Zero (fixed frequencies)</td>
<td style="padding:8px 12px;">Zero (fixed slopes)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px; font-weight:bold;">Position info in V?</td>
<td style="padding:8px 12px;">No (Q, K only)</td>
<td style="padding:8px 12px;">No (attention scores only)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px; font-weight:bold;">Native extrapolation</td>
<td style="padding:8px 12px;">Limited (needs extension methods)</td>
<td style="padding:8px 12px;">Good (linear bias generalizes)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px; font-weight:bold;">Recency bias</td>
<td style="padding:8px 12px;">Soft (decaying dot product)</td>
<td style="padding:8px 12px;">Strong (linear penalty, unbounded)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px; font-weight:bold;">Long-range tasks</td>
<td style="padding:8px 12px;">Better (preserves global attention)</td>
<td style="padding:8px 12px;">Weaker (linear penalty suppresses)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px 12px; font-weight:bold;">Context extension</td>
<td style="padding:8px 12px;">Rich ecosystem (YaRN, NTK, PI)</td>
<td style="padding:8px 12px;">Limited (no analog methods)</td>
</tr>
<tr>
<td style="padding:8px 12px; font-weight:bold;">Adoption (2025)</td>
<td style="padding:8px 12px;">Dominant (LLaMA, Mistral, Qwen)</td>
<td style="padding:8px 12px;">Niche (MPT, BLOOM)</td>
</tr>
</tbody>
</table>
</div>

The bottom line: ALiBi's simplicity is appealing, but its strong recency bias is a liability for tasks requiring long-range uniform attention (retrieval, global reasoning, code understanding). RoPE's flexibility and compatibility with context-extension methods made it the industry default. ALiBi has been **largely superseded** — the ALiBi paper itself acknowledges RoPE's advantages for modern use cases.

---

## 6. YaRN ([[yarn|paper]]): Extending RoPE's Context Window

RoPE's extrapolation problem created an entire subfield of context-extension research. YaRN (Yet another RoPE extensioN) represents the current best practice.

### 6.1 The Problem

RoPE's frequencies are $\theta_j = 10000^{-2j/d}$. A model trained at length $L$ has only seen rotation angles $m \cdot \theta_j$ for $m \in [0, L)$. At length $L' > L$, high-frequency dimensions (large $\theta_j$) encounter angles outside their training distribution. The model's learned attention patterns break down.

### 6.2 Position Interpolation (Baseline)

The simplest fix: scale all positions by $L/L'$, so position $m$ becomes $m \cdot L/L'$. All angles stay within $[0, L \cdot \theta_j)$ — the training distribution. But this **compresses** positional resolution uniformly, making nearby tokens harder to distinguish.

### 6.3 NTK-Aware Interpolation (YaRN's Key Insight)

Not all frequencies need the same treatment:
- **High-frequency dimensions** encode local position (nearby token distinctions). They rarely see out-of-distribution values even at extended lengths. Don't touch them.
- **Low-frequency dimensions** encode global position (where in the sequence). They hit out-of-distribution angles first. Interpolate aggressively.

YaRN modifies the base frequency:

$$\text{base}' = \text{base} \cdot \left(\frac{L'}{L}\right)^{d/(d-2)}$$

This effectively applies frequency-dependent scaling: low frequencies are compressed (interpolated) while high frequencies are largely preserved.

### 6.4 Attention Temperature Scaling

Extending context increases the number of positions the attention mechanism must distribute probability over, increasing entropy. YaRN compensates with a temperature scaling factor $t$:

$$\text{softmax}\left(\frac{q^T k}{\sqrt{d_k} \cdot t}\right)$$

This sharpens the attention distribution to counteract the dilution from more positions.

### 6.5 The Recipe

Starting from a pretrained RoPE model: apply NTK-aware frequency scaling, add attention temperature scaling, fine-tune for ~400 steps on long-context data. This achieves effective extension (e.g., 4K to 128K) with 10x fewer tokens and 2.5x fewer steps than prior methods. YaRN models can even extrapolate beyond their fine-tuning length — a model fine-tuned at 64K maintains quality at 128K.

---

## 7. iRoPE: Interleaved RoPE (Llama 4)

Llama 4 introduced the most radical positional encoding innovation since RoPE itself: **interleave layers with and without positional encoding.**

### 7.1 The Design

Instead of applying RoPE to every attention layer, iRoPE alternates:
- **RoPE layers:** Standard rotary encoding, providing explicit position information
- **No-PE layers:** No positional encoding at all — pure content-based attention

Combined with inference-time temperature scaling for length generalization, this enabled Scout to generalize from a 256K training context to a **10-million-token** inference context.

### 7.2 Why It Works

The intuition is that not all layers need position information equally. Some layers benefit from pure content-based reasoning (semantic similarity, entity matching) where position is irrelevant or even harmful. Other layers need position to establish local structure (syntax, coreference). By letting the model have both, iRoPE avoids the overhead of position encoding in layers that don't need it while preserving it where it matters.

This connects to a DeepMind finding cited in the HF blog: models primarily use lower RoPE frequencies, and **removing the lowest frequencies actually improves performance** on some tasks. iRoPE takes this further — entire layers operate without any positional signal.

### 7.3 Implications for Context Length

The 256K-to-10M extrapolation ratio (~40x) is far beyond what any prior method achieved. For comparison, ALiBi demonstrated 2x extrapolation (1024 to 2048), and YaRN validated up to ~32x (4K to 128K). iRoPE suggests that **less positional encoding, strategically applied, enables better length generalization** — the model learns to rely on content-based attention for long-range dependencies and uses position encoding only for local structure.

---

## 8. The Full Evolution

<div style="background:#1a1a2e; border-radius:12px; padding:28px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:20px; font-family:sans-serif; font-weight:bold;">Positional Encoding Evolution Timeline</div>
<div style="position:relative; padding-left:40px;">

<div style="position:absolute; left:18px; top:0; bottom:0; width:3px; background:linear-gradient(180deg, #e94560, #4ecdc4, #f7dc6f, #a29bfe, #fd79a8);"></div>

<div style="position:relative; margin-bottom:28px;">
<div style="position:absolute; left:-32px; top:4px; width:14px; height:14px; border-radius:50%; background:#e94560;"></div>
<div style="color:#e94560; font-weight:bold; font-size:13px;">2017 — Sinusoidal (Vaswani et al.)</div>
<div style="color:#aaa; font-size:12px; margin-top:4px;">Fixed sin/cos added to embeddings. First principled approach. Rotation matrices hidden in the math.</div>
<div style="color:#666; font-size:11px;">Used by: Original Transformer, BERT</div>
</div>

<div style="position:relative; margin-bottom:28px;">
<div style="position:absolute; left:-32px; top:4px; width:14px; height:14px; border-radius:50%; background:#e94560; opacity:0.7;"></div>
<div style="color:#e94560; font-weight:bold; font-size:13px; opacity:0.85;">2019 — Learned Embeddings (Radford et al.)</div>
<div style="color:#aaa; font-size:12px; margin-top:4px;">Trainable position lookup table. Flexible but hard ceiling on context length.</div>
<div style="color:#666; font-size:11px;">Used by: GPT-2, GPT-3</div>
</div>

<div style="position:relative; margin-bottom:28px;">
<div style="position:absolute; left:-32px; top:4px; width:14px; height:14px; border-radius:50%; background:#4ecdc4;"></div>
<div style="color:#4ecdc4; font-weight:bold; font-size:13px;">2021 — RoPE (Su et al.)</div>
<div style="color:#aaa; font-size:12px; margin-top:4px;">Multiplicative rotation encoding relative position. The breakthrough: derived from first principles, not designed.</div>
<div style="color:#666; font-size:11px;">Used by: LLaMA 1/2/3, Mistral, Qwen, PaLM, GPT-NeoX, Falcon</div>
</div>

<div style="position:relative; margin-bottom:28px;">
<div style="position:absolute; left:-32px; top:4px; width:14px; height:14px; border-radius:50%; background:#f7dc6f;"></div>
<div style="color:#f7dc6f; font-weight:bold; font-size:13px;">2022 — ALiBi (Press et al.)</div>
<div style="color:#aaa; font-size:12px; margin-top:4px;">Zero-parameter linear bias on attention. Best native extrapolation, but strong recency bias limits adoption.</div>
<div style="color:#666; font-size:11px;">Used by: BLOOM, MPT</div>
</div>

<div style="position:relative; margin-bottom:28px;">
<div style="position:absolute; left:-32px; top:4px; width:14px; height:14px; border-radius:50%; background:#a29bfe;"></div>
<div style="color:#a29bfe; font-weight:bold; font-size:13px;">2023 — YaRN (Peng et al.)</div>
<div style="color:#aaa; font-size:12px; margin-top:4px;">NTK-aware frequency scaling + temperature. Extends RoPE models 10-32x with ~400 fine-tuning steps.</div>
<div style="color:#666; font-size:11px;">Used by: extended LLaMA variants, Mixtral long-context</div>
</div>

<div style="position:relative;">
<div style="position:absolute; left:-32px; top:4px; width:14px; height:14px; border-radius:50%; background:#fd79a8;"></div>
<div style="color:#fd79a8; font-weight:bold; font-size:13px;">2025 — iRoPE (Meta / Llama 4)</div>
<div style="color:#aaa; font-size:12px; margin-top:4px;">Interleaved RoPE/no-PE layers. 40x extrapolation ratio (256K train to 10M inference).</div>
<div style="color:#666; font-size:11px;">Used by: Llama 4 Scout, Llama 4 Maverick</div>
</div>

</div>
</div>

---

## Core Insights from the Literature

### Insight 1: The rotation matrices were hiding in plain sight for four years
**Paper:** Su et al. "RoFormer" ([[rope|2021 (paper)]]), reconstructed in [[hf-positional-encoding-design|HF blog]]

The sinusoidal encoding from 2017 contained 2D rotation matrices as a mathematical consequence of the angle addition theorem. But the original Transformer used them additively (add PE to embeddings), which pollutes semantic information and prevents the relative-position property from cleanly emerging. The conceptual leap was not mathematical — it was **architectural**: apply the rotations multiplicatively to Q and K, right before the dot product. The math was there all along; the insight was where to apply it. **Guideline:** When improving a component, look at existing math for structural properties that the current implementation fails to exploit.

### Insight 2: Uniqueness of the rotary solution
**Paper:** EleutherAI "Rotary Embeddings: A Relative Revolution" ([[eleutherai-rope|blog]])

RoPE is not one design among many — it is the **unique solution** to the constraint "inner product depends only on relative position" in complex space. The derivation proceeds from three constraints (relative position dependence, position-independent magnitude, initial condition) and admits exactly one family of solutions: position-proportional rotation with content-dependent base phase. This is why RoPE has been so hard to improve upon: you cannot satisfy the same constraints with a fundamentally different mechanism. **Guideline:** When a method is derived from first principles with a uniqueness proof, improving it means relaxing the constraints (as iRoPE does by removing PE from some layers), not finding a "better" solution to the same constraints.

### Insight 3: Less positional encoding can mean better generalization
**Paper:** Meta "Llama 4" ([[llama-4|2025 (report)]]), supported by DeepMind findings on RoPE frequency usage

Llama 4's iRoPE removes positional encoding from alternating layers and achieves 40x length extrapolation. DeepMind independently found that models primarily use lower RoPE frequencies and that removing the lowest frequencies improves some tasks. The emerging principle: positional encoding is a form of inductive bias, and **over-encoding position can hurt** generalization, especially at long contexts. Layers that operate on pure content-based attention may better capture long-range semantic dependencies that don't depend on exact position. **Guideline:** When designing for extreme context lengths, consider reducing positional encoding rather than strengthening it.

### Insight 4: Frequency-dependent treatment is the key to context extension
**Paper:** Peng et al. "YaRN" ([[yarn|2023 (paper)]])

YaRN's insight is that high-frequency and low-frequency RoPE dimensions serve different purposes and have different out-of-distribution vulnerabilities. High-frequency dimensions encode local position (nearby token distinctions) and rarely see novel values at moderate extension ratios. Low-frequency dimensions encode global position and hit OOD angles first. Treating all frequencies uniformly (as position interpolation does) unnecessarily degrades local position resolution. **Guideline:** Context extension methods should scale frequencies non-uniformly — preserve high frequencies (local structure) and interpolate low frequencies (global position).

### Insight 5: ALiBi proves that position can live entirely in attention bias
**Paper:** Press et al. "Train Short, Test Long" ([[alibi|2022 (paper)]])

ALiBi demonstrated a surprising result: you can completely remove positional embeddings from the model input and encode position solely as a bias on attention scores. The token embeddings become purely semantic — no positional contamination at any layer. This "separation of concerns" principle influenced later designs even though ALiBi itself was superseded. The geometric slope schedule ($m_h = 2^{-8h/H}$) creates an automatic multi-scale attention pattern that requires zero learned parameters. **Guideline:** Position and semantics are fundamentally different signals; architectures that cleanly separate them (RoPE on Q/K only, ALiBi as a bias) outperform those that mix them (additive PE on embeddings).

---

## Key Takeaways

1. **Self-attention is permutation-equivariant.** Without explicit position encoding, a transformer treats its input as a set, not a sequence. This is the foundational problem that all positional encoding methods address.

2. **The evolution from additive to multiplicative, absolute to relative, is not arbitrary.** Each step solves a concrete failure: semantic pollution (fixed by multiplicative), length ceiling (fixed by relative), information loss (fixed by rotation in complex space).

3. **RoPE is the unique solution** to "make the QK dot product depend only on relative position" — derived from first principles, not designed by intuition. Its universal adoption (LLaMA, Mistral, Qwen, PaLM) reflects this mathematical inevitability.

4. **ALiBi trades flexibility for simplicity.** Its zero-parameter linear bias extrapolates well but imposes a strong recency bias that limits long-range reasoning. RoPE's richer encoding won the adoption race.

5. **Context extension is a frequency-domain problem.** YaRN's success comes from recognizing that high-frequency (local) and low-frequency (global) RoPE dimensions need different treatment when extending beyond training length.

6. **Less position encoding can be more.** Llama 4's iRoPE — alternating PE and no-PE layers — achieves 40x length extrapolation, suggesting that over-encoding position hurts generalization at extreme context lengths.

7. **Position information should enter where it's used.** Additive PE at the input must survive multiple transformations. RoPE on Q/K and ALiBi on attention scores inject position exactly where the model consumes it — a recurring principle in modern architecture design.

---

## References

- [[attention-is-all-you-need|Vaswani et al. "Attention Is All You Need" (2017) (paper)]] — sinusoidal positional encoding, [[ch-03]]
- Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019) — learned positional embeddings
- [[rope|Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) (paper)]] — RoPE
- [[alibi|Press, Smith, Lewis "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (2022) (paper)]] — ALiBi
- [[yarn|Peng et al. "YaRN: Efficient Context Window Extension of Large Language Models" (2023) (paper)]]
- [[llama-4|Meta AI "Llama 4: The Beginning of a New Era of Natively Multimodal AI" (2025) (report)]] — iRoPE
- [[eleutherai-rope|Biderman et al. "Rotary Embeddings: A Relative Revolution" (EleutherAI blog)]] — first-principles derivation
- [[hf-positional-encoding-design|Fleetwood "You Could Have Designed State of the Art Positional Encoding" (HF blog)]] — evolutionary walkthrough
